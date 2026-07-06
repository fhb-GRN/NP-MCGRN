from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import APPNP
from torch_scatter import scatter_add


_MAX_NORM = 85.0


def _safe_cosh(x: Tensor) -> Tensor:
    return torch.cosh(torch.clamp(x, min=-_MAX_NORM, max=_MAX_NORM))


def _safe_sinh(x: Tensor) -> Tensor:
    return torch.sinh(torch.clamp(x, min=-_MAX_NORM, max=_MAX_NORM))


def _expand_proj_dims(x: Tensor) -> Tensor:
    zeros = torch.zeros(x.shape[:-1] + torch.Size([1]), device=x.device, dtype=x.dtype)
    return torch.cat((zeros, x), dim=-1)


class RadiusManifold:
    def __init__(self, radius: Callable[[], Tensor]) -> None:
        self._radius = radius

    @property
    def radius(self) -> Tensor:
        return torch.clamp(torch.relu(self._radius()), min=1e-8, max=1e8)

    def exp_map_mu0(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Euclidean:
    def exp_map_mu0(self, x: Tensor) -> Tensor:
        return x / 2


class Hyperboloid(RadiusManifold):
    def exp_map_mu0(self, x: Tensor) -> Tensor:
        x = _expand_proj_dims(x)
        radius = self.radius
        tangent = x[..., 1:]
        tangent_norm = torch.norm(tangent, p=2, keepdim=True, dim=-1) / radius
        tangent_unit = F.normalize(tangent, p=2, dim=-1) * radius
        mapped = torch.cat(
            (_safe_cosh(tangent_norm) * radius, _safe_sinh(tangent_norm) * tangent_unit),
            dim=-1,
        )
        if not torch.isfinite(mapped).all():
            raise FloatingPointError("Non-finite hyperbolic component output")
        return mapped


class Sphere(RadiusManifold):
    def exp_map_mu0(self, x: Tensor) -> Tensor:
        x = _expand_proj_dims(x)
        radius = self.radius
        tangent = x[..., 1:]
        tangent_norm = torch.norm(tangent, p=2, keepdim=True, dim=-1) / radius
        tangent_unit = F.normalize(tangent, p=2, dim=-1) * radius
        mapped = torch.cat(
            (torch.cos(tangent_norm) * radius, torch.sin(tangent_norm) * tangent_unit),
            dim=-1,
        )
        if not torch.isfinite(mapped).all():
            raise FloatingPointError("Non-finite spherical component output")
        return mapped


class Component(nn.Module):
    def __init__(self, dim: int, fixed_curvature: bool) -> None:
        super().__init__()
        self.dim = dim
        self.fixed_curvature = fixed_curvature
        self.manifold = None
        self.fc_mean: Optional[nn.Linear] = None
        self.fc_logvar: Optional[nn.Linear] = None

    def init_layers(self, in_dim: int, scalar_parametrization: bool) -> None:
        self.manifold = self.create_manifold()
        self.fc_mean = nn.Linear(in_dim, self.mean_dim)
        self.fc_logvar = nn.Linear(in_dim, 1 if scalar_parametrization else self.true_dim)

    def create_manifold(self):
        raise NotImplementedError

    @property
    def true_dim(self) -> int:
        raise NotImplementedError

    @property
    def mean_dim(self) -> int:
        return self.true_dim


class EuclideanComponent(Component):
    def __init__(self, dim: int, fixed_curvature: bool) -> None:
        super().__init__(dim, fixed_curvature=True)

    def create_manifold(self) -> Euclidean:
        return Euclidean()

    @property
    def true_dim(self) -> int:
        return self.dim


class HyperbolicComponent(Component):
    def __init__(self, dim: int, fixed_curvature: bool, radius: float = 1.0) -> None:
        super().__init__(dim + 1, fixed_curvature)
        self._nradius = nn.Parameter(torch.tensor(radius), requires_grad=not fixed_curvature)

    def create_manifold(self) -> Hyperboloid:
        return Hyperboloid(lambda: self._nradius)

    @property
    def true_dim(self) -> int:
        return self.dim - 1


class SphericalComponent(Component):
    def __init__(self, dim: int, fixed_curvature: bool, radius: float = 1.0) -> None:
        super().__init__(dim + 1, fixed_curvature)
        self._pradius = nn.Parameter(torch.tensor(radius), requires_grad=not fixed_curvature)

    def create_manifold(self) -> Sphere:
        return Sphere(lambda: self._pradius)

    @property
    def true_dim(self) -> int:
        return self.dim - 1


_SPACE_CREATOR_MAP: Dict[str, Callable[[int, bool], Component]] = {
    "e": EuclideanComponent,
    "h": HyperbolicComponent,
    "s": SphericalComponent,
}


def _parse_component_str(space_str: str) -> Tuple[int, str, int]:
    space_str = space_str.split("-")[0]
    i = 0
    while i < len(space_str) and space_str[i].isdigit():
        i += 1
    multiplier = space_str[:i] or "1"

    j = i
    while j < len(space_str) and space_str[j].isalpha():
        j += 1
    space_type = space_str[i:j]
    dim = space_str[j:]
    return int(multiplier), space_type, int(dim)


def parse_components(arg: str, fixed_curvature: bool) -> List[Component]:
    arg = arg.lower().strip()
    if not arg:
        return []

    components: List[Component] = []
    for spec in arg.split(","):
        multiplier, space_type, dim = _parse_component_str(spec.strip())
        if multiplier < 1:
            raise ValueError("Multiplier must be >= 1")
        if dim < 1:
            raise ValueError("Dimension must be >= 1")
        if space_type not in _SPACE_CREATOR_MAP:
            raise NotImplementedError(
                f"Unsupported space type '{space_type}'. Only e, h, and s are available in this standalone model."
            )
        ctor = _SPACE_CREATOR_MAP[space_type]
        components.extend(ctor(dim, fixed_curvature) for _ in range(multiplier))
    return components


class RelevantNeighborSampler:
    def __init__(self, k: int, alpha: float) -> None:
        self.k = k
        self.alpha = alpha

    def sample(self, corr: Tensor, edge_index: Tensor) -> Tensor:
        num_nodes = corr.size(0)
        row, col = edge_index
        adj = torch.zeros((num_nodes, num_nodes), device=corr.device, dtype=corr.dtype)
        adj[row, col] = 1.0
        adj.fill_diagonal_(1.0)
        similarity = corr + self.alpha * (adj @ corr)
        similarity = torch.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)
        similarity.fill_diagonal_(-float("inf"))
        k = min(self.k, max(num_nodes - 1, 1))
        return torch.topk(similarity, k, dim=-1).indices



class NPMCGRN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        components: Optional[List[Component]] = None,
        scalar_parametrization: bool = True,
        gate_sn: bool = False,
        lambda_lip: float = 0.0,
        sphere_topk: int = 15,
        sphere_alpha: float = 0.5,
        sphere_heads: int = 8,
        sphere_layers: int = 1,
        sphere_ff_mult: int = 4,
        sphere_chunk_size: int = 16,
        tf_index: Optional[Tensor] = None,
        use_hyp_struct: bool = True,
        hyp_input_max_norm: float = 6.0,
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.sphere_topk = sphere_topk
        self.sphere_alpha = sphere_alpha
        self.sphere_heads = sphere_heads
        self.sphere_layers = sphere_layers
        self.sphere_ff_mult = sphere_ff_mult
        self.sphere_chunk_size = sphere_chunk_size
        self.use_hyp_struct = use_hyp_struct
        self.hyp_expr_scale = hidden_channels**-0.5
        self.hyp_input_max_norm = hyp_input_max_norm

        if tf_index is None:
            tf_index_tensor = torch.empty(0, dtype=torch.long)
        elif isinstance(tf_index, Tensor):
            tf_index_tensor = tf_index.detach().long().view(-1)
        else:
            tf_index_tensor = torch.as_tensor(tf_index, dtype=torch.long).view(-1)
        self.register_buffer("tf_index", tf_index_tensor)

        if sphere_heads < 1 or hidden_channels % sphere_heads != 0:
            raise ValueError("sphere_heads must be positive and divide hidden_channels")
        if sphere_layers < 1:
            raise ValueError("sphere_layers must be >= 1")
        if sphere_chunk_size < 1:
            raise ValueError("sphere_chunk_size must be >= 1")
        if hyp_input_max_norm <= 0:
            raise ValueError("hyp_input_max_norm must be positive")

        self.gate_sn = gate_sn
        self.lambda_lip = lambda_lip

        self.mu_feat_list = nn.Linear(in_channels, hidden_channels)
        self.appnp_feat_list = APPNP(K=1, alpha=0.0)
        self.appnp_feat_list_1 = APPNP(K=3, alpha=0.0)

        self.register_buffer("id_eye", torch.eye(num_nodes))
        self.mu_id_list = nn.Linear(num_nodes, hidden_channels, bias=False)

        self.euclidean_comp: Optional[EuclideanComponent] = None
        self.hyperbolic_comp: Optional[HyperbolicComponent] = None
        self.spherical_comp: Optional[SphericalComponent] = None
        self.euclidean_proj: Optional[nn.Module] = None
        self.hyperbolic_proj: Optional[nn.Module] = None
        self.spherical_proj: Optional[nn.Module] = None

        self.sphere_neighbor_sampler = RelevantNeighborSampler(sphere_topk, sphere_alpha)
        self.sphere_input_proj = nn.Linear(hidden_channels, hidden_channels)
        self.sphere_input_norm = nn.LayerNorm(hidden_channels)
        sphere_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=sphere_heads,
            dim_feedforward=sphere_ff_mult * hidden_channels,
            dropout=0.1,
            batch_first=True,
        )
        self.sphere_transformer = nn.TransformerEncoder(sphere_layer, num_layers=sphere_layers)
        self.sphere_output_norm = nn.LayerNorm(hidden_channels)
        self.sphere_corr: Optional[Tensor] = None
        self.sphere_topk_indices: Optional[Tensor] = None
        self.sphere_graph_signature = None

        self.hyp_expr_norm = nn.LayerNorm(hidden_channels)
        self.hyp_struct_mlp = nn.Sequential(
            nn.Linear(7, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.hyp_struct_alpha = nn.Parameter(torch.tensor(0.1))
        self.hyp_struct_features: Optional[Tensor] = None
        self.hyp_struct_graph_signature = None

        if components is not None:
            for comp in components:
                comp.init_layers(hidden_channels, scalar_parametrization=scalar_parametrization)

                if isinstance(comp, EuclideanComponent):
                    self.euclidean_comp = comp
                    self.euclidean_proj = (
                        nn.Identity() if comp.dim == hidden_channels else nn.Linear(comp.dim, hidden_channels)
                    )
                elif isinstance(comp, HyperbolicComponent):
                    self.hyperbolic_comp = comp
                    self.hyperbolic_proj = (
                        nn.Identity() if comp.dim == hidden_channels else nn.Linear(comp.dim, hidden_channels)
                    )
                elif isinstance(comp, SphericalComponent):
                    self.spherical_comp = comp
                    self.spherical_proj = (
                        nn.Identity() if comp.dim == hidden_channels else nn.Linear(comp.dim, hidden_channels)
                    )

        def _sn(layer: nn.Linear) -> nn.Module:
            return nn.utils.spectral_norm(layer) if self.gate_sn else layer

        self.gate_net = nn.Sequential(
            _sn(nn.Linear(3 * hidden_channels, hidden_channels)),
            nn.ReLU(inplace=True),
            _sn(nn.Linear(hidden_channels, 3)),
        )
        self.mix_logit = nn.Parameter(torch.tensor(-4.6))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in [self.mu_feat_list, self.mu_id_list]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        for layer in self.gate_net:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        for proj in [self.euclidean_proj, self.hyperbolic_proj, self.spherical_proj]:
            if isinstance(proj, nn.Linear):
                nn.init.xavier_uniform_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)

        nn.init.xavier_uniform_(self.sphere_input_proj.weight)
        if self.sphere_input_proj.bias is not None:
            nn.init.zeros_(self.sphere_input_proj.bias)

        self.hyp_expr_norm.reset_parameters()
        for layer in self.hyp_struct_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        with torch.no_grad():
            self.hyp_struct_alpha.fill_(0.1)

        for module in self.sphere_transformer.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        for comp in [self.euclidean_comp, self.hyperbolic_comp, self.spherical_comp]:
            if comp is None:
                continue
            for module in comp.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

    def _graph_signature(self, edge_index: Tensor) -> Tuple[int, int, int, int, int]:
        row, col = edge_index
        return (
            int(self.num_nodes),
            int(edge_index.size(1)),
            int(row.sum().item()),
            int(col.sum().item()),
            int((row.square().sum() + col.square().sum()).item()),
        )

    def _compute_feature_corr(self, x: Tensor) -> Tensor:
        corr = torch.corrcoef(x)
        return torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_hyperbolic_struct_features(self, edge_index: Tensor) -> Tensor:
        row, col = edge_index
        device = edge_index.device
        dtype = self.hyp_struct_alpha.dtype
        ones = torch.ones(row.size(0), device=device, dtype=dtype)
        out_degree = scatter_add(ones, row, dim=0, dim_size=self.num_nodes)
        in_degree = scatter_add(ones, col, dim=0, dim_size=self.num_nodes)
        total_degree = in_degree + out_degree
        denom = total_degree.clamp_min(1.0)

        is_tf = torch.zeros(self.num_nodes, device=device, dtype=dtype)
        if self.tf_index.numel() > 0:
            tf_index = self.tf_index.to(device=device)
            valid_tf = tf_index[(tf_index >= 0) & (tf_index < self.num_nodes)]
            is_tf[valid_tf] = 1.0

        source_only = ((out_degree > 0) & (in_degree == 0)).to(dtype)
        target_only = ((in_degree > 0) & (out_degree == 0)).to(dtype)
        return torch.stack(
            [
                torch.log1p(out_degree),
                torch.log1p(in_degree),
                out_degree / denom,
                in_degree / denom,
                is_tf,
                source_only,
                target_only,
            ],
            dim=1,
        )

    def _get_hyperbolic_struct_features(self, edge_index: Tensor, dtype: torch.dtype) -> Tensor:
        graph_signature = self._graph_signature(edge_index)
        if (
            self.hyp_struct_features is None
            or self.hyp_struct_graph_signature != graph_signature
            or self.hyp_struct_features.device != edge_index.device
        ):
            self.hyp_struct_features = self._compute_hyperbolic_struct_features(edge_index)
            self.hyp_struct_graph_signature = graph_signature
        return self.hyp_struct_features.to(dtype=dtype)

    def _clip_hyperbolic_input(self, hyp_view: Tensor) -> Tensor:
        norm = hyp_view.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
        scale = (self.hyp_input_max_norm / norm).clamp(max=1.0)
        return hyp_view * scale

    def _encode_hyperbolic_view(self, expr_repr: Tensor, edge_index: Tensor) -> Tensor:
        hyp_view = self.hyp_expr_norm(expr_repr) * self.hyp_expr_scale
        if self.use_hyp_struct:
            struct_feat = self._get_hyperbolic_struct_features(edge_index, expr_repr.dtype)
            struct_view = self.hyp_struct_mlp(struct_feat)
            hyp_view = hyp_view + self.hyp_struct_alpha.to(dtype=expr_repr.dtype) * struct_view
        return self._clip_hyperbolic_input(hyp_view)

    def _initialize_spherical_neighbors(self, x: Tensor, edge_index: Tensor) -> None:
        self.sphere_corr = self._compute_feature_corr(x)
        self.sphere_topk_indices = self.sphere_neighbor_sampler.sample(self.sphere_corr, edge_index)
        self.sphere_graph_signature = self._graph_signature(edge_index)

    def initialize_spherical_neighbors(self, x: Tensor, edge_index: Tensor) -> None:
        self._initialize_spherical_neighbors(x, edge_index)

    def _build_spherical_sequences(self, base_repr: Tensor, topk_indices: Tensor) -> Tensor:
        center = base_repr.unsqueeze(1)
        neighbors = base_repr[topk_indices]
        return torch.cat([center, neighbors], dim=1)

    def _encode_spherical_view(self, base_repr: Tensor, x: Tensor, edge_index: Tensor) -> Tensor:
        if self.sphere_topk_indices is None or self.sphere_graph_signature != self._graph_signature(edge_index):
            self._initialize_spherical_neighbors(x, edge_index)

        sphere_tokens = self.sphere_input_norm(self.sphere_input_proj(base_repr))
        sphere_sequences = self._build_spherical_sequences(sphere_tokens, self.sphere_topk_indices)
        centers = []
        chunk_size = min(self.sphere_chunk_size, sphere_sequences.size(0))
        for start in range(0, sphere_sequences.size(0), chunk_size):
            chunk_input = sphere_sequences[start : start + chunk_size]
            if self.training:
                chunk_encoded = checkpoint(
                    lambda inp: self.sphere_transformer(inp),
                    chunk_input,
                    use_reentrant=False,
                )
            else:
                chunk_encoded = self.sphere_transformer(chunk_input)
            centers.append(chunk_encoded[:, 0, :])

        sphere_center = torch.cat(centers, dim=0)
        sphere_view = self.sphere_output_norm(sphere_center + sphere_tokens)
        return F.normalize(sphere_view, p=2, dim=1)

    @staticmethod
    def _component_mean(comp: Component, x: Tensor) -> Tensor:
        if comp.fc_mean is None or comp.manifold is None:
            raise RuntimeError("Component layers have not been initialized")
        z_mean = comp.fc_mean(x)
        return comp.manifold.exp_map_mu0(z_mean)

    def _compute_lip_loss(self) -> Tensor:
        if not (self.lambda_lip > 0):
            return torch.tensor(0.0, device=self.mix_logit.device)

        loss = torch.tensor(0.0, device=self.mix_logit.device)
        for module in self.gate_net:
            if isinstance(module, nn.Linear):
                loss += (torch.linalg.norm(module.weight, ord=2) - 1.0) ** 2
        return self.lambda_lip * loss

    def encode(self, x: Tensor, edge_index: Tensor, *, return_weights: bool = False):
        mu_f = self.appnp_feat_list(self.mu_feat_list(x), edge_index)
        z_f_e = F.normalize(mu_f, p=2, dim=1)

        mu_f = self.appnp_feat_list_1(self.mu_feat_list(x), edge_index)
        z_f_h = self._encode_hyperbolic_view(mu_f, edge_index)
        z_f_s = self._encode_spherical_view(z_f_e, x, edge_index)

        z_i = F.normalize(self.mu_id_list(self.id_eye), p=2, dim=1) * 0.8

        proj_outs = []
        if self.euclidean_comp is not None and self.euclidean_proj is not None:
            z_e = self._component_mean(self.euclidean_comp, z_f_e)
            proj_outs.append(F.relu(self.euclidean_proj(z_e)))

        if self.hyperbolic_comp is not None and self.hyperbolic_proj is not None:
            z_h = self._component_mean(self.hyperbolic_comp, z_f_h)
            proj_outs.append(F.relu(self.hyperbolic_proj(z_h)))

        if self.spherical_comp is not None and self.spherical_proj is not None:
            z_s = self._component_mean(self.spherical_comp, z_f_s)
            proj_outs.append(F.relu(self.spherical_proj(z_s)))

        if len(proj_outs) != 3:
            raise RuntimeError("NPMCGRN expects exactly e, h, and s components")

        expert_stack = torch.stack(proj_outs, dim=1)
        logits = self.gate_net(expert_stack.flatten(1))
        weights = torch.softmax(logits, 1).unsqueeze(-1)
        mixture = F.normalize((expert_stack * weights).sum(1), p=2, dim=1)
        z_f_e = z_f_e + torch.sigmoid(self.mix_logit) * mixture
        z_f_e = F.normalize(z_f_e, p=2, dim=1)

        if return_weights:
            return z_f_e, z_i, weights.squeeze(-1)
        return z_f_e, z_i

    def decode(self, z_f: Tensor, z_i: Tensor, edge_index: Tensor, sigmoid: bool = True, temp: float = 1.0) -> Tensor:
        device = z_f.device
        edge_count = edge_index.size(1)
        value = torch.zeros(edge_count, device=device)
        s_f = (z_f[edge_index[0]] * z_f[edge_index[1]]).sum(1)
        s_i = torch.sigmoid(z_i[edge_index[0], 0] + z_i[edge_index[1], 0])
        detached = s_f.detach()
        logits = torch.stack([detached, torch.zeros_like(detached)], 1)
        if self.training:
            alpha = F.gumbel_softmax(logits, tau=temp, hard=True)[:, 0]
        else:
            alpha = F.softmax(logits, 1)[:, 0]
        value += alpha * s_f + (1 - alpha) * s_i
        return torch.clamp(value, 0, 1) if sigmoid else value

    def decode_all(self, z_f: Tensor, z_i: Tensor, sigmoid: bool = True, temp: float = 1.0) -> Tensor:
        device = z_f.device
        adj = torch.zeros(self.num_nodes, self.num_nodes, device=device)
        fv = z_f @ z_f.t()
        nv = torch.sigmoid(z_i[:, 0].unsqueeze(1) + z_i[:, 0].unsqueeze(0))
        detached = fv.flatten().detach()
        logits = torch.stack([detached, torch.zeros_like(detached)], 1)
        if self.training:
            alpha = F.gumbel_softmax(logits, tau=temp, hard=True)[:, 0]
        else:
            alpha = F.softmax(logits, 1)[:, 0]
        alpha = alpha.view(self.num_nodes, self.num_nodes)
        adj += alpha * fv + (1 - alpha) * nv
        return torch.clamp(adj, 0, 1) if sigmoid else adj

    def recon_loss(self, z_f: Tensor, z_i: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor, temp: float = 1.0):
        pos = self.decode(z_f, z_i, pos_edge_index, sigmoid=True, temp=temp)
        loss_pos = -torch.log(pos + 1e-15).sum()
        neg = self.decode(z_f, z_i, neg_edge_index, sigmoid=True, temp=temp)
        loss_neg = -torch.log(1 - neg + 1e-15).sum()
        return loss_pos + loss_neg

    @torch.no_grad()
    def test(self, z_f: Tensor, z_i: Tensor, pos_ei: Tensor, neg_ei: Tensor, temp: float = 1.0):
        pos_pred = self.decode(z_f, z_i, pos_ei, sigmoid=True, temp=temp)
        neg_pred = self.decode(z_f, z_i, neg_ei, sigmoid=True, temp=temp)
        y = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)]).cpu().numpy()
        pred = torch.cat([pos_pred, neg_pred]).cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)

    def forward(self, x: Tensor, edge_index: Tensor, temp: float = 1.0):
        z_f, z_i = self.encode(x, edge_index)
        recon_adj = self.decode_all(z_f, z_i, temp=temp)
        return recon_adj, self._compute_lip_loss()
