import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import numpy as np
import pandas as pd
import torch
from model import NPMCGRN, parse_components
from utils import adj2saprse_tensor, convert_to_pyg_data, load_data, scRNADataset, set_random_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--hidden_channels", type=int, default=512)
    parser.add_argument("--gate_sn", default=False)
    parser.add_argument("--lambda_lip", type=float, default=0.0)
    parser.add_argument("--sphere_topk", type=int, default=15)
    parser.add_argument("--sphere_alpha", type=float, default=0.5)
    parser.add_argument("--sphere_heads", type=int, default=8)
    parser.add_argument("--sphere_layers", type=int, default=1)
    parser.add_argument("--sphere_ff_mult", type=int, default=4)
    parser.add_argument("--sphere_chunk_size", type=int, default=16)
    parser.add_argument("--disable_hyp_struct", action="store_true")
    parser.add_argument("--hyp_input_max_norm", type=float, default=6)
    parser.add_argument("--net_type", type=str, default="STRING")
    parser.add_argument("--data_type", type=str, default="mHSC-E")
    parser.add_argument("--num", type=int, default=1000)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.eval_every < 1:
        raise ValueError("--eval_every must be >= 1")

    set_random_seed(2025)
    exp_file = ("../Benchmark Dataset/"
        + args.net_type
        + " Dataset/"
        + args.data_type
        + "/TFs+"
        + str(args.num)
        + "/BL--ExpressionData.csv"
    )
    tf_file = (
        "../Benchmark Dataset/"
        + args.net_type
        + " Dataset/"
        + args.data_type
        + "/TFs+"
        + str(args.num)
        + "/TF.csv"
    )
    target_file = (
        "../Benchmark Dataset/"
        + args.net_type
        + " Dataset/"
        + args.data_type
        + "/TFs+"
        + str(args.num)
        + "/Target.csv"
    )
    train_file = "../data/" + args.net_type + "/" + args.data_type + " " + str(args.num) + "/Train_set.csv"
    val_file = "../data/" + args.net_type + "/" + args.data_type + " " + str(args.num) + "/Validation_set.csv"
    test_file = "../data/" + args.net_type + "/" + args.data_type + " " + str(args.num) + "/Test_set.csv"

    data_input = pd.read_csv(exp_file, index_col=0)
    loader = load_data(data_input)
    feature = loader.exp_data()

    tf = pd.read_csv(tf_file, index_col=0)["index"].values.astype(np.int64)
    _target = pd.read_csv(target_file, index_col=0)["index"].values.astype(np.int64)

    feature = torch.from_numpy(feature).float()
    tf = torch.from_numpy(tf).long()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_feature = feature.to(device)
    tf = tf.to(device)

    train_data1 = pd.read_csv(train_file, index_col=0).values
    validation_data1 = pd.read_csv(val_file, index_col=0).values
    test_data1 = pd.read_csv(test_file, index_col=0).values

    train_load = scRNADataset(train_data1, feature.shape[0], flag=False)
    adj = train_load.Adj_Generate(tf, loop=False)
    adj = adj2saprse_tensor(adj)

    train_data1 = torch.from_numpy(train_data1).long()
    val_data1 = torch.from_numpy(validation_data1).long()
    test_data1 = torch.from_numpy(test_data1).long()

    train_data, val_data, test_data = convert_to_pyg_data(data_feature, adj, train_data1, val_data1, test_data1)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)


    comps = parse_components("e2,h2,s2", fixed_curvature=True)
    model = NPMCGRN(
        num_nodes=train_data.num_nodes,
        in_channels=feature.shape[1],
        hidden_channels=args.hidden_channels,
        components=comps,
        gate_sn=args.gate_sn,
        lambda_lip=args.lambda_lip,
        sphere_topk=args.sphere_topk,
        sphere_alpha=args.sphere_alpha,
        sphere_heads=args.sphere_heads,
        sphere_layers=args.sphere_layers,
        sphere_ff_mult=args.sphere_ff_mult,
        sphere_chunk_size=args.sphere_chunk_size,
        tf_index=tf,
        use_hyp_struct=not args.disable_hyp_struct,
        hyp_input_max_norm=args.hyp_input_max_norm,
    ).to(device)
    model.initialize_spherical_neighbors(train_data.x, train_data.edge_index)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    max_temp, min_temp, decay_step = 2.0, 0.1, 150.0
    decay_w = np.log(max_temp / min_temp)

    def get_temp(epoch):
        return max(max_temp * np.exp(-(epoch - 1) / decay_step * decay_w), min_temp)

    @torch.no_grad()
    def eval_auc_ap(pos_ei, neg_ei, epoch):
        model.eval()
        z_f, z_i, _weights = model.encode(train_data.x, train_data.edge_index, return_weights=True)
        auc, ap = model.test(z_f, z_i, pos_ei, neg_ei, temp=get_temp(epoch))
        return auc, ap

    best_val_ap = -float("inf")
    best_val_auc = -float("inf")
    best_epoch = 0
    best_state_dict = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        z_f, z_i = model.encode(train_data.x, train_data.edge_index)
        recon_loss = model.recon_loss(
            z_f,
            z_i,
            train_data.pos_edge_label_index,
            train_data.neg_edge_label_index,
            temp=get_temp(epoch),
        )
        lip_loss = model._compute_lip_loss()
        loss = recon_loss + lip_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        should_eval = epoch % args.eval_every == 0 or epoch == args.epochs
        if not should_eval:
            continue

        val_auc, val_ap = eval_auc_ap(val_data.pos_edge_label_index, val_data.neg_edge_label_index, epoch)
        is_best = val_ap > best_val_ap or (val_ap == best_val_ap and val_auc > best_val_auc)
        if is_best:
            best_val_ap = val_ap
            best_val_auc = val_auc
            best_epoch = epoch
            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

        print(
            f"Epoch {epoch:03d} | Loss {recon_loss:.4f} | "
            f"Val AUC {val_auc:.4f} | Val AP {val_ap:.4f}"
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_auc, test_ap = eval_auc_ap(test_data.pos_edge_label_index, test_data.neg_edge_label_index, args.epochs)
    print(f"Best Epoch: {best_epoch:03d} | Best Val AUC: {best_val_auc:.4f} | Best Val AP: {best_val_ap:.4f}")
    print(f"Final Test  AUC: {test_auc:.4f} | AP: {test_ap:.4f}")


if __name__ == "__main__":
    main()
