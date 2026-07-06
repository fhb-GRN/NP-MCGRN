Table 1  Variants of NP-MCGRN and their modifications.

|       Model       | Euclidean branch | Hyperbolic branch | Spherical branch | Curvature-aware gating | Node-prior decoder |
| :---------------: | :--------------: | :---------------: | :--------------: | :--------------------: | :----------------: |
|     NP-MCGRN      |        √         |         √         |        √         |           √            |         √          |
|   NP-MCGRN(E+H)   |        √         |         √         |        ×         |           √            |         √          |
|   NP-MCGRN(E+S)   |        √         |         ×         |        √         |           √            |         √          |
|   NP-MCGRN(H+S)   |        ×         |         √         |        √         |           √            |         √          |
| NP-MCGRN w/o Gate |        √         |         √         |        √         |           ×            |         √          |
| NP-MCGRNw/o Prior |        √         |         √         |        √         |           √            |         ×          |