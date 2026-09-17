| family         | feature_removed   | method                  |   positives |   ap_lift |   ap_lift_lo |   ap_lift_hi |   d_ap_lift |   d_ap_lift_lo |   d_ap_lift_hi |   d_ap_lift_p |
|:---------------|:------------------|:------------------------|------------:|----------:|-------------:|-------------:|------------:|---------------:|---------------:|--------------:|
| Coinbase       | has_coinbase      | unseeded ZSH            |         749 |     23.4  |        21.2  |        25.7  |   nan       |       nan      |      nan       |       nan     |
| Coinbase       | has_coinbase      | seeded, all families    |         749 |     23.8  |        21.6  |        26.2  |     0.466   |         0.323  |        0.613   |         0.002 |
| Coinbase       | has_coinbase      | seeded, family withheld |         749 |     23.8  |        21.6  |        26.2  |     0.466   |         0.323  |        0.613   |         0.002 |
| ManyInManyOut  | nan               | unseeded ZSH            |       57088 |     22.1  |        21.9  |        22.4  |   nan       |       nan      |      nan       |       nan     |
| ManyInManyOut  | nan               | seeded, all families    |       57088 |     21.2  |        21    |        21.5  |    -0.901   |        -1.08   |       -0.752   |         0.002 |
| ManyInManyOut  | nan               | seeded, family withheld |       57088 |     19.1  |        18.9  |        19.3  |    -3.04    |        -3.2    |       -2.89    |         0.002 |
| SingleInFanOut | nan               | unseeded ZSH            |       48946 |     22.4  |        22.2  |        22.7  |   nan       |       nan      |      nan       |       nan     |
| SingleInFanOut | nan               | seeded, all families    |       48946 |     21.1  |        20.9  |        21.4  |    -1.32    |        -1.43   |       -1.21    |         0.002 |
| SingleInFanOut | nan               | seeded, family withheld |       48946 |     19.3  |        19    |        19.5  |    -3.13    |        -3.42   |       -3.02    |         0.002 |
| FanIn          | nan               | unseeded ZSH            |      119797 |     13.9  |        13.8  |        14    |   nan       |       nan      |      nan       |       nan     |
| FanIn          | nan               | seeded, all families    |      119797 |     13.7  |        13.5  |        13.8  |    -0.263   |        -0.33   |       -0.16    |         0.002 |
| FanIn          | nan               | seeded, family withheld |      119797 |     13.9  |        13.8  |        14    |    -0.0275  |        -0.0896 |        0.0651  |         0.725 |
| FanOut         | nan               | unseeded ZSH            |      178490 |      4.14 |         4.09 |         4.18 |   nan       |       nan      |      nan       |       nan     |
| FanOut         | nan               | seeded, all families    |      178490 |      5.45 |         5.4  |         5.51 |     1.31    |         1.27   |        1.37    |         0.002 |
| FanOut         | nan               | seeded, family withheld |      178490 |      4.95 |         4.89 |         5    |     0.808   |         0.768  |        0.861   |         0.002 |
| OneInOneOut    | nan               | unseeded ZSH            |      505982 |      4.14 |         4.1  |         4.18 |   nan       |       nan      |      nan       |       nan     |
| OneInOneOut    | nan               | seeded, all families    |      505982 |      4.26 |         4.23 |         4.3  |     0.123   |         0.11   |        0.134   |         0.002 |
| OneInOneOut    | nan               | seeded, family withheld |      505982 |      4.14 |         4.1  |         4.17 |    -0.00366 |        -0.0141 |        0.00472 |         0.34  |
| OpReturn       | has_op_return     | unseeded ZSH            |      987688 |      2.12 |         2.1  |         2.14 |   nan       |       nan      |      nan       |       nan     |
| OpReturn       | has_op_return     | seeded, all families    |      987688 |      2.07 |         2.06 |         2.09 |    -0.0431  |        -0.0467 |       -0.0393  |         0.002 |
| OpReturn       | has_op_return     | seeded, family withheld |      987688 |      2.1  |         2.08 |         2.12 |    -0.0175  |        -0.0228 |       -0.0123  |         0.002 |
| RBF            | rbf_enabled       | unseeded ZSH            |      314504 |      2.9  |         2.88 |         2.93 |   nan       |       nan      |      nan       |       nan     |
| RBF            | rbf_enabled       | seeded, all families    |      314504 |      2.94 |         2.91 |         2.96 |     0.0311  |         0.0223 |        0.0419  |         0.002 |
| RBF            | rbf_enabled       | seeded, family withheld |      314504 |      2.82 |         2.79 |         2.84 |    -0.0873  |        -0.0965 |       -0.0781  |         0.002 |