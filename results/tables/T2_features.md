| Feature                   |   Most frequent value share | Kept   | Reason if removed                                           |   rank |        mi |   weight |
|:--------------------------|----------------------------:|:-------|:------------------------------------------------------------|-------:|----------:|---------:|
| input_count               |                     0.812   | yes    |                                                             |     10 |   0.608   |   0.0154 |
| output_count              |                     0.464   | yes    |                                                             |      6 |   0.742   |   0.0332 |
| vsize                     |                     0.0913  | yes    |                                                             |      2 |   1.46    |   0.173  |
| total_input_value         |                     0.00219 | yes    |                                                             |      7 |   0.69    |   0.0264 |
| total_output_value        |                     0.138   | no     | |Spearman rho| = 0.9829 with kept feature total_input_value |    nan | nan       | nan      |
| fee                       |                     0.00277 | yes    |                                                             |      3 |   1.05    |   0.094  |
| fee_rate_sat_per_vbyte    |                     0.00204 | yes    |                                                             |      5 |   0.962   |   0.0437 |
| avg_input_value           |                     0.00221 | no     | |Spearman rho| = 0.9852 with kept feature total_input_value |    nan | nan       | nan      |
| avg_output_value          |                     0.138   | yes    |                                                             |      9 |   0.628   |   0.0181 |
| input_output_ratio        |                     0.0021  | yes    |                                                             |      8 |   0.678   |   0.0216 |
| weight                    |                     0.0538  | no     | |Spearman rho| = 0.9994 with kept feature vsize             |    nan | nan       | nan      |
| size                      |                     0.0801  | yes    |                                                             |      1 |   1.49    |   0.489  |
| input_address_count       |                     1       | no     | most frequent value covers 0.99963 of DEV rows              |    nan | nan       | nan      |
| output_address_count      |                     1       | no     | most frequent value covers 0.99983 of DEV rows              |    nan | nan       | nan      |
| input_script_count        |                     1       | no     | most frequent value covers 0.99963 of DEV rows              |    nan | nan       | nan      |
| total_addresses           |                     0.999   | no     | most frequent value covers 0.99946 of DEV rows              |    nan | nan       | nan      |
| value_difference          |                     0.00277 | no     | |Spearman rho| = 1.0000 with kept feature fee               |    nan | nan       | nan      |
| fee_rate_sat_per_byte     |                     0.0017  | yes    |                                                             |      4 |   0.978   |   0.0611 |
| has_coinbase              |                     1       | no     | most frequent value covers 0.99963 of DEV rows              |    nan | nan       | nan      |
| has_op_return             |                     0.987   | yes    |                                                             |     12 |   0.00908 |   0.0118 |
| rbf_enabled               |                     0.513   | yes    |                                                             |     11 |   0.0873  |   0.0134 |
| value_concentration_ratio |                     0.899   | no     | audit-only column (not a v2 candidate)                      |    nan | nan       | nan      |
| is_consolidation          |                   nan       | no     | count rule; used only as annotation (L1)                    |    nan | nan       | nan      |
| is_distribution           |                   nan       | no     | count rule; used only as annotation (L1)                    |    nan | nan       | nan      |
| is_peer_to_peer           |                   nan       | no     | count rule; used only as annotation (L1)                    |    nan | nan       | nan      |
| is_batch_payment          |                   nan       | no     | count rule; used only as annotation (L1)                    |    nan | nan       | nan      |
| is_coinjoin_like          |                   nan       | no     | count rule; used only as annotation (L1)                    |    nan | nan       | nan      |