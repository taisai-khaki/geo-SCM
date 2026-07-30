# Simpson-Paradox Diagnostics

Interpretation guide:
- `reversal_groups_share` near 1.0 means many subgroup coefficients have the opposite sign of pooled.
- `sign_opposite=1` in within-between decomposition means within-country and between-country effects conflict.

## Raw interaction screen (no controls)
- H1_DV1_Stability | eci_t: reversals=1/3, pooled=0.0443, weighted-group=0.2616
- H1_DV1_Stability | exposed: reversals=0/1, pooled=0.0443, weighted-group=0.0218
- H1_DV1_Stability | income_q: reversals=2/4, pooled=0.0443, weighted-group=-0.2064
- H1_DV1_Stability | size_q: reversals=1/2, pooled=0.0443, weighted-group=0.1551
- H2_DV2_ExportRecovery | eci_t: reversals=0/3, pooled=0.3573, weighted-group=0.4593
- H2_DV2_ExportRecovery | exposed: reversals=0/1, pooled=0.3573, weighted-group=1.3623
- H2_DV2_ExportRecovery | income_q: reversals=3/4, pooled=0.3573, weighted-group=-0.1139
- H2_DV2_ExportRecovery | size_q: reversals=2/4, pooled=0.3573, weighted-group=0.5596
- H3_DV3_Diversification | eci_t: reversals=2/3, pooled=-0.0156, weighted-group=0.0069
- H3_DV3_Diversification | exposed: reversals=0/1, pooled=-0.0156, weighted-group=-0.0272
- H3_DV3_Diversification | income_q: reversals=3/4, pooled=-0.0156, weighted-group=0.0560
- H3_DV3_Diversification | size_q: reversals=2/4, pooled=-0.0156, weighted-group=0.0029

## Controlled + year-FE interaction screen
- H1_DV1_Stability | eci_t: reversals=1/3, pooled=-0.2639, weighted-group=-0.0313
- H1_DV1_Stability | exposed: reversals=0/1, pooled=-0.2639, weighted-group=-0.2243
- H1_DV1_Stability | income_q: reversals=1/3, pooled=-0.2639, weighted-group=-0.4126
- H1_DV1_Stability | size_q: reversals=0/2, pooled=-0.2639, weighted-group=-0.3437
- H2_DV2_ExportRecovery | eci_t: reversals=2/3, pooled=-0.1145, weighted-group=-0.0261
- H2_DV2_ExportRecovery | exposed: reversals=0/1, pooled=-0.1145, weighted-group=-0.1056
- H2_DV2_ExportRecovery | income_q: reversals=2/4, pooled=-0.1145, weighted-group=-0.0162
- H2_DV2_ExportRecovery | size_q: reversals=1/4, pooled=-0.1145, weighted-group=-0.1810
- H3_DV3_Diversification | eci_t: reversals=1/3, pooled=-0.0132, weighted-group=-0.0117
- H3_DV3_Diversification | exposed: reversals=0/1, pooled=-0.0132, weighted-group=-0.0038
- H3_DV3_Diversification | income_q: reversals=3/4, pooled=-0.0132, weighted-group=0.0441
- H3_DV3_Diversification | size_q: reversals=2/4, pooled=-0.0132, weighted-group=0.0171

## Within vs between decomposition
- H1_DV1_Stability: within=0.4377 (p=0.3716), between=-0.2051 (p=0.1811), opposite_sign=1
- H2_DV2_ExportRecovery: within=0.2458 (p=0.4731), between=-0.1521 (p=0.04793), opposite_sign=1
- H3_DV3_Diversification: within=0.0132 (p=0.5719), between=-0.0020 (p=0.8713), opposite_sign=1

Conclusion: Simpson-like aggregation risk exists when pooled and subgroup/within-between signs conflict.