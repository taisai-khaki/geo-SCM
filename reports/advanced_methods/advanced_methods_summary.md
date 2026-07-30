# Advanced Methods Results

Implemented methods:
- Wild cluster bootstrap (Rademacher) for FE coefficient `z_eci:pe`.
- Robust FE checks: Huber RLM and trimmed (5-95) FE.
- Causal heterogeneity exploration via T-learner random forests and policy-tree segmentation.

## Wild Cluster Bootstrap
- DV1_GVC_Linkage_Change: coef=0.3519, cluster-p=0.4076, wild-bootstrap p=0.7797, reps=59
- DV2_Export_Recovery: coef=-0.1031, cluster-p=0.1827, wild-bootstrap p=0.1186, reps=59
- DV3_Partner_Diversification: coef=0.0006, cluster-p=0.963, wild-bootstrap p=0.9661, reps=59

## Robust FE
- DV1_GVC_Linkage_Change: OLS_FE_cluster=0.3519(p=0.4076); RLM_Huber_FE=0.1766(p=0.1272); Trimmed_5_95_FE_cluster=0.0053(p=0.9508)
- DV2_Export_Recovery: OLS_FE_cluster=-0.1031(p=0.1827); RLM_Huber_FE=-0.0884(p=7.114e-09); Trimmed_5_95_FE_cluster=-0.0562(p=0.03759)
- DV3_Partner_Diversification: OLS_FE_cluster=0.0006(p=0.963); RLM_Huber_FE=0.0040(p=0.04903); Trimmed_5_95_FE_cluster=0.0073(p=0.3704)

## Causal Heterogeneity (T-learner)
- DV1_GVC_Linkage_Change: mean CATE=0.1982, p25/p50/p75=(-0.0622, 0.1624, 0.4342), share positive=68.63%
- DV2_Export_Recovery: mean CATE=0.0923, p25/p50/p75=(-0.0426, 0.0827, 0.2237), share positive=68.48%
- DV3_Partner_Diversification: mean CATE=-0.0536, p25/p50/p75=(-0.1297, -0.0557, -0.0115), share positive=21.28%

Caution:
- T-learner CATE outputs are exploratory and rely on unconfoundedness assumptions.
- Use these to guide segmentation and mechanism discussion, not as definitive causal proof alone.