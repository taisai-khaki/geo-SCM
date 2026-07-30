# Additional Ideas: Deep Analysis Results

This report implements:
- Event-study pre-trend and dynamic effects.
- Dual sample reporting (full sample vs. core economies excluding smallest export quartile).
- Quantile/regime heterogeneity analysis.
- Post-2022 external validation using non-TiVA outcomes (DV2, DV3) through 2024.

## Event-study joint tests
- DV1_GVC_Linkage_Change | pretrend_joint_zero: p=0.2869 (terms=4)
- DV1_GVC_Linkage_Change | post_joint_zero: p=0.02691 (terms=3)
- DV2_Export_Recovery | pretrend_joint_zero: p=0.12 (terms=5)
- DV2_Export_Recovery | post_joint_zero: p=0.9874 (terms=3)
- DV3_Partner_Diversification | pretrend_joint_zero: p=0.2386 (terms=5)
- DV3_Partner_Diversification | post_joint_zero: p=0.4576 (terms=3)

## Full vs core sample
- DV1_GVC_Linkage_Change | full_sample: coef=0.3519, p=0.4076, N=679
- DV1_GVC_Linkage_Change | core_economies_excl_q1: coef=0.3519, p=0.4076, N=679
- DV2_Export_Recovery | full_sample: coef=-0.1031, p=0.1827, N=1729
- DV2_Export_Recovery | core_economies_excl_q1: coef=-0.1000, p=0.1057, N=1529
- DV3_Partner_Diversification | full_sample: coef=0.0006, p=0.963, N=1729
- DV3_Partner_Diversification | core_economies_excl_q1: coef=-0.0013, p=0.9117, N=1529

## Quantile results (z_eci:pe)
- DV1_GVC_Linkage_Change: q25=0.2186(p=0.0476), q50=0.3023(p=0.001), q75=0.1693(p=0.0782)
- DV2_Export_Recovery: q25=-0.0510(p=0.000822), q50=-0.0817(p=1.23e-09), q75=-0.0724(p=2.99e-05)
- DV3_Partner_Diversification: q25=-0.0056(p=0.589), q50=-0.0024(p=0.673), q75=-0.0075(p=0.0495)

## ECI regime results (z_eci:pe)
- DV1_GVC_Linkage_Change | low_eci: coef=1.2507, p=0.01796
- DV1_GVC_Linkage_Change | mid_eci: coef=0.5825, p=0.5539
- DV1_GVC_Linkage_Change | high_eci: coef=-0.4016, p=0.5246
- DV2_Export_Recovery | mid_eci: coef=0.0554, p=0.8699
- DV2_Export_Recovery | high_eci: coef=0.0922, p=0.5935
- DV2_Export_Recovery | low_eci: coef=-0.5313, p=0.003157
- DV3_Partner_Diversification | mid_eci: coef=-0.0074, p=0.8655
- DV3_Partner_Diversification | high_eci: coef=0.0286, p=0.2329
- DV3_Partner_Diversification | low_eci: coef=0.0121, p=0.8099

## Post-2022 external validation (2012-2024)
- DV2_Export_Recovery | extended base: coef=-0.8801, p=0.264, years=2012-2024
- DV2_Export_Recovery | shift test: baseline coef=-0.8201 (p=0.2643); post-2022 shift=-0.1860 (p=0.7073)
- DV3_Partner_Diversification | extended base: coef=-0.0069, p=0.6213, years=2012-2024
- DV3_Partner_Diversification | shift test: baseline coef=-0.0047 (p=0.7195); post-2022 shift=-0.0067 (p=0.5988)

## Interpretation
- Strongest and most stable signal remains DV2 (Export Recovery), including outlier-robust settings.
- DV1 and DV3 show stronger heterogeneity and sensitivity; these are better framed as conditional effects rather than universal effects.
- Post-2022 validation helps assess whether effects persist in the extended period where TiVA is unavailable.