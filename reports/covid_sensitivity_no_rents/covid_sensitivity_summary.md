# COVID-19 Sensitivity Analysis

Primary confirmatory base aligned to legacy-corrected setup (unless stated otherwise).
- Includes natural resource rents in controls: `False`

## Specifications
- `baseline_full`: all years in the estimation window.
- `exclude_2020`: drops pandemic onset year.
- `exclude_2020_2021`: drops both pandemic years.
- `covid_interaction_check`: full sample with `exposed:covid_period` and `z_eci:pe:covid_period` contamination checks.

## Key Terms
- H1-H3 key term: `z_eci:pe`
- H4 key term: `z_eci:pe:z_coi`
- H5 key term: `z_eci:pe:z_gpr`

## H1-H3 Snapshot
- baseline_full | H1 (DV1_GVC_Linkage_Stability): coef=0.0140, p=0.8684, N=754, years=2013-2022
- baseline_full | H2 (DV2_Export_Recovery): coef=-0.1118, p=0.2628, N=1924, years=2012-2022
- baseline_full | H3 (DV3_Partner_Diversification): coef=0.0014, p=0.9072, N=1924, years=2012-2022
- exclude_2020 | H1 (DV1_GVC_Linkage_Stability): coef=0.0738, p=0.6124, N=679, years=2013-2022
- exclude_2020 | H2 (DV2_Export_Recovery): coef=-0.1223, p=0.2682, N=1747, years=2012-2022
- exclude_2020 | H3 (DV3_Partner_Diversification): coef=-0.0000, p=0.9987, N=1747, years=2012-2022
- exclude_2020_2021 | H1 (DV1_GVC_Linkage_Stability): coef=0.2930, p=0.2326, N=604, years=2013-2022
- exclude_2020_2021 | H2 (DV2_Export_Recovery): coef=-0.1170, p=0.3304, N=1571, years=2012-2022
- exclude_2020_2021 | H3 (DV3_Partner_Diversification): coef=-0.0018, p=0.8801, N=1571, years=2012-2022
- covid_interaction_check | H1 (DV1_GVC_Linkage_Stability): coef=0.2633, p=0.2856, N=754, years=2013-2022
- covid_interaction_check | H2 (DV2_Export_Recovery): coef=-0.1133, p=0.3363, N=1924, years=2012-2022
- covid_interaction_check | H3 (DV3_Partner_Diversification): coef=-0.0008, p=0.9446, N=1924, years=2012-2022

## H4-H5 Snapshot
- baseline_full | H4 (DV1_GVC_Linkage_Stability): coef=0.0859, p=0.4298, N=754, years=2013-2022
- baseline_full | H5 (DV1_GVC_Linkage_Stability): coef=0.1306, p=0.3738, N=754, years=2013-2022
- baseline_full | H4 (DV2_Export_Recovery): coef=0.1454, p=0.1521, N=1924, years=2012-2022
- baseline_full | H5 (DV2_Export_Recovery): coef=0.0179, p=0.878, N=1924, years=2012-2022
- baseline_full | H4 (DV3_Partner_Diversification): coef=0.0025, p=0.8535, N=1924, years=2012-2022
- baseline_full | H5 (DV3_Partner_Diversification): coef=-0.0141, p=0.08503, N=1924, years=2012-2022
- exclude_2020 | H4 (DV1_GVC_Linkage_Stability): coef=-0.1347, p=0.4315, N=679, years=2013-2022
- exclude_2020 | H5 (DV1_GVC_Linkage_Stability): coef=0.2549, p=0.1104, N=679, years=2013-2022
- exclude_2020 | H4 (DV2_Export_Recovery): coef=0.1556, p=0.1486, N=1747, years=2012-2022
- exclude_2020 | H5 (DV2_Export_Recovery): coef=0.0254, p=0.8389, N=1747, years=2012-2022
- exclude_2020 | H4 (DV3_Partner_Diversification): coef=0.0050, p=0.7145, N=1747, years=2012-2022
- exclude_2020 | H5 (DV3_Partner_Diversification): coef=-0.0155, p=0.07924, N=1747, years=2012-2022
- exclude_2020_2021 | H4 (DV1_GVC_Linkage_Stability): coef=-0.4377, p=0.06112, N=604, years=2013-2022
- exclude_2020_2021 | H5 (DV1_GVC_Linkage_Stability): coef=0.3503, p=0.032, N=604, years=2013-2022
- exclude_2020_2021 | H4 (DV2_Export_Recovery): coef=0.2202, p=0.03746, N=1571, years=2012-2022
- exclude_2020_2021 | H5 (DV2_Export_Recovery): coef=0.0193, p=0.8648, N=1571, years=2012-2022
- exclude_2020_2021 | H4 (DV3_Partner_Diversification): coef=0.0095, p=0.5191, N=1571, years=2012-2022
- exclude_2020_2021 | H5 (DV3_Partner_Diversification): coef=-0.0178, p=0.05007, N=1571, years=2012-2022
- covid_interaction_check | H4 (DV1_GVC_Linkage_Stability): coef=0.0842, p=0.4588, N=754, years=2013-2022
- covid_interaction_check | H5 (DV1_GVC_Linkage_Stability): coef=0.1113, p=0.3965, N=754, years=2013-2022
- covid_interaction_check | H4 (DV2_Export_Recovery): coef=0.1453, p=0.1529, N=1924, years=2012-2022
- covid_interaction_check | H5 (DV2_Export_Recovery): coef=0.0127, p=0.9135, N=1924, years=2012-2022
- covid_interaction_check | H4 (DV3_Partner_Diversification): coef=0.0025, p=0.8528, N=1924, years=2012-2022
- covid_interaction_check | H5 (DV3_Partner_Diversification): coef=-0.0149, p=0.09227, N=1924, years=2012-2022
