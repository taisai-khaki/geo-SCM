# COVID-19 Sensitivity Analysis

Primary confirmatory base aligned to legacy-corrected setup (unless stated otherwise).
- Includes natural resource rents in controls: `True`

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
- baseline_full | H1 (DV1_GVC_Linkage_Stability): coef=-0.1567, p=0.3645, N=679, years=2013-2021
- baseline_full | H2 (DV2_Export_Recovery): coef=-0.1023, p=0.1827, N=1729, years=2012-2021
- baseline_full | H3 (DV3_Partner_Diversification): coef=0.0006, p=0.963, N=1729, years=2012-2021
- exclude_2020 | H1 (DV1_GVC_Linkage_Stability): coef=-0.1712, p=0.1517, N=604, years=2013-2021
- exclude_2020 | H2 (DV2_Export_Recovery): coef=-0.0991, p=0.2112, N=1554, years=2012-2021
- exclude_2020 | H3 (DV3_Partner_Diversification): coef=-0.0002, p=0.9849, N=1554, years=2012-2021
- exclude_2020_2021 | H1 (DV1_GVC_Linkage_Stability): coef=-0.0432, p=0.7894, N=529, years=2013-2019
- exclude_2020_2021 | H2 (DV2_Export_Recovery): coef=-0.1339, p=0.01568, N=1385, years=2012-2019
- exclude_2020_2021 | H3 (DV3_Partner_Diversification): coef=-0.0004, p=0.97, N=1385, years=2012-2019
- covid_interaction_check | H1 (DV1_GVC_Linkage_Stability): coef=0.0041, p=0.9765, N=679, years=2013-2021
- covid_interaction_check | H2 (DV2_Export_Recovery): coef=-0.1113, p=0.05125, N=1729, years=2012-2021
- covid_interaction_check | H3 (DV3_Partner_Diversification): coef=-0.0016, p=0.8778, N=1729, years=2012-2021

## H4-H5 Snapshot
- baseline_full | H4 (DV1_GVC_Linkage_Stability): coef=0.4677, p=5.199e-05, N=679, years=2013-2021
- baseline_full | H5 (DV1_GVC_Linkage_Stability): coef=-0.4069, p=0.003202, N=679, years=2013-2021
- baseline_full | H4 (DV2_Export_Recovery): coef=0.1396, p=0.1602, N=1729, years=2012-2021
- baseline_full | H5 (DV2_Export_Recovery): coef=-0.0417, p=0.6705, N=1729, years=2012-2021
- baseline_full | H4 (DV3_Partner_Diversification): coef=-0.0056, p=0.6511, N=1729, years=2012-2021
- baseline_full | H5 (DV3_Partner_Diversification): coef=-0.0077, p=0.4614, N=1729, years=2012-2021
- exclude_2020 | H4 (DV1_GVC_Linkage_Stability): coef=0.3034, p=0.01428, N=604, years=2013-2021
- exclude_2020 | H5 (DV1_GVC_Linkage_Stability): coef=-0.1875, p=0.1849, N=604, years=2013-2021
- exclude_2020 | H4 (DV2_Export_Recovery): coef=0.1178, p=0.2795, N=1554, years=2012-2021
- exclude_2020 | H5 (DV2_Export_Recovery): coef=-0.0350, p=0.7593, N=1554, years=2012-2021
- exclude_2020 | H4 (DV3_Partner_Diversification): coef=-0.0031, p=0.7985, N=1554, years=2012-2021
- exclude_2020 | H5 (DV3_Partner_Diversification): coef=-0.0066, p=0.5182, N=1554, years=2012-2021
- exclude_2020_2021 | H4 (DV1_GVC_Linkage_Stability): coef=0.1657, p=0.4149, N=529, years=2013-2019
- exclude_2020_2021 | H5 (DV1_GVC_Linkage_Stability): coef=0.0480, p=0.7861, N=529, years=2013-2019
- exclude_2020_2021 | H4 (DV2_Export_Recovery): coef=0.1192, p=0.02679, N=1385, years=2012-2019
- exclude_2020_2021 | H5 (DV2_Export_Recovery): coef=-0.1010, p=0.06711, N=1385, years=2012-2019
- exclude_2020_2021 | H4 (DV3_Partner_Diversification): coef=0.0019, p=0.8773, N=1385, years=2012-2019
- exclude_2020_2021 | H5 (DV3_Partner_Diversification): coef=-0.0079, p=0.3864, N=1385, years=2012-2019
- covid_interaction_check | H4 (DV1_GVC_Linkage_Stability): coef=0.4612, p=8.301e-05, N=679, years=2013-2021
- covid_interaction_check | H5 (DV1_GVC_Linkage_Stability): coef=-0.4263, p=0.001435, N=679, years=2013-2021
- covid_interaction_check | H4 (DV2_Export_Recovery): coef=0.1396, p=0.1599, N=1729, years=2012-2021
- covid_interaction_check | H5 (DV2_Export_Recovery): coef=-0.0525, p=0.5894, N=1729, years=2012-2021
- covid_interaction_check | H4 (DV3_Partner_Diversification): coef=-0.0056, p=0.6512, N=1729, years=2012-2021
- covid_interaction_check | H5 (DV3_Partner_Diversification): coef=-0.0088, p=0.4334, N=1729, years=2012-2021
