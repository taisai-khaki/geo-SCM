# Data Sample Audit (2022 inclusion)

- Confirmatory sample ending in 2021 is driven by complete-case filtering when `wdi_natural_resource_rents_pct_gdp` is included.
- In 2022, rents is missing for all 231 country rows in the regression panel.
- Dropping rents restores 2022 observations (H2/H3: +195 rows total; H1: +75 rows total).

Files:
- control_missing_by_year.csv
- sample_inclusion_scenarios.csv
- h1_h3_with_vs_without_rents.csv