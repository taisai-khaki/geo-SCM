# Comprehensive Country-Level Analysis (Expanded Panel)

Sample base:
- Rows: `2533`
- Countries: `231`
- Years: `2012-2022`

Generated tables:
- `comprehensive_table1_descriptive.csv`
- `comprehensive_table1_vif.csv`
- `comprehensive_table2_4_main_models.csv`
- `comprehensive_table5_moderation.csv`
- `comprehensive_table6_robustness.csv`

Notes:
- Post period follows the paper note in extracted table text: 2019-2022.
- FE specifications use country and year fixed effects with clustered SEs.
- Main controls include natural resource rents: `True`.
- Full-controls models use broad controls plus US-China intensity; tariff control excluded in the main comprehensive tables to preserve wider country coverage. Tariff appears in the dedicated regression panel and can be reintroduced for stricter replication.