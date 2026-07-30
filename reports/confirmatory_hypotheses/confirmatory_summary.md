# Confirmatory Hypothesis Test Pack

Design:
- H1-H3 tested via FE DiD with key term `z_eci:pe` and wild-cluster bootstrap p-values.
- H4 tested via `z_eci:pe:z_coi` moderation.
- H5 tested via `z_eci:pe:z_gpr` moderation.
- Pretrend and placebo checks included for outcome-level identification diagnostics.
- Multiple-testing correction uses Benjamini-Hochberg FDR across the confirmatory family.

## Hypothesis decisions
- H1: NOT_SUPPORTED | raw sig=0, FDR sig=0, pretrend_pass=1, placebo_pass=0
- H2: NOT_SUPPORTED | raw sig=0, FDR sig=0, pretrend_pass=1, placebo_pass=1
- H3: NOT_SUPPORTED | raw sig=0, FDR sig=0, pretrend_pass=1, placebo_pass=1
- H4: NOT_SUPPORTED | raw sig=1, FDR sig=1, pretrend_pass=1, placebo_pass=0
- H5: NOT_SUPPORTED | raw sig=1, FDR sig=1, pretrend_pass=1, placebo_pass=0

Output files:
- `confirmatory_primary_tests.csv`
- `confirmatory_moderation_tests.csv`
- `confirmatory_pretrend_placebo.csv`
- `confirmatory_multiple_testing.csv`
- `confirmatory_robustness_layer.csv`
- `hypothesis_test_matrix.csv`