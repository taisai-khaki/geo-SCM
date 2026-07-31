# Final Design Completion Outputs

This directory is the authoritative completed analysis package. It supersedes earlier exploratory and legacy output directories when the same question is addressed.

## Read these first

1. `analysis_summary.md`: compact conclusion, sample sizes, and interpretation.
2. `confirmatory_tariff_weighted_tests.csv`: H1-H3 results using the primary continuous tariff-weighted exposure.
3. `h4_tariff_weighted_tests.csv`, `h5_and_h4_omnibus_tests.csv`, and `h5_profile_matched_gpr_sensitivity.csv`: moderation results and GPR sensitivity.
4. `h6_tariff_weighted_tests.csv` and `h6_tariff_weighted_threshold_summary.csv`: pre-specified frozen-median threshold test plus clearly labelled exploratory threshold search.
5. `E1_upper_middle_income_gvc_tests.csv`, `E2_low_eci_export_recovery_tests.csv`, and `E3_high_income_gpr_diversification_observed.csv`: targeted exploratory heterogeneity tests and formal subgroup-difference evidence.
6. `full_reported_multiplicity_family.csv`: the 66-test Benjamini-Hochberg family. This is the source of record for adjusted inference.

## Main conclusion

No confirmatory H1-H5 or H6 result remains statistically significant after the 66-test Benjamini-Hochberg correction at q < .05. Some raw or nominal signals remain useful for transparent theory refinement, but they are not confirmatory support.

The direct-country GPR analysis uses 42 countries with observed data. The profile-matched KNN sensitivity expands coverage but has limited validation performance and must not be represented as primary evidence.

## Reproduction and audit files

- `source_and_construction_audit.csv`: inputs, transformations, and caveats.
- `third_country_sample_audit.csv`: country exclusions and final coverage.
- `frozen_pre_shock_constructs_completed.csv`, `historical_income_groups_2015_2017.csv`, and `frozen_regime_thresholds.csv`: frozen baseline definitions.
- `tariff_weighted_channel_exposure_2015_2017.csv` in `data/processed/`: reproducible country-level exposure profiles.
- `ustr_tariff_line_extraction_audit.csv` and `ustr_section301_hs6_membership_audit.csv`: tariff-line parsing and HS6 mapping audit.
- `panel_with_completed_design_constructs.csv`: regression-ready country-year panel used by the final analysis.
- `equivalence_and_mde_tariff_weighted.csv`: smallest detectable effects and equivalence checks.
- `destination_entry_mechanism_tests.csv`: direct new/persistent destination-entry mechanism tests.
- `gpr_coverage_audit.csv`, `gpr_profile_matching_validation.csv`, and `gpr_profile_matched_imputed_values.csv`: GPR coverage and sensitivity audit.

## Figures

- `figure_completed_confirmatory_coefficients.png`: H1-H5 coefficient plot.
- `figure_h6_tariff_weighted_threshold.png`: H6 threshold test and exploratory search.
- `figure_targeted_exploratory_heterogeneity.png`: E1-E3 subgroup patterns.

Run `python scripts/run_completed_design_analysis.py --bootstrap-reps 999 --threshold-bootstrap-reps 999 --seed 20260730` after downloading the raw sources described in `docs/completed_design_data_sources.md` to rebuild this directory.
