# Structural-Regime Analysis Completion

This package implements the attached ten-step structural-regime checklist as a separate extension of the completed tariff-weighted design.

## Frozen Design
- Structural profile countries: 229; source period: 2012-2017.
- Candidate structural variables: 10; clustering excludes ECI, COI, treatment, outcomes, GPR, and post-shock classifications.
- Selected structural solution: k=2; outcome-independent selection score=0.195.
- Primary post period: 2018 onward; 2019 onward is sensitivity.

## Diagnostics and Inference
- Full structural model family: 84 tests.
- BH-significant tests at q<.05: 0.
- BH-significant pooled regime differences: 0.
- Continuous moderator tests, pooled regime differences, placebo tests, and event-study pretrend joints are reported separately and linked through the full family.

## Interpretation
The structural-regime argument is not promoted automatically from a positive subgroup coefficient. The decision rule requires pooled differences, multiplicity-adjusted evidence, acceptable identification diagnostics, adjustment stability, adequate power, and influence/geographic checks.

## Outputs
- `structural_profile_2012_2017.csv` and `structural_profile_audit.csv`
- `exposure_balance.csv`, `exposure_smd.csv`, `exposure_correlations.csv`, `pre_outcome_process_tests.csv`, and `structural_vif.csv`
- `continuous_structural_moderator_tests.csv` and `progressive_adjustment_models.csv`
- `structural_regime_selection.csv`, `structural_regime_profiles.csv`, and `structural_regime_assignments.csv`
- `structural_regime_specific_coefficients.csv`, `structural_regime_difference_tests.csv`, and `structural_regime_power_mde.csv`
- `structural_regime_event_study_coefficients.csv`, `structural_regime_event_study_pretrend_tests.csv`, `structural_regime_placebo_tests.csv`, and `structural_regime_leave_one_country_out.csv`
- `omitted_confounding_sensitivity.csv` and `full_structural_multiplicity_family.csv`
