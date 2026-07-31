# Capability-Conversion Threshold Redesign

## Classification

H6 is a theory-refining, exploratory analysis. The low-complexity pattern was discovered in this dataset, so this package does not relabel it as confirmatory evidence.

## H6 Statement

H6: Following tariff-induced disruption, increases in productive complexity are associated with weaker export recovery among countries below a minimum level of baseline productive complexity; this negative relationship attenuates once countries possess a sufficiently broad productive capability base.

## Design Corrections

- ECI and COI are frozen at their 2015-2017 country averages.
- The primary exposure is continuous pre-shock US-China trade-nexus intensity, standardized across countries. Top-tercile and top-quartile exposure are robustness checks.
- The primary recovery outcome is log export recovery. Ratio, winsorized-ratio, and pretrend-adjusted outcomes are robustness checks.
- The primary model has country and year fixed effects and does not condition on contemporaneous controls that may be post-treatment. A pre-shock-controls-by-year sensitivity is included.
- H5 is not used as primary evidence because country GPR is directly observed for only a limited country subset and no deterministic imputation is used here.

## Threshold

- Selected baseline-ECI breakpoint: -0.9010.
- Wild-bootstrap 95% percentile interval over the threshold grid: -0.9010 to 0.7518.
- Threshold search grid: 0.20 to 0.80 by 0.05.

## Primary H6 Tests

- H6a_beta_low_lt_0: estimate=0.0108, wild-bootstrap p=0.536, FDR q=0.7968.
- H6b_beta_high_minus_low_gt_0: estimate=-0.0357, wild-bootstrap p=0.594, FDR q=0.7968.
- Directional pattern (both H6 conditions): False.
- Both primary tests survive FDR across the complete reported H6 exploration family: False.

## Robustness and Internal Validation

- Share of non-primary H6 test rows with the predicted directional sign: 0.250.
- Repeated World Bank region-stratified country holdouts: 100/100 successful splits.
- Holdout share with negative low-regime slope: 0.510.
- Holdout share with positive high-minus-low contrast: 0.460.
- Holdout share with both predicted directions: 0.400.

## Redesigned H1-H5 Context

The redesigned H1-H3 table uses frozen pre-shock capability measures, continuous exposure, signed GVC change, log export recovery, and diversification that excludes the United States and China. The equivalence/MDE table distinguishes precise near-zero estimates from imprecise ones.
- H4 omnibus joint test across outcome-specific interactions: chi2=0.9532, p=0.8126.
- Directly observed country-GPR coverage ranges from 0.190 to 0.192 of country-year rows by year.

## Measurement Scope

- The local bilateral source is country-partner-year data, not HS product-level data. It supports continuous nexus exposure, US-market dependence, China import dependence, diversification excluding US/China, and destination-entry mechanisms, but not a valid tariff-weighted product exposure measure.
- The direct mechanism test therefore focuses on market search and redirection (new and persistent non-US/China export destinations). Product reallocation and input-substitution mechanisms require a product-level trade/input dataset.
- GPR profile validation uses only frozen pre-shock covariates; it does not feed imputed values into the primary models.

## Files

- h6_threshold_search.csv and h6_threshold_bootstrap_distribution.csv: breakpoint search and uncertainty.
- h6_primary_and_robustness_tests.csv: all H6 directional tests, 999-replication wild-bootstrap p-values, and full-family BH q-values.
- h6_holdout_validation.csv: repeated country-holdout validation.
- redesigned_h1_h3_tests.csv, h4_redesigned_tests.csv, and h4_omnibus_joint_test.csv: redesigned baseline and moderation analyses.
- mechanism_destination_entry_tests.csv: direct market-search/redirection tests.
- gpr_observed_coverage.csv and gpr_pre_shock_profile_validation.csv: GPR coverage and no-outcome-leakage imputation audit.
- figure_h6_capability_conversion_threshold.png: marginal exposure effect over baseline ECI.