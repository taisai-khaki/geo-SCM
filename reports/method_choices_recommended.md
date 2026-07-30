# Recommended Method Choices (Final)

This note records the final methodological choices to keep the paper internally consistent with the implemented analysis.

## Primary Confirmatory Specification (recommended)
- Estimation window: `2012-2022` (model-specific complete-case ranges).
- Main control set: **exclude natural resource rents** in primary confirmatory models.
  - Rationale: keeps 2022 in-sample for H2/H3 and avoids mechanically truncating all models at 2021.
- GPR handling: **profile-based country imputation** from observed country-specific GPR (`gpr_country_annual`) using year-specific weighted KNN over country profile features (`k=5`, inverse-distance weights).
  - First pass requires at least 2 common profile features.
  - If no match, second pass allows 1 common feature.
  - No global-index blanket replacement used in the primary path.
- Inference for H1-H3: wild-cluster bootstrap with **999 replications**.
- Multiple testing: Benjamini-Hochberg FDR across confirmatory family.

## Why this is the recommended primary path
- It directly addresses the 2022 sample truncation issue.
- It avoids treating global GPR as a universal substitute for missing country GPR.
- It uses stronger finite-sample inference for DiD slope terms (999 bootstrap).

## Executed output package
- Folder: `reports/confirmatory_hypotheses_profile_knn_no_rents_999`
- Key files:
  - `confirmatory_primary_tests.csv`
  - `confirmatory_moderation_tests.csv`
  - `confirmatory_pretrend_placebo.csv`
  - `confirmatory_multiple_testing.csv`
  - `confirmatory_robustness_layer.csv`
  - `hypothesis_test_matrix.csv`
  - `gpr_profile_knn_audit_by_year_method.csv`

## High-level outcome from this recommended specification
- H1-H5 remain `NOT_SUPPORTED` in strict confirmatory criteria.
- H1 placebo remains significant, so H1 identification remains fragile.
- This supports a transparent paper structure: confirmatory null/weak results + clearly labeled exploratory heterogeneity.
