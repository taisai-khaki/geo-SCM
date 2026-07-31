# Completed Design Analysis

## Purpose
This package completes the six redesign tasks without replacing earlier results. It freezes pre-shock constructs, uses a product-overlap tariff treatment, corrects outcome definitions, uses pooled formal E1-E3 contrasts, reports configured country wild-bootstrap inference, and applies Benjamini-Hochberg correction across every reported inferential test.

## Design Decisions
- The causal sample excludes the United States and China because they are the direct policy parties; 229 third countries remain in the panel before outcome-specific missingness.
- Income group is the modal World Bank analytical classification over 2015-2017, with rare ties resolved using 2017. ECI and COI are frozen 2015-2017 averages; regime cut points are frozen pre-shock terciles.
- The primary treatment is the annual Section 301 duty weighted by a country's pre-shock affected US export basket and China's pre-shock US-market share at HS6. It measures diversion opportunity, not a tariff paid by the third country.
- The China-import product-overlap channel and raw covered-US-basket channel are supplementary. HS6 remains an any-covered-HTS8 approximation, fully audited in the tariff-line files.
- H5 uses directly observed country GPR as the primary exploratory analysis. A same-historical-income-prioritised KNN profile match is reported separately as a no-outcome-leakage sensitivity, not as a replacement for observed data.

## Confirmatory Estimates
- H1: b=0.1434, wild p=0.050
- H2: b=0.0368, wild p=0.298
- H3: b=-0.0064, wild p=0.629
- H4: minimum outcome-specific wild p=0.279; valid stacked joint test: cluster Wald p=0.127.
- H5 remains exploratory because direct country GPR coverage is limited; both direct-observation and profile-matched sensitivity estimates are supplied.

## H6 Threshold
- Primary breakpoint: 0.0917 (pre-shock median; fixed before outcome estimation).
- Outcome-selected breakpoint (exploratory only): -0.9070 (wild 95% selection interval -0.9070 to 0.2816).
- H6a, low-regime ECI slope < 0: b=0.0970, wild p=0.803
- H6b, high-minus-low ECI slope > 0: b=-0.1791, wild p=0.902

## Targeted Exploratory Heterogeneity
- E1, upper-middle-income GVC slope: b=0.0784, wild p=0.823; use E1_upper_middle_income_gvc_omnibus.csv and the contrast rows for the formal subgroup test.
- E2, low-pre-shock-ECI export-recovery slope: b=0.2229, wild p=0.029; use E2_low_eci_export_recovery_omnibus.csv and the contrast rows for the formal subgroup test.
- E3, high-income direct-GPR diversification moderation: b=0.0121, wild p=0.247; the profile-matched sensitivity is separate and explicitly labelled.
- A within-group p-value alone is never treated as a subgroup finding: use the pooled coefficient-difference rows and their full-family q-values.

## Multiplicity And Mechanism
- Complete reported family size: 66 tests.
- Tests with BH q < .05: 0.
- The destination-entry models directly test market search and redirection rather than inferring it only from aggregate diversification. See destination_entry_mechanism_tests.csv.

## Interpretation Rule
Do not elevate H6 or an exploratory subgroup to primary support unless its directional tests, formal group-difference tests, outcome corrections, and multiplicity-adjusted evidence converge. The report preserves null and contrary results alongside favorable point estimates.
