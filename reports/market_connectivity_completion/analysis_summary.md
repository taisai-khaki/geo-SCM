# Market-connectivity hypothesis extension

## Frozen design

Hypothesis: pre-shock trade openness positively moderates the relationship between productive complexity and post-shock export-partner diversification among countries exposed to the US-China tariff conflict.

Primary estimand: `ECI_pre x Exposure_pre x Post x Openness_pre`.
Primary outcome: partner diversification excluding the United States and China (`1-HHI`).

The primary estimate is `b=0.024791` with 999-replication wild-bootstrap `p=0.008`, based on 1,991 observations and 181 countries. The result is a focused extension within the same country-year dataset and shock design; it is not preregistered and is not an independent confirmation.

## Mandatory computations

### Downstream versus upstream integration

Pre-shock export intensity is exports divided by GDP and represents downstream external integration. Pre-shock import intensity is imports divided by GDP and represents upstream sourcing integration. The individual four-way interactions are:

- Export intensity: `b=0.020925`, wild-bootstrap `p=0.244`.
- Import intensity: `b=0.431688`, wild-bootstrap `p=0.030`.
- Joint export-minus-import equality test: `b=-0.438055`, wild-bootstrap `p=0.066`.

The evidence is therefore more consistent with an upstream sourcing channel than a downstream export-intensity channel, although the channel difference is not conventionally significant at the 5 percent level.

### Extensive versus intensive adaptation

New-destination formation remains the extensive-margin mechanism and is not supported by the conditional mechanism tests. The new intensive-margin estimates show:

- Incumbent-partner diversification: `b=0.022950`, wild-bootstrap `p=0.002`.
- Incumbent-partner entropy: `b=0.066403`, wild-bootstrap `p=0.006`.
- Relationship-portfolio reallocation: `b=-0.014114`, wild-bootstrap `p=0.387`.
- Incumbent destination retention: `b=-0.002637`, wild-bootstrap `p=0.545`.
- Continuing export share: `b=-0.011520`, wild-bootstrap `p=0.096`.

The most defensible interpretation is that the openness-conditioned association is concentrated in a more even distribution of flows across incumbent partners, not in demonstrable new-destination formation, measured share reallocation, or simple retention.

### Timing

The phase-specific four-way estimates are:

- Tariff onset, 2018-2019: `b=0.007530`, wild-bootstrap `p=0.302`.
- Pandemic overlap, 2020-2021: `b=0.032500`, wild-bootstrap `p=0.002`.
- Persistence, 2022: `b=0.043896`, wild-bootstrap `p=0.053`.

The omnibus equality test has wild-bootstrap `p=0.074`. The pattern is delayed and strongest during the compound tariff-pandemic period, so the paper should use a compound-disruption or post-shock reconfiguration interpretation rather than claiming an immediate tariff-only effect.

### Marginal effects and robustness

Bootstrap confidence intervals for ECI marginal effects at low, median, and high exposure and openness are in `market_connectivity_marginal_effects.csv`, with the corresponding figure in `figure_market_connectivity_marginal_effects.png`. Openness robustness specifications are in `market_connectivity_openness_robustness.csv`: log openness (`p=0.088`), 1 percent winsorization (`p=0.034`), exclusion of the highest-openness 1 percent (`p=0.033`), exclusion of the highest-openness 5 percent (`p=0.058`), and exclusion of the smallest 5 percent by real GDP (`p=0.039`).

### Multiplicity and stability terminology

The fixed family has 12 unique tests and five FDR-adjusted `q<0.05` results. Duplicate empirical rows are counted once. Channel decomposition, phase decomposition, marginal effects, openness robustness, and influence diagnostics are reported as diagnostics or robustness evidence rather than silently added to the confirmatory family.

The former holdout analysis is labeled `region-stratified repeated subsample stability analysis`. It re-estimates the coefficient on retained countries and is not out-of-sample validation. Leave-one-country and leave-one-region results are also stability diagnostics, not independent validation.

## Key files

- `market_connectivity_channel_constructs.csv`: frozen downstream export and upstream import intensity constructs.
- `market_connectivity_channel_decomposition.csv`: individual, joint, and equality tests.
- `intensive_margin_country_year_outcomes.csv`: country-year incumbent relationship outcomes.
- `intensive_margin_measure_definitions.csv`: operational definitions for intensive outcomes.
- `market_connectivity_intensive_margin_tests.csv`: openness-conditioned intensive-margin models.
- `market_connectivity_phase_results.csv` and `market_connectivity_phase_equality_tests.csv`: timing estimates and equality tests.
- `market_connectivity_marginal_effects.csv` and `figure_market_connectivity_marginal_effects.png`: interpretable simple effects and bootstrap intervals.
- `market_connectivity_openness_robustness.csv`: alternative openness specifications and sample exclusions.
- `market_connectivity_multiplicity_family.csv`: corrected unique test family and FDR q-values.
- `mandatory_computations_metadata.json`: seed, Python version, requested/successful bootstrap count, and primary result metadata.