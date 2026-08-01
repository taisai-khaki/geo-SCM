# Corrected market-connectivity hypothesis extension

## Design correction

The primary intensive-margin design uses only 2015-2022 observations. The incumbent destination set and baseline destination shares are constructed from 2015-2017, so no post-period or future relationship information enters the 2015-2017 pre-period outcomes. A separate sensitivity design uses the full 2012-2022 model window with an incumbent set and baseline shares constructed from 2012-2017.

The primary openness interaction remains `b=0.024791` with 999-replication wild-bootstrap `p=0.008`, based on 181 countries and 1,991 observations. The corrected primary intensive models have 1,448 observations across 181 countries and eight years. The sensitivity models have 1,991 observations across 181 countries and eleven years.

## Intensive-margin results

The substantive conclusion is stable across incumbent definitions:

- Primary 2015-2017 incumbent set, incumbent diversification: `b=0.021398`, `p=0.003`.
- Primary 2015-2017 incumbent set, incumbent entropy: `b=0.067151`, `p=0.010`.
- Sensitivity 2012-2017 incumbent set, incumbent diversification: `b=0.022681`, `p=0.002`.
- Sensitivity 2012-2017 incumbent set, incumbent entropy: `b=0.067562`, `p=0.002`.

Portfolio reallocation and incumbent retention remain statistically unsupported in both designs. Continuing export share is negative but not conventionally significant in the primary design (`p=.078`) or sensitivity (`p=.152`). The defensible interpretation is stable openness-conditioned rebalancing across incumbent partners, not demonstrated new relationship formation, generalized retention, or a broad portfolio-reallocation effect.

## Channel correction

The export/import channel models now use the exact primary sample: 181 countries and 1,991 observations before prespecified exclusions. The raw Atlas components produce:

- Export intensity: `b=0.027728`, individual wild-bootstrap `p=0.113`; joint `p=0.337`.
- Import intensity: `b=0.457784`, individual wild-bootstrap `p=0.025`; joint `b=0.546129`, `p=0.007`.
- Export-minus-import equality test: `b=-0.533261`, `p=0.010`.

Import interaction estimates remain positive in every requested transformation and exclusion specification, including log(1+x), 1 percent winsorization, exclusion of the highest import-intensity 1 percent and 5 percent, and exclusion of the smallest 5 percent by real GDP. Joint import estimates also retain a positive sign. However, statistical precision weakens in some tail-exclusion specifications, and the export-minus-import difference is not consistently supported outside the raw specification.

The raw export/import components sum to an internally compatible Atlas trade-intensity measure, which is not identical to the WDI total-openness moderator used in the primary model. A separate WDI-compatible specification allocates the WDI primary total across observed Atlas export/import shares; it is included to make the source distinction transparent. Accordingly, the channel evidence should be described as a suggestive upstream-integration association, not as a definitive causal decomposition of the primary total-openness coefficient.

## Multiplicity and validation

The corrected primary family contains 12 unique tests and 5 FDR-adjusted `q<.05` results. The five sensitivity intensive models and channel stress tests are not added as duplicate confirmatory families. Repeated regional subsamples remain stability analysis, not independent out-of-sample validation.

## Key files

- `market_connectivity_intensive_margin_tests.csv`: corrected primary 2015-2022 intensive-margin estimates.
- `market_connectivity_intensive_margin_sensitivity.csv`: 2012-2022 sensitivity using the 2012-2017 incumbent set.
- `intensive_margin_country_year_outcomes.csv`: primary country-year outcomes.
- `intensive_margin_country_year_outcomes_sensitivity.csv`: sensitivity country-year outcomes.
- `intensive_margin_measure_definitions.csv`: both design definitions and windows.
- `market_connectivity_channel_decomposition.csv`: raw exact-primary-sample channel estimates.
- `market_connectivity_channel_stress_tests.csv`: all individual, joint, and equality stress estimates.
- `market_connectivity_channel_stress_summary.csv`: wide specification-level comparison, sample sizes, and sign indicators.
- `market_connectivity_channel_constructs.csv`: raw, WDI-compatible, and source-compatibility constructs.
- `market_connectivity_multiplicity_family.csv`: corrected 12-test family and q-values.
