# Manuscript Decision Note: H6 and the Redesigned Evidence

## Bottom Line

Do not add H6 as a supported final hypothesis in the current paper. It is a legitimate theory-refining proposition, but the preregistration-style pooled threshold test does not support it in this databank.

## Why H6 Is Not Validated

The primary specification freezes ECI at the 2015-2017 average, uses continuous pre-shock US-China nexus exposure, log export recovery, country and year fixed effects, and 999-replication country wild-bootstrap inference.

- The selected ECI breakpoint is -0.901, the lower boundary of the admissible threshold grid rather than a stable interior threshold.
- Its wild-bootstrap grid interval is wide: -0.901 to 0.752.
- H6a, the low-regime slope, is +0.0108 rather than negative (wild-bootstrap p = 0.536; BH q = 0.797).
- H6b, the high-minus-low slope difference, is -0.0357 rather than positive (wild-bootstrap p = 0.594; BH q = 0.797).
- Neither the ratio, winsorized-ratio, pretrend-adjusted, small-country-exclusion, weighted, exposure-threshold, nor pre-shock-controls sensitivity produces FDR-significant support.
- Across 100 World-Bank-region-stratified country holdouts, only 40 percent reproduce both predicted signs and only 9 percent obtain both one-sided cluster p-values below 0.05.

These results are inconsistent with a validated capability-conversion threshold. The earlier low-ECI signal appears sensitive to contemporaneous ECI, the explosive ratio outcome, binary exposure construction, and subgroup selection.

## What the Redesign Establishes

The redesign supplies defensible methodological corrections even though it does not rescue the substantive claim.

- Freeze ECI, COI, income/regime variables, and exposure before the shock.
- Prefer continuous pre-shock nexus exposure to a single arbitrary exposure cutoff.
- Use log export recovery as the primary recovery measure and report ratio outcomes only as sensitivity checks.
- Define the primary GVC outcome as signed annual forward-linkage change. Negative absolute change is a different stability construct and should not be mixed into the same results table.
- Use diversification excluding the United States and China to avoid an arithmetic relationship between treatment exposure and the HHI outcome.
- Do not rely on deterministic country-GPR imputation. Direct country GPR covers 44 countries; profile-KNN leave-one-out correlations are only 0.25 to 0.40.
- Do not infer country-class heterogeneity from one subgroup being significant and another not. The pooled income-group equality test is not significant (joint Wald p = 0.503), despite a negative within-LIC coefficient against zero.

## Redesigned Hypothesis Results

The redesigned H1-H3 tests do not support universal positive effects: H1 signed GVC change beta = -0.0134 (wild-bootstrap p = 0.976), H2 log export recovery beta = -0.0013 (p = 0.979), and H3 diversification excluding the US and China beta = -0.0071 (p = 0.743). The H4 omnibus joint test across outcomes is also null (p = 0.813). Observed-GPR-only H5 estimates remain exploratory and are not significant under wild-bootstrap inference.

The direct market-search/redirection mechanisms are likewise not supported: new destinations, their export share, and persistent new destinations all have FDR-adjusted wild-bootstrap q-values of 0.439 or higher.

## Writing Recommendation

Use the redesigned package as the current evidence base. The legacy output remains an audit trail, not the basis for a positive complexity-resilience claim.

The defensible paper options are:

1. A transparent null-results or theory-correction paper explaining why broad country-level capability measures do not translate mechanically into resilience under a mixed disruption/diversion shock.
2. A new study with product-level tariff-weighted disruption and diversion exposure, a pre-specified H6 test, and an independent shock or holdout dataset.

Do not present H6 as supported, and do not frame the failed threshold test as indirect evidence for a capability trap. The meaningful contribution of the current package is the identification of the measurement and design conditions that created the apparent earlier pattern.