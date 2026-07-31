# Structural-Regime Completion Package

This directory contains the completed ten-step structural-regime extension. It is separate from the earlier tariff-weighted confirmatory package and uses the frozen repository commit recorded in `structural_analysis_metadata.csv` and `structural_design_protocol.md`.

## Read first

1. `analysis_summary.md` for the concise result and interpretation.
2. `structural_design_protocol.md` for the pre-analysis rules.
3. `structural_profile_2012_2017.csv` and `structural_profile_audit.csv` for the country-level pre-shock dataset.
4. `full_structural_multiplicity_family.csv` for the authoritative 84-test BH correction.
5. `structural_regime_difference_tests.csv`, `structural_regime_power_mde.csv`, and the event-study/placebo files for the decision-rule checks.

## Completed requirements

- Structural variables are constructed from 2012-2017 information only: GDP level, GDP-per-capita growth and volatility, manufacturing share, openness, rents, institutions, export concentration, population, and region.
- Exposure balance, standardized mean differences, exposure correlations, pre-shock outcome-process tests, confounder screening, and VIF diagnostics are included.
- Ten continuous structural moderators are tested separately across three outcomes with wild-cluster bootstrap inference.
- Models A-E compare fixed effects, structural-variable-by-year adjustment, region-by-year adjustment, country-specific trends, and continuous GPS overlap weights.
- Two-versus-three outcome-independent structural clusters are evaluated using silhouette, within-cluster SSE, 100-replication membership stability, counts, exposure composition, and profiles. The selected solution is `k=2`.
- Pooled regime models include regime-by-year fixed effects, formal pairwise and omnibus tests, 999-replication bootstrap inference, leave-one-country-out estimates, and regime-specific MDEs.
- Each retained regime has event-study coefficients, joint pretrend tests, placebo tests, geographic composition, exposure counts, TiVA coverage, and influence diagnostics.
- Omitted-confounding diagnostics report approximate Oster coefficient stability and partial-R2 benchmarks. They are not used to elevate any result automatically.

## Current result

The full structural family contains 84 model tests. No test has BH-adjusted `q < .05`. The structural regime evidence therefore remains exploratory and should not replace the original hypotheses in the main theory.

## Rebuild

```powershell
python scripts\run_structural_regime_analysis.py --bootstrap-reps 999 --stability-reps 100 --seed 20260731
python scripts\build_final_results_bundle.py
```

The raw World Bank structural source is stored at `data/raw/structural_wdi_2012_2017.csv` and records manufacturing value-added share and population for 2012-2017.
