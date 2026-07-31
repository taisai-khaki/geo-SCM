# Structural-Regime Pre-Analysis Protocol

Fixed repository commit: `e2be0ec89bb2a4e78734fba6eec86e7037b0d657`.

This extension is frozen before outcome-regime models are estimated. The country-level structural profile uses only 2012-2017 information. ECI, COI, treatment exposure, resilience outcomes, GPR, and post-2017 classifications are excluded from the clustering feature matrix. Region is retained for composition diagnostics but is not one-hot encoded into the numeric clustering matrix. ECI remains the focal explanatory variable.

Primary outcomes:

- adverse deviation from the 2015-2017 forward-GVC linkage mean;
- log export recovery relative to the 2015-2017 export baseline;
- partner diversification excluding the United States and China.

Structural candidates:

`pre_log_real_gdp_pc, pre_real_gdp_pc_growth_mean, pre_real_gdp_pc_growth_slope, pre_real_gdp_pc_growth_volatility, pre_manufacturing_value_added_share, pre_trade_openness, pre_resource_rents, pre_institutional_quality, pre_export_concentration, pre_log_population`.

The primary continuous exposure is the frozen pre-shock US-China trade-nexus measure (`exposure_pre`), standardized across the eligible third-country sample. The primary post indicator begins in 2018; 2019 onward is a sensitivity period. Model controls are country and year fixed effects, with structural-variable-by-year adjustment, region-by-year adjustment, country-specific linear trends, and continuous GPS overlap weights as separate sensitivity models. Contemporaneous post-shock controls are not used.

Missing structural cells are median-imputed only for the unsupervised clustering step, with imputation counts retained in the assignments audit. The inferential family includes continuous structural moderator coefficients, progressive-adjustment focal coefficients, pooled regime-specific coefficients and contrasts, event-study pretrend joint tests, and placebo tests. Descriptive exposure-balance, correlation, VIF, and regime-composition diagnostics are not treated as confirmatory hypothesis tests. All model tests use one Benjamini-Hochberg correction. Wild-cluster bootstrap inference uses 999 Rademacher replications.

Decision rule: a structural regime enters the main theory only if the pooled regime-difference or moderator test survives the complete-family correction, pretrends and placebo tests are acceptable, signs are stable across adjustment models, no single country or region drives the estimate, the regime has adequate exposed-country counts and power, and the pattern appears in a corrected outcome. Otherwise it remains exploratory or appendix-only.
