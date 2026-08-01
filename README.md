# JSCM Databank Builder

This project builds a country-year databank aligned with the paper's data streams:

- Harvard Atlas of Economic Complexity (country-year and bilateral trade tables)
- OECD TiVA (principal shares, selected measures)
- UN Comtrade/BACI trade backbone (through Atlas harmonized trade tables)

## Run

```powershell
cd C:\Users\L03128674\projects\geo-SCM
python .\scripts\build_databank.py --force-download
```

Optional year controls:

```powershell
python .\scripts\build_databank.py --start-year 2012 --end-year 2024 --force-download
```

## Core Outputs

- `data/processed/databank_country_year_<start>_<end>.csv`
- `data/processed/source_atlas_country_year_<start>_<end>.csv`
- `data/processed/source_atlas_country_country_year_<start>_<end>.csv`
- `data/processed/source_oecd_tiva_mainsh_<start>_<end>_selected.csv`
- `reports/data_dictionary.csv`
- `reports/data_quality_summary.json`
- `metadata/source_manifest.json`
- `metadata/source_paper/Paper__JSCM_Wiley.pdf`
- `reports/final_results_all_in_one.json`
- `reports/method_choices_recommended.md`
- `reports/section3_research_design_methodology_rewrite.tex`

## Controls + DiD

Build control datasets (WDI/WGI/GPR/tariff), assemble regression panel, and run paper-style DiD models:

```powershell
python .\scripts\build_controls_and_run_did.py --start-year 2012 --end-year 2022
```

Key outputs:

- `data/processed/regression_panel_2012_2022.csv`
- `reports/did_model_results.csv`
- `reports/did_model_results.md`
- `reports/did_model_summaries.txt`
- `reports/regression_build_metadata.json`

## Comprehensive Tables

Run expanded-country table generation (Table 1-6 style outputs):

```powershell
python .\scripts\run_comprehensive_tables.py --start-year 2012 --end-year 2022
```

Key outputs:

- `reports/comprehensive_table1_descriptive.csv`
- `reports/comprehensive_table1_vif.csv`
- `reports/comprehensive_table2_4_main_models.csv`
- `reports/comprehensive_table2_4_detailed_terms.csv`
- `reports/comprehensive_table5_moderation.csv`
- `reports/comprehensive_table6_robustness.csv`
- `reports/comprehensive_analysis_summary.md`

## Pattern Discovery

Run clustering + visualization + moderator screening:

```powershell
python .\scripts\run_pattern_discovery.py
```

Key outputs:

- `reports/exploratory_patterns/clusters_pca_scatter.png`
- `reports/exploratory_patterns/cluster_profile_heatmap.png`
- `reports/exploratory_patterns/cluster_outcome_summary.csv`
- `reports/exploratory_patterns/moderator_interaction_screen.csv`
- `reports/exploratory_patterns/top_moderator_candidates.csv`

## Cluster-Regime Model

Run formal heterogeneity model with `ECI x Post x Exposed x Cluster` and cluster-specific marginal effects:

```powershell
python .\scripts\run_cluster_regime_models.py
```

Key outputs:

- `reports/cluster_regime_models/cluster_specific_marginal_effects.csv`
- `reports/cluster_regime_models/cluster_specific_effects_plot.png`
- `reports/cluster_regime_models/DV1_GVC_Linkage_Change_interaction_terms.csv`
- `reports/cluster_regime_models/DV2_Export_Recovery_interaction_terms.csv`
- `reports/cluster_regime_models/DV3_Partner_Diversification_interaction_terms.csv`
## Capability-Conversion Redesign

The current conservative analysis is the theory-refining H6 package. It freezes ECI, COI, income groups, and exposure at their 2015-2017 values; uses continuous pre-shock US-China nexus exposure as the primary feasible treatment measure; uses signed GVC change, log export recovery, and diversification excluding the United States and China; and reports 999-replication country wild-bootstrap inference.

Run the full package:

    python .\scripts\run_capability_conversion_analysis.py --bootstrap-reps 999 --threshold-bootstrap-reps 999 --holdout-splits 100

Key outputs:

- `reports/capability_conversion_redesign/analysis_summary.md`
- `reports/capability_conversion_redesign/h6_theory_methods_addendum.tex`
- `reports/capability_conversion_redesign/h6_primary_and_robustness_tests.csv`
- `reports/capability_conversion_redesign/h6_holdout_validation_summary.csv`
- `reports/capability_conversion_redesign/redesigned_h1_h3_tests.csv`
- `reports/capability_conversion_redesign/h4_omnibus_joint_test.csv`
- `reports/capability_conversion_redesign/mechanism_destination_entry_tests.csv`
- `reports/capability_conversion_redesign/figure_h6_capability_conversion_threshold.png`

The package explicitly labels H6 as theory-refining rather than confirmatory. The stored bilateral data are country-partner-year aggregates, so product-level tariff-weighted exposure is not claimed; the report records this constraint and provides continuous exposure plus destination-redirection mechanisms as the valid analyses supported by the current databank.
Current empirical decision: the final 999-replication corrected analysis does not validate H6. Read `reports/capability_conversion_redesign/manuscript_decision_note.md` before treating the theory addendum as a claim for the paper.
## Mandatory Market-Connectivity Computations

Run the final focused market-connectivity extension with 999-replication wild-bootstrap inference:

```powershell
python .\scripts\run_market_connectivity_mandatory_computations.py --bootstrap-reps 999 --seed 20260731
```

This separates downstream export intensity from upstream import intensity, tests intensive versus extensive destination adaptation, estimates tariff/pandemic/persistence phases, reports marginal effects, applies the corrected 12-test unique family, and labels repeated subsamples as stability analysis rather than out-of-sample validation. See `reports/market_connectivity_completion/README.md` for interpretation and file-by-file use.

## Corrected Intensive and Channel Analysis

The mandatory market-connectivity computation now uses a leakage-free 2015-2022 primary intensive-margin window, a 2012-2022 incumbent-set sensitivity, and exact-primary-sample export/import channel stress tests. The primary intensive results support incumbent-partner diversification and entropy across both incumbent definitions. Channel results are directionally suggestive of upstream integration but are not uniformly precise after outlier transformations. See `reports/market_connectivity_completion/README.md`.
