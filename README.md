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
