# Completed Design: Source and Reproduction Guide

## Scope

This guide documents the raw inputs for the final pre-shock, tariff-weighted redesign. It is the authoritative analysis package for the completed six-step extension: frozen baseline constructs, channel-specific tariff-weighted exposure, corrected outcomes, H6/E1-E3 tests, equivalence/MDE analysis, full-family FDR correction, and a destination-entry mechanism test.

## Raw source downloads

Run the checksum-verified downloader from the project root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_completed_design_sources.ps1
```

The downloader stores the following files in `data/raw/capability_redesign_sources/`. That directory is ignored by Git because the BACI archive is large and redistribution may be restricted by the original provider.

| File | Provider | Purpose |
|---|---|---|
| `world_bank_historical_income_classifications.xlsx` | World Bank | Frozen 2015-2017 income groups |
| `BACI_HS12_V202601.zip` | CEPII BACI | Bilateral HS6 trade flows and tariff-weighted exposure |
| `ustr_2018_list1_notice.pdf` | USTR | Section 301 List 1 tariff lines |
| `ustr_2018_list2_notice.pdf` | USTR | Section 301 List 2 tariff lines |
| `ustr_2018_list3_notice.pdf` | USTR | Section 301 List 3 tariff lines |
| `ustr_2018_list3_modification.pdf` | USTR | List 3 rate modification |
| `ustr_2019_list4_original_notice.pdf` | USTR | List 4 notice |
| `ustr_2019_list4a_notice.pdf` | USTR | List 4A implementation and rate |

The downloader verifies every file with the SHA-256 values embedded in `scripts/download_completed_design_sources.ps1`.

## Rebuild command

After the raw files are present, rebuild every final output with fixed random seeds:

```powershell
python scripts\run_completed_design_analysis.py --bootstrap-reps 999 --threshold-bootstrap-reps 999 --seed 20260730
python scripts\build_final_results_bundle.py
```

The first command recreates `reports/final_design_completion/`, the processed tariff-weighted exposure profile, figures, and all model tables. The second command writes the comprehensive `reports/final_results_all_in_one.json` package.

## Construction choices

- ECI, COI, income groups, exposure, and regimes are frozen from 2015-2017 data before estimating post-shock effects.
- The primary treatment is the continuous `s_tariff_weighted_us_diversion` measure: each third country's pre-shock US export basket in Section 301-affected HS6 products, weighted by China's pre-shock US import share and the annual tariff schedule.
- The causal sample excludes the direct policy parties, USA and China.
- The GVC primary outcome is adverse-deviation stability. Export recovery is log recovery relative to the pre-shock baseline. Partner diversification excludes USA and China destinations, with entropy/effective-destination checks.
- The HS6 assignment is an any-covered-HTS8 approximation, not an exact tariff-liability measure. The line-count and membership audits are in `reports/final_design_completion/`.
- H5 is reported first with observed country-specific GPR only. A same-historical-income-prioritized KNN profile match is a sensitivity analysis, not a primary replacement for missing country observations.

## Key outputs

See `reports/final_design_completion/README.md` for the authoritative reading order and interpretation of the completed analysis.
