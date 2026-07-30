# Sufficiency Assessment Against Shared JSCM Paper Design

Assessment date: 2026-05-21

## 1) Data coverage in `2012-2022` (paper window)

- Total rows: `2,533`
- Countries: `231`
- Years: `2012-2022`

Key variable non-missing counts:

- `eci`: `2,529 / 2,533`
- `coi`: `2,529 / 2,533`
- `export_recovery_index`: `2,529 / 2,533`
- `partner_diversification_1_minus_hhi`: `2,529 / 2,533`
- `tiva_fexgr_dva_share`: `880 / 2,533`
- `delta_tiva_fexgr_dva_share`: `800 / 2,533`

Interpretation:

- Atlas-based variables are near-complete.
- TiVA-based forward-linkage variable is available for `80` countries (not all countries).

## 2) Balanced-panel feasibility (`2012-2022`)

Countries with complete `2012-2022` coverage for:

- `eci`, `coi`, `export_recovery_index`, `partner_diversification_1_minus_hhi`: `229` countries
- TiVA forward-linkage variable (`tiva_fexgr_dva_share`): `80` countries
- TiVA forward-linkage annual change (`delta_tiva_fexgr_dva_share`, 2013-2022): `80` countries

Core sample intersection (all above together): `80` countries.

## 3) Treatment construction feasibility (US/China exposure pre-period)

Using bilateral trade (`exports + imports`) and pre-period `2015-2017`:

- Countries with a valid 3-year exposure metric: `230`
- Exposed (top tertile): `77`
- Non-exposed: `153`

Within the 80-country core sample:

- Exposed: `22`
- Non-exposed: `58`

Interpretation:

- Exposure and `Post × Exposed` can be constructed reliably.
- Treated/control split exists, but treated size is modest inside the TiVA-complete sample.

## 4) Post-2022 availability (requested extension)

In `2023-2024`:

- `eci`, `coi`, `export_recovery_index`, partner diversification: available (460 non-missing rows each out of 462)
- TiVA forward-linkage variable: `0` rows

Interpretation:

- Post-2022 extension is valid for Atlas-based variables.
- Any model requiring TiVA forward-linkage cannot be extended beyond 2022 with current OECD TiVA feed.

## 5) Sufficiency verdict vs paper-like analysis

What is sufficient now:

- Reproducing/exporting paper-style variables for ECI/COI, ERI, and partner diversification.
- Building treatment exposure from bilateral trade and running DiD-style models on those variables.

What is not sufficient yet for a full close replication of the shared paper specification:

- TiVA-based DV for all countries (limited to 80-country coverage).
- Post-2022 TiVA-based DV (not available in current TiVA source).
- Additional covariates referenced in the paper but not yet integrated in this databank (e.g., WDI/WGI/GPR/tariff and related controls).
