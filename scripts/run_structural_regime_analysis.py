from __future__ import annotations

"""Complete the pre-shock structural-regime analysis checklist.

This package is deliberately separate from the completed tariff-weighted design.
It freezes a country-level structural profile using only 2012-2017 information,
then runs exposure diagnostics, continuous effect modification, progressive
adjustment, outcome-independent clustering, pooled regime tests, identification
checks, power diagnostics, and one multiplicity correction over model tests.
"""

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_capability_conversion_analysis import (  # noqa: E402
    cluster_covariance,
    contrast_statistics,
    fit_twfe,
    wild_cluster_bootstrap_contrast,
    zscore,
)


BASELINE_START = 2012
BASELINE_END = 2017
PRIMARY_POST_START = 2018
SENSITIVITY_POST_START = 2019
BOOTSTRAP_DEFAULT = 999
STABILITY_DEFAULT = 100

OUTCOMES = {
    "gvc": ("gvc_adverse_deviation_stability", "Adverse forward-GVC deviation stability"),
    "recovery": ("log_export_recovery", "Log export recovery"),
    "diversification": (
        "partner_diversification_excl_us_china",
        "Partner diversification excluding US and China",
    ),
}

STRUCTURAL_VARS = [
    "pre_log_real_gdp_pc",
    "pre_real_gdp_pc_growth_mean",
    "pre_real_gdp_pc_growth_slope",
    "pre_real_gdp_pc_growth_volatility",
    "pre_manufacturing_value_added_share",
    "pre_trade_openness",
    "pre_resource_rents",
    "pre_institutional_quality",
    "pre_export_concentration",
    "pre_log_population",
]

STRUCTURAL_LABELS = {
    "pre_log_real_gdp_pc": "Log real GDP per capita",
    "pre_real_gdp_pc_growth_mean": "Mean real GDP-per-capita growth",
    "pre_real_gdp_pc_growth_slope": "Real GDP-per-capita growth trend",
    "pre_real_gdp_pc_growth_volatility": "GDP-per-capita growth volatility",
    "pre_manufacturing_value_added_share": "Manufacturing value-added share",
    "pre_trade_openness": "Trade openness",
    "pre_resource_rents": "Natural-resource-rent dependence",
    "pre_institutional_quality": "Institutional quality",
    "pre_export_concentration": "Export concentration",
    "pre_log_population": "Log population",
}

WDI_INDICATORS = {
    "manufacturing_value_added_share": "NV.IND.MANF.ZS",
    "population": "SP.POP.TOTL",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def slope(values: pd.Series, years: pd.Series) -> float:
    mask = pd.to_numeric(values, errors="coerce").notna() & pd.to_numeric(years, errors="coerce").notna()
    if int(mask.sum()) < 3:
        return np.nan
    x = pd.to_numeric(years.loc[mask], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(values.loc[mask], errors="coerce").to_numpy(dtype=float)
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return 0.0 if np.nanstd(y) == 0 else np.nan
    return float(np.polyfit(x, y, 1)[0])


def correlation(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 4 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan, np.nan, int(len(frame))
    result = stats.pearsonr(frame["x"], frame["y"])
    return float(result.statistic), float(result.pvalue), int(len(frame))


def fetch_wdi_indicator(indicator: str) -> pd.DataFrame:
    url = (
        "https://api.worldbank.org/v2/country/all/indicator/"
        f"{indicator}?date={BASELINE_START}:{BASELINE_END}&format=json&per_page=20000"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "geo-SCM-structural-analysis/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = []
    for item in payload[1]:
        if item.get("value") is None or not item.get("countryiso3code"):
            continue
        rows.append(
            {
                "country_iso3_code": item["countryiso3code"],
                "year": int(item["date"]),
                "indicator": indicator,
                "value": float(item["value"]),
            }
        )
    return pd.DataFrame(rows)


def load_wdi_structural_source(base_dir: Path) -> tuple[pd.DataFrame, Path, bool]:
    raw_dir = base_dir / "data" / "raw"
    ensure_dir(raw_dir)
    source_path = raw_dir / "structural_wdi_2012_2017.csv"
    downloaded = False
    if source_path.exists():
        source = pd.read_csv(source_path)
    else:
        pieces = [fetch_wdi_indicator(code) for code in WDI_INDICATORS.values()]
        source = pd.concat(pieces, ignore_index=True)
        source.to_csv(source_path, index=False)
        downloaded = True
    return source, source_path, downloaded


def mode_value(values: pd.Series) -> Any:
    values = values.dropna()
    if values.empty:
        return np.nan
    return values.mode().iloc[0]


def assign_terciles(values: pd.Series) -> pd.Series:
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=3, labels=["low", "middle", "high"]).astype("string")


def build_structural_profile(
    panel: pd.DataFrame, wdi: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = panel.loc[panel["analysis_eligible_third_country"].eq(1)].copy()
    eligible["year"] = pd.to_numeric(eligible["year"], errors="coerce").astype(int)
    pre = eligible.loc[eligible["year"].between(BASELINE_START, BASELINE_END)].copy()
    wdi_wide = (
        wdi.pivot_table(index=["country_iso3_code", "year"], columns="indicator", values="value", aggfunc="mean")
        .reset_index()
        .rename(
            columns={
                WDI_INDICATORS["manufacturing_value_added_share"]: "manufacturing_value_added_share",
                WDI_INDICATORS["population"]: "population",
            }
        )
    )
    pre = pre.merge(wdi_wide, on=["country_iso3_code", "year"], how="left")

    records: list[dict[str, Any]] = []
    for country, group in pre.groupby("country_iso3_code", sort=True):
        group = group.sort_values("year")
        log_gdp = np.log(pd.to_numeric(group["wdi_gdp_pc_const_2015_usd"], errors="coerce").where(lambda x: x > 0))
        growth = log_gdp.diff()
        row: dict[str, Any] = {
            "country_iso3_code": country,
            "country_name": mode_value(group["country_name"]),
            "wb_region": mode_value(group["wb_region"]),
            "pre_log_real_gdp_pc": float(log_gdp.mean()),
            "pre_real_gdp_pc_growth_mean": float(growth.mean()),
            "pre_real_gdp_pc_growth_slope": slope(growth, group["year"]),
            "pre_real_gdp_pc_growth_volatility": float(growth.std(ddof=1)),
            "pre_manufacturing_value_added_share": float(pd.to_numeric(group["manufacturing_value_added_share"], errors="coerce").mean()),
            "pre_trade_openness": float(pd.to_numeric(group["wdi_trade_openness_pct_gdp"], errors="coerce").mean()),
            "pre_resource_rents": float(pd.to_numeric(group["wdi_natural_resource_rents_pct_gdp"], errors="coerce").mean()),
            "pre_institutional_quality": float(pd.to_numeric(group["wgi_institutional_quality_composite"], errors="coerce").mean()),
            "pre_export_concentration": float(pd.to_numeric(group["partner_hhi_export"], errors="coerce").mean()),
            "pre_log_population": float(np.log(pd.to_numeric(group["population"], errors="coerce").where(lambda x: x > 0)).mean()),
            "eci_pre": float(pd.to_numeric(group["eci_pre_raw"], errors="coerce").mean()),
            "coi_pre": float(pd.to_numeric(group["coi_pre_raw"], errors="coerce").mean()),
            "exposure_pre": float(pd.to_numeric(group["nexus_exposure_pre"], errors="coerce").mean()),
        }
        for key, (outcome, _) in OUTCOMES.items():
            values = pd.to_numeric(group[outcome], errors="coerce")
            row[f"{key}_pre_mean"] = float(values.mean())
            row[f"{key}_pre_slope"] = slope(values, group["year"])
            row[f"{key}_pre_n"] = int(values.notna().sum())
        records.append(row)

    profile = pd.DataFrame(records)
    profile["exposure_tercile"] = assign_terciles(profile["exposure_pre"])
    profile["eci_tercile_legacy_reference"] = assign_terciles(profile["eci_pre"])
    profile["coi_tercile_legacy_reference"] = assign_terciles(profile["coi_pre"])
    for variable in STRUCTURAL_VARS + ["eci_pre", "coi_pre", "exposure_pre"]:
        profile[f"z_{variable}"] = zscore(profile[variable])
    profile["structural_complete_case"] = profile[STRUCTURAL_VARS].notna().all(axis=1).astype(int)

    profile_audit = pd.DataFrame(
        [
            {
                "variable": variable,
                "label": STRUCTURAL_LABELS[variable],
                "n_countries": int(profile[variable].notna().sum()),
                "missing_countries": int(profile[variable].isna().sum()),
                "source_period": "2012-2017",
                "used_in_cluster_features": 1,
            }
            for variable in STRUCTURAL_VARS
        ]
    )
    outcome_audit = pd.DataFrame(
        [
            {
                "outcome_key": key,
                "outcome": outcome,
                "label": label,
                "n_countries_with_pre_mean": int(profile[f"{key}_pre_mean"].notna().sum()),
                "n_countries_with_pre_slope": int(profile[f"{key}_pre_slope"].notna().sum()),
                "pre_period": f"{BASELINE_START}-{BASELINE_END}",
            }
            for key, (outcome, label) in OUTCOMES.items()
        ]
    )
    return profile, profile_audit, outcome_audit


def run_exposure_diagnostics(profile: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
    balance_rows: list[dict[str, Any]] = []
    smd_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    process_rows: list[dict[str, Any]] = []
    trend_rows: list[dict[str, Any]] = []

    for variable in STRUCTURAL_VARS:
        for group_name, group in profile.groupby("exposure_tercile", dropna=False):
            values = pd.to_numeric(group[variable], errors="coerce")
            balance_rows.append(
                {
                    "variable": variable,
                    "label": STRUCTURAL_LABELS[variable],
                    "exposure_tercile": group_name,
                    "n": int(values.notna().sum()),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                }
            )
        low = profile.loc[profile["exposure_tercile"].eq("low"), variable].dropna()
        for comparison in ["middle", "high"]:
            other = profile.loc[profile["exposure_tercile"].eq(comparison), variable].dropna()
            pooled = np.sqrt((low.var(ddof=1) + other.var(ddof=1)) / 2.0)
            smd = (other.mean() - low.mean()) / pooled if pooled > 0 else np.nan
            smd_rows.append(
                {
                    "variable": variable,
                    "label": STRUCTURAL_LABELS[variable],
                    "comparison": f"{comparison}-low",
                    "n_low": int(len(low)),
                    "n_comparison": int(len(other)),
                    "standardized_mean_difference": float(smd),
                }
            )
        r, p, n = correlation(profile[variable], profile["exposure_pre"])
        correlation_rows.append(
            {
                "variable": variable,
                "label": STRUCTURAL_LABELS[variable],
                "exposure_measure": "exposure_pre",
                "pearson_r": r,
                "p_value": p,
                "n": n,
            }
        )
        for key, (_, label) in OUTCOMES.items():
            for metric in ["pre_mean", "pre_slope"]:
                r, p, n = correlation(profile[variable], profile[f"{key}_{metric}"])
                process_rows.append(
                    {
                        "variable": variable,
                        "label": STRUCTURAL_LABELS[variable],
                        "outcome_key": key,
                        "outcome_label": label,
                        "outcome_process_metric": metric,
                        "pearson_r": r,
                        "p_value": p,
                        "n": n,
                    }
                )
            for group_name, group in profile.groupby("exposure_tercile", dropna=False):
                values = pd.to_numeric(group[f"{key}_pre_slope"], errors="coerce")
                trend_rows.append(
                    {
                        "outcome_key": key,
                        "outcome_label": label,
                        "exposure_tercile": group_name,
                        "n": int(values.notna().sum()),
                        "mean_pre_shock_slope": float(values.mean()),
                        "sd_pre_shock_slope": float(values.std(ddof=1)),
                    }
                )

    vif_rows: list[dict[str, Any]] = []
    complete = profile[STRUCTURAL_VARS].dropna().copy()
    matrix = complete.to_numpy(dtype=float)
    for index, variable in enumerate(STRUCTURAL_VARS):
        target = matrix[:, index]
        others = np.delete(matrix, index, axis=1)
        design = np.column_stack([np.ones(len(others)), others])
        fitted = design @ np.linalg.pinv(design) @ target
        ss_total = np.sum((target - target.mean()) ** 2)
        r2 = 1.0 - np.sum((target - fitted) ** 2) / ss_total if ss_total > 0 else np.nan
        vif_rows.append(
            {
                "variable": variable,
                "label": STRUCTURAL_LABELS[variable],
                "vif": float(1.0 / (1.0 - r2)) if pd.notna(r2) and r2 < 1 else np.inf,
                "n_complete": int(len(complete)),
            }
        )

    exposure_df = pd.DataFrame(correlation_rows)
    process_df = pd.DataFrame(process_rows)
    candidate_rows = []
    for variable in STRUCTURAL_VARS:
        exposure_p = exposure_df.loc[exposure_df["variable"].eq(variable), "p_value"].min()
        outcome_p = process_df.loc[process_df["variable"].eq(variable), "p_value"].min()
        candidate_rows.append(
            {
                "variable": variable,
                "label": STRUCTURAL_LABELS[variable],
                "exposure_association_p": exposure_p,
                "minimum_pre_outcome_process_p": outcome_p,
                "plausible_confounder_screen": int(pd.notna(exposure_p) and pd.notna(outcome_p) and exposure_p < 0.05 and outcome_p < 0.05),
                "screening_rule": "associated with exposure and at least one pre-shock outcome process at p<.05",
            }
        )

    outputs = {
        "exposure_balance": pd.DataFrame(balance_rows),
        "exposure_smd": pd.DataFrame(smd_rows),
        "exposure_correlations": exposure_df,
        "pre_outcome_process_tests": process_df,
        "pretrend_by_exposure_tercile": pd.DataFrame(trend_rows),
        "structural_vif": pd.DataFrame(vif_rows),
        "confounder_candidates": pd.DataFrame(candidate_rows),
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    return outputs


def add_base_terms(
    frame: pd.DataFrame, exposure_col: str = "z_exposure_pre", post_col: str = "post_2018"
) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    out["post_x_exposure"] = out[post_col] * out[exposure_col]
    out["post_x_eci"] = out[post_col] * out["z_eci_pre"]
    out["eci_exposure_post"] = out[post_col] * out[exposure_col] * out["z_eci_pre"]
    return out, ["post_x_exposure", "post_x_eci", "eci_exposure_post"]


def add_moderator_terms(frame: pd.DataFrame, moderator: str) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    out["post_x_moderator"] = out["post_2018"] * out[moderator]
    out["post_x_exposure_x_moderator"] = out["post_2018"] * out["z_exposure_pre"] * out[moderator]
    out["post_x_eci_x_moderator"] = out["post_2018"] * out["z_eci_pre"] * out[moderator]
    out["eci_exposure_post_x_moderator"] = out["post_2018"] * out["z_exposure_pre"] * out["z_eci_pre"] * out[moderator]
    return out, [
        "post_x_moderator",
        "post_x_exposure_x_moderator",
        "post_x_eci_x_moderator",
        "eci_exposure_post_x_moderator",
    ]


def bootstrap_row(
    fit: Any,
    term: str,
    test_id: str,
    hypothesis: str,
    outcome: str,
    reps: int,
    seed: int,
    alternative: str = "two-sided",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contrast = np.zeros(len(fit.term_names), dtype=float)
    contrast[fit.term_index(term)] = 1.0
    row = contrast_statistics(fit, contrast, alternative=alternative)
    row.update(wild_cluster_bootstrap_contrast(fit, contrast, reps=reps, seed=seed, alternative=alternative))
    row.update(
        {
            "test_id": test_id,
            "hypothesis": hypothesis,
            "outcome": outcome,
            "test_type": "coefficient",
            "expected_direction": "two-sided",
            "alternative": alternative,
            "pvalue_for_fdr": row.get("p_wild_bootstrap", row.get("p_cluster")),
            "n_obs": fit.n_obs,
            "n_countries": fit.n_countries,
            "n_years": fit.n_years,
            "within_r2": fit.r2_within,
            "inference": "wild_cluster_bootstrap",
        }
    )
    if extra:
        row.update(extra)
    return row


def fit_terms(frame: pd.DataFrame, outcome: str, terms: list[str], weight_col: str | None = None) -> Any:
    return fit_twfe(frame, outcome, terms, weight_col=weight_col)


def run_continuous_moderators(
    panel: pd.DataFrame, out_dir: Path, reps: int, seed: int
) -> pd.DataFrame:
    moderators = {f"z_{variable}": STRUCTURAL_LABELS[variable] for variable in STRUCTURAL_VARS}
    rows: list[dict[str, Any]] = []
    for outcome_key, (outcome, outcome_label) in OUTCOMES.items():
        for offset, (moderator, label) in enumerate(moderators.items()):
            frame, terms = add_base_terms(panel)
            frame, mod_terms = add_moderator_terms(frame, moderator)
            terms += mod_terms
            try:
                fit = fit_terms(frame, outcome, terms)
                rows.append(
                    bootstrap_row(
                        fit,
                        "eci_exposure_post_x_moderator",
                        f"continuous_{outcome_key}_{moderator}",
                        "Continuous structural effect modification",
                        outcome_label,
                        reps,
                        seed + offset + 1000 * list(OUTCOMES).index(outcome_key),
                        extra={"outcome_key": outcome_key, "moderator": moderator, "moderator_label": label},
                    )
                )
            except Exception as exc:
                rows.append(
                    {
                        "test_id": f"continuous_{outcome_key}_{moderator}",
                        "hypothesis": "Continuous structural effect modification",
                        "outcome": outcome_label,
                        "outcome_key": outcome_key,
                        "moderator": moderator,
                        "moderator_label": label,
                        "error": str(exc),
                        "pvalue_for_fdr": np.nan,
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "continuous_structural_moderator_tests.csv", index=False)
    return result


def build_region_year_terms(frame: pd.DataFrame, region_col: str = "wb_region") -> list[str]:
    regions = sorted(frame[region_col].dropna().astype(str).unique())
    if len(regions) <= 1:
        return []
    base = regions[0]
    terms: list[str] = []
    for region in regions[1:]:
        for year in sorted(frame["year"].dropna().astype(int).unique()):
            term = f"region_year_{region}_{year}".replace(" ", "_").replace(",", "")
            frame[term] = ((frame[region_col].astype(str) == region) & (frame["year"].eq(year))).astype(float)
            terms.append(term)
    return terms


def build_country_trend_terms(frame: pd.DataFrame, outcome: str) -> list[str]:
    eligible = (
        frame.assign(_outcome=pd.to_numeric(frame[outcome], errors="coerce"))
        .groupby("country_iso3_code")
        ._outcome.count()
    )
    countries = sorted(eligible.loc[eligible >= 3].index.astype(str).tolist())
    if len(countries) <= 1:
        return []
    base = countries[0]
    time = pd.to_numeric(frame["year"], errors="coerce") - BASELINE_START
    terms: list[str] = []
    for country in countries[1:]:
        term = f"country_trend_{country}"
        frame[term] = time * (frame["country_iso3_code"].astype(str) == country).astype(float)
        terms.append(term)
    return terms


def make_gps_overlap_weight(profile: pd.DataFrame) -> pd.Series:
    covariates = profile[STRUCTURAL_VARS].copy()
    complete = covariates.notna().all(axis=1) & profile["exposure_pre"].notna()
    x = covariates.loc[complete].to_numpy(dtype=float)
    y = profile.loc[complete, "exposure_pre"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), StandardScaler().fit_transform(x)])
    beta = np.linalg.pinv(design) @ y
    residual = y - design @ beta
    sd = float(np.std(residual, ddof=1)) or 1.0
    weights = pd.Series(np.nan, index=profile.index, dtype=float)
    weights.loc[complete] = np.exp(-0.5 * (residual / sd) ** 2)
    mean = float(weights.mean())
    return weights / mean if mean > 0 else weights


def run_adjustment_models(panel: pd.DataFrame, profile: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    profile_weights = profile[["country_iso3_code"]].copy()
    profile_weights["gps_overlap_weight"] = make_gps_overlap_weight(profile).to_numpy()
    base_panel = panel.merge(profile_weights, on="country_iso3_code", how="left")
    for outcome_key, (outcome, outcome_label) in OUTCOMES.items():
        for model_name in ["A_FE", "B_structural_x_year", "C_region_x_year", "D_country_trends", "E_gps_overlap"]:
            frame, terms = add_base_terms(base_panel)
            frame, coi_terms = add_moderator_terms(frame, "z_coi_pre")
            terms += coi_terms
            weight_col = None
            if model_name == "B_structural_x_year":
                for variable in STRUCTURAL_VARS:
                    column = f"z_{variable}"
                    for year in sorted(frame["year"].dropna().astype(int).unique())[1:]:
                        term = f"structural_year_{variable}_{year}"
                        frame[term] = frame[column] * frame["year"].eq(year).astype(float)
                        terms.append(term)
            elif model_name == "C_region_x_year":
                terms += build_region_year_terms(frame)
            elif model_name == "D_country_trends":
                terms += build_country_trend_terms(frame, outcome)
            elif model_name == "E_gps_overlap":
                weight_col = "gps_overlap_weight"
            try:
                fit = fit_terms(frame, outcome, terms, weight_col=weight_col)
                for focal, label in [
                    ("eci_exposure_post", "ECI x exposure x post"),
                    ("eci_exposure_post_x_moderator", "ECI x exposure x post x COI"),
                ]:
                    rows.append(
                        bootstrap_row(
                            fit,
                            focal,
                            f"adjustment_{model_name}_{outcome_key}_{focal}",
                            "Progressive confounding adjustment",
                            outcome_label,
                            reps=0,
                            seed=0,
                            extra={
                                "outcome_key": outcome_key,
                                "model": model_name,
                                "focal_term": focal,
                                "focal_label": label,
                                "inference": "country_clustered",
                                "pvalue_for_fdr": fit.term_row(focal)["p_cluster_two_sided"],
                            },
                        )
                    )
            except Exception as exc:
                for focal, label in [
                    ("eci_exposure_post", "ECI x exposure x post"),
                    ("eci_exposure_post_x_moderator", "ECI x exposure x post x COI"),
                ]:
                    rows.append(
                        {
                            "test_id": f"adjustment_{model_name}_{outcome_key}_{focal}",
                            "hypothesis": "Progressive confounding adjustment",
                            "outcome": outcome_label,
                            "outcome_key": outcome_key,
                            "model": model_name,
                            "focal_term": focal,
                            "focal_label": label,
                            "error": str(exc),
                            "pvalue_for_fdr": np.nan,
                        }
                    )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "progressive_adjustment_models.csv", index=False)
    pd.DataFrame(
        [{"weight": "gps_overlap_weight", "definition": "exp(-0.5 * standardized exposure residual squared)", "method": "continuous GPS overlap weighting"}]
    ).to_csv(out_dir / "balancing_weight_definition.csv", index=False)
    return result


def align_labels(reference: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    table = np.zeros((k, k), dtype=int)
    for left, right in zip(reference, labels):
        table[int(left), int(right)] += 1
    rows, cols = linear_sum_assignment(-table)
    mapping = {int(col): int(row) for row, col in zip(rows, cols)}
    return np.array([mapping.get(int(value), int(value)) for value in labels], dtype=int)


def choose_structural_regimes(
    profile: pd.DataFrame, out_dir: Path, seed: int, stability_reps: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, list[str]]:
    cluster_profile = profile.copy()
    imputation_count = cluster_profile[STRUCTURAL_VARS].isna().sum(axis=1)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x = scaler.fit_transform(imputer.fit_transform(cluster_profile[STRUCTURAL_VARS]))
    selection_rows: list[dict[str, Any]] = []
    assignments: dict[int, pd.Series] = {}
    for k in [2, 3]:
        model = KMeans(n_clusters=k, random_state=seed, n_init=50)
        labels = model.fit_predict(x)
        silhouette = float(silhouette_score(x, labels))
        within = float(model.inertia_)
        rng = np.random.default_rng(seed + k)
        aris: list[float] = []
        membership = np.zeros((stability_reps, len(cluster_profile)), dtype=int)
        for rep in range(stability_reps):
            sample_idx = rng.choice(len(cluster_profile), size=max(k * 4, int(0.8 * len(cluster_profile))), replace=False)
            sampled_model = KMeans(n_clusters=k, random_state=seed + 10000 * k + rep, n_init=20)
            sampled_model.fit(x[sample_idx])
            predicted = align_labels(labels[sample_idx], sampled_model.predict(x[sample_idx]), k)
            predicted_all = align_labels(labels[sample_idx], sampled_model.predict(x), k)
            membership[rep] = predicted_all
            aris.append(float(adjusted_rand_score(labels[sample_idx], predicted)))
        assignment_stability = float(np.mean(membership == labels[None, :]))
        stability = float(np.mean(aris))
        selection_rows.append(
            {
                "k": k,
                "n_countries": len(cluster_profile),
                "silhouette": silhouette,
                "within_cluster_sse": within,
                "bootstrap_mean_ari": stability,
                "bootstrap_assignment_stability": assignment_stability,
                "selection_rule": "silhouette primary; if gap <= 0.02, higher bootstrap stability; ties resolve to lower k; no outcomes used",
            }
        )
        assignments[k] = pd.Series(labels, index=cluster_profile.index)
    selection = pd.DataFrame(selection_rows)
    best_silhouette = float(selection["silhouette"].max())
    silhouette_gap = best_silhouette - float(selection["silhouette"].min())
    if silhouette_gap <= 0.02:
        selected_k = int(selection.sort_values(["bootstrap_assignment_stability", "k"], ascending=[False, True]).iloc[0]["k"])
        basis = "silhouette gap <= 0.02; higher bootstrap stability; ties resolve to lower k"
    else:
        selected_k = int(selection.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
        basis = "highest silhouette; ties resolve to lower k"
    selection["silhouette_gap_to_best"] = best_silhouette - selection["silhouette"]
    selection["selection_score"] = selection["silhouette"]
    selection["selection_basis"] = basis
    selected_labels = assignments[selected_k]
    cluster_profile = cluster_profile.copy()
    cluster_profile["structural_cluster_numeric"] = selected_labels.to_numpy()
    order = cluster_profile.groupby("structural_cluster_numeric")["pre_log_real_gdp_pc"].mean().sort_values().index.tolist()
    label_map = {cluster: f"R{index + 1}" for index, cluster in enumerate(order)}
    cluster_profile["structural_regime"] = cluster_profile["structural_cluster_numeric"].map(label_map)
    profile = profile.copy()
    profile["structural_cluster_numeric"] = cluster_profile["structural_cluster_numeric"]
    profile["structural_regime"] = cluster_profile["structural_regime"]
    profile["cluster_imputed_fields"] = imputation_count
    assignments_out = profile[["country_iso3_code", "structural_regime", "cluster_imputed_fields", "exposure_tercile", "wb_region"] + STRUCTURAL_VARS].copy()
    profile_summary = (
        profile.groupby("structural_regime", dropna=False)[STRUCTURAL_VARS].mean().reset_index()
    )
    selection.to_csv(out_dir / "structural_regime_selection.csv", index=False)
    assignments_out.to_csv(out_dir / "structural_regime_assignments.csv", index=False)
    profile_summary.to_csv(out_dir / "structural_regime_profiles.csv", index=False)
    return profile, selection, assignments_out, selected_k, [label_map[c] for c in order]


def wald_test(
    fit: Any, contrasts: np.ndarray, reps: int, seed: int
) -> dict[str, Any]:
    contrasts = np.atleast_2d(contrasts).astype(float)
    covariance = contrasts @ fit.covariance @ contrasts.T
    inverse = np.linalg.pinv(covariance)
    estimate = contrasts @ fit.beta
    statistic = float(estimate @ inverse @ estimate)
    rank = int(np.linalg.matrix_rank(covariance))
    p_cluster = float(stats.chi2.sf(statistic, df=max(rank, 1)))
    p_wild = np.nan
    successes = 0
    if reps > 0:
        restriction_covariance = contrasts @ fit.inv_xx @ contrasts.T
        restricted = fit.beta - fit.inv_xx @ contrasts.T @ np.linalg.pinv(restriction_covariance) @ (contrasts @ fit.beta)
        yhat_null = fit.x_resid @ restricted
        residual_null = fit.y_resid - yhat_null
        rng = np.random.default_rng(seed)
        stars: list[float] = []
        for _ in range(reps):
            signs = rng.choice(np.array([-1.0, 1.0]), size=fit.n_clusters)
            ystar = yhat_null + residual_null * signs[fit.cluster_codes]
            beta_star = fit.inv_xx @ (fit.x_resid.T @ ystar)
            residual_star = ystar - fit.x_resid @ beta_star
            covariance_star, _ = cluster_covariance(
                fit.x_resid,
                residual_star,
                fit.cluster_codes,
                n_clusters=fit.n_clusters,
                k_full=fit.k_full,
                inv_xx=fit.inv_xx,
            )
            star_cov = contrasts @ covariance_star @ contrasts.T
            star_stat = float((contrasts @ beta_star) @ np.linalg.pinv(star_cov) @ (contrasts @ beta_star))
            if np.isfinite(star_stat):
                stars.append(star_stat)
        successes = len(stars)
        if stars:
            p_wild = float((1 + np.sum(np.asarray(stars) >= statistic)) / (len(stars) + 1))
    return {
        "wald_chi2": statistic,
        "wald_df": max(rank, 1),
        "p_cluster": p_cluster,
        "p_wild_bootstrap": p_wild,
        "bootstrap_reps_requested": reps,
        "bootstrap_reps_success": successes,
        "pvalue_for_fdr": p_wild if pd.notna(p_wild) else p_cluster,
    }


def add_regime_terms(frame: pd.DataFrame, regimes: list[str]) -> tuple[pd.DataFrame, list[str], dict[str, np.ndarray]]:
    out, terms = add_base_terms(frame)
    reference = regimes[0]
    contrast_vectors: dict[str, np.ndarray] = {}
    for regime in regimes[1:]:
        mask = out["structural_regime"].eq(regime).astype(float)
        for base in ["post_x_exposure", "post_x_eci", "eci_exposure_post"]:
            term = f"{base}_x_{regime}"
            out[term] = out[base] * mask
            terms.append(term)
    for regime in regimes[1:]:
        for year in sorted(out["year"].dropna().astype(int).unique()):
            term = f"regime_year_{regime}_{year}"
            out[term] = (out["structural_regime"].eq(regime) & out["year"].eq(year)).astype(float)
            terms.append(term)
    for regime in regimes:
        contrast_vectors[regime] = np.zeros(len(terms), dtype=float)
    contrast_vectors[reference][terms.index("eci_exposure_post")] = 1.0
    for regime in regimes[1:]:
        contrast_vectors[regime][terms.index("eci_exposure_post")] = 1.0
        contrast_vectors[regime][terms.index(f"eci_exposure_post_x_{regime}")] = 1.0
    return out, terms, contrast_vectors


def run_regime_models(
    panel: pd.DataFrame, regimes: list[str], out_dir: Path, reps: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], Any]]:
    coefficient_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    fits: dict[tuple[str, str], Any] = {}
    for outcome_index, (outcome_key, (outcome, outcome_label)) in enumerate(OUTCOMES.items()):
        frame, terms, regime_vectors = add_regime_terms(panel, regimes)
        try:
            fit = fit_terms(frame, outcome, terms)
        except Exception as exc:
            contrast_rows.append({"test_id": f"regime_model_error_{outcome_key}", "outcome_key": outcome_key, "error": str(exc), "pvalue_for_fdr": np.nan})
            continue
        fits[(outcome_key, "pooled")] = fit
        for regime_index, regime in enumerate(regimes):
            vector = regime_vectors[regime]
            stats_row = contrast_statistics(fit, vector, alternative="two-sided")
            boot = wild_cluster_bootstrap_contrast(fit, vector, reps=reps, seed=seed + outcome_index * 100 + regime_index, alternative="two-sided")
            coefficient_rows.append(
                {
                    "test_id": f"regime_slope_{outcome_key}_{regime}",
                    "hypothesis": "Structural-regime-specific ECI effect",
                    "outcome_key": outcome_key,
                    "outcome": outcome_label,
                    "regime": regime,
                    "regime_reference": regimes[0],
                    "test_type": "regime_specific_coefficient",
                    **stats_row,
                    **boot,
                    "pvalue_for_fdr": boot["p_wild_bootstrap"],
                    "n_obs": fit.n_obs,
                    "n_countries": fit.n_countries,
                    "n_years": fit.n_years,
                }
            )
        for left, right in itertools.combinations(regimes, 2):
            vector = regime_vectors[right] - regime_vectors[left]
            stats_row = contrast_statistics(fit, vector, alternative="two-sided")
            boot = wild_cluster_bootstrap_contrast(fit, vector, reps=reps, seed=seed + outcome_index * 1000 + len(contrast_rows), alternative="two-sided")
            contrast_rows.append(
                {
                    "test_id": f"regime_difference_{outcome_key}_{right}_minus_{left}",
                    "hypothesis": "Formal structural-regime coefficient difference",
                    "outcome_key": outcome_key,
                    "outcome": outcome_label,
                    "regime_left": left,
                    "regime_right": right,
                    "test_type": "pairwise_regime_difference",
                    **stats_row,
                    **boot,
                    "pvalue_for_fdr": boot["p_wild_bootstrap"],
                    "n_obs": fit.n_obs,
                    "n_countries": fit.n_countries,
                    "n_years": fit.n_years,
                }
            )
        matrix = np.vstack([regime_vectors[r] for r in regimes[1:]])
        joint = wald_test(fit, matrix, reps=reps, seed=seed + outcome_index * 7000)
        contrast_rows.append(
            {
                "test_id": f"regime_omnibus_{outcome_key}",
                "hypothesis": "Joint equality of structural-regime coefficients",
                "outcome_key": outcome_key,
                "outcome": outcome_label,
                "test_type": "joint_regime_equality",
                **joint,
                "n_obs": fit.n_obs,
                "n_countries": fit.n_countries,
                "n_years": fit.n_years,
                "available_regimes": ",".join(regimes),
            }
        )
    coefficients = pd.DataFrame(coefficient_rows)
    contrasts = pd.DataFrame(contrast_rows)
    coefficients.to_csv(out_dir / "structural_regime_specific_coefficients.csv", index=False)
    contrasts.to_csv(out_dir / "structural_regime_difference_tests.csv", index=False)
    return coefficients, contrasts, fits


def run_regime_power(profile: pd.DataFrame, coefficients: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for _, row in coefficients.iterrows():
        regime = row.get("regime")
        count = profile.loc[profile["structural_regime"].eq(regime)]
        exposure_cut = profile["exposure_pre"].quantile(2 / 3)
        exposed = count["exposure_pre"].ge(exposure_cut).sum()
        se = safe_float(row.get("se"))
        rows.append(
            {
                "test_id": row.get("test_id"),
                "outcome_key": row.get("outcome_key"),
                "regime": regime,
                "n_countries": int(len(count)),
                "exposed_countries": int(exposed),
                "n_observations": row.get("n_obs"),
                "estimate": row.get("estimate"),
                "se": se,
                "mde_80pct": float((stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * se) if pd.notna(se) else np.nan,
                "adequate_exposed_count_rule_10": int(exposed >= 10),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "structural_regime_power_mde.csv", index=False)
    return result


def run_event_study_and_placebo(
    panel: pd.DataFrame, regimes: list[str], out_dir: Path, reps: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_rows: list[dict[str, Any]] = []
    pretrend_rows: list[dict[str, Any]] = []
    placebo_rows: list[dict[str, Any]] = []
    for outcome_index, (outcome_key, (outcome, outcome_label)) in enumerate(OUTCOMES.items()):
        for regime_index, regime in enumerate(regimes):
            subset = panel.loc[panel["structural_regime"].eq(regime)].copy()
            available_years = sorted(subset["year"].dropna().astype(int).unique())
            event_terms: list[str] = []
            for year in available_years:
                if year == 2017:
                    continue
                term = f"event_{year}_x_exposure"
                subset[term] = subset["z_exposure_pre"] * subset["year"].eq(year).astype(float)
                event_terms.append(term)
            try:
                fit = fit_terms(subset, outcome, event_terms)
                for year, term in zip([y for y in available_years if y != 2017], event_terms):
                    vector = np.zeros(len(fit.term_names), dtype=float)
                    vector[fit.term_index(term)] = 1.0
                    row = contrast_statistics(fit, vector, alternative="two-sided")
                    event_rows.append(
                        {
                            "test_id": f"event_{outcome_key}_{regime}_{year}",
                            "outcome_key": outcome_key,
                            "outcome": outcome_label,
                            "regime": regime,
                            "event_year": year,
                            "estimate": row["estimate"],
                            "se": row["se"],
                            "p_cluster": row["p_cluster"],
                            "n_obs": fit.n_obs,
                            "n_countries": fit.n_countries,
                        }
                    )
                pre_year_terms = [term for year, term in zip([y for y in available_years if y != 2017], event_terms) if year <= 2016]
                if pre_year_terms:
                    matrix = np.zeros((len(pre_year_terms), len(fit.term_names)), dtype=float)
                    for index, term in enumerate(pre_year_terms):
                        matrix[index, fit.term_index(term)] = 1.0
                    joint = wald_test(fit, matrix, reps=reps, seed=seed + outcome_index * 100 + regime_index)
                    pretrend_rows.append(
                        {
                            "test_id": f"event_pretrend_{outcome_key}_{regime}",
                            "hypothesis": "No differential pre-shock exposure trend within regime",
                            "outcome_key": outcome_key,
                            "outcome": outcome_label,
                            "regime": regime,
                            "test_type": "event_study_pretrend_joint",
                            **joint,
                            "n_obs": fit.n_obs,
                            "n_countries": fit.n_countries,
                        }
                    )
            except Exception as exc:
                pretrend_rows.append({"test_id": f"event_pretrend_{outcome_key}_{regime}", "outcome_key": outcome_key, "regime": regime, "error": str(exc), "pvalue_for_fdr": np.nan})

            placebo = subset.loc[subset["year"].between(BASELINE_START, BASELINE_END)].copy()
            placebo["placebo_post"] = placebo["year"].ge(2015).astype(float)
            placebo["placebo_post_x_exposure"] = placebo["placebo_post"] * placebo["z_exposure_pre"]
            placebo["placebo_post_x_eci"] = placebo["placebo_post"] * placebo["z_eci_pre"]
            placebo["placebo_eci_exposure_post"] = placebo["placebo_post"] * placebo["z_exposure_pre"] * placebo["z_eci_pre"]
            try:
                fit = fit_terms(placebo, outcome, ["placebo_post_x_exposure", "placebo_post_x_eci", "placebo_eci_exposure_post"])
                row = bootstrap_row(
                    fit,
                    "placebo_eci_exposure_post",
                    f"placebo_{outcome_key}_{regime}",
                    "Pre-period pseudo-shock test",
                    outcome_label,
                    reps,
                    seed + 30000 + outcome_index * 100 + regime_index,
                    extra={"outcome_key": outcome_key, "regime": regime, "placebo_post_start": 2015},
                )
                placebo_rows.append(row)
            except Exception as exc:
                placebo_rows.append({"test_id": f"placebo_{outcome_key}_{regime}", "outcome_key": outcome_key, "regime": regime, "error": str(exc), "pvalue_for_fdr": np.nan})
    event = pd.DataFrame(event_rows)
    pretrend = pd.DataFrame(pretrend_rows)
    placebo = pd.DataFrame(placebo_rows)
    event.to_csv(out_dir / "structural_regime_event_study_coefficients.csv", index=False)
    pretrend.to_csv(out_dir / "structural_regime_event_study_pretrend_tests.csv", index=False)
    placebo.to_csv(out_dir / "structural_regime_placebo_tests.csv", index=False)
    return event, pretrend, placebo


def run_regime_composition(profile: pd.DataFrame, panel: pd.DataFrame, regimes: list[str], out_dir: Path) -> pd.DataFrame:
    exposure_cut = profile["exposure_pre"].quantile(2 / 3)
    profile = profile.copy()
    profile["exposed_top_tercile"] = profile["exposure_pre"].ge(exposure_cut).astype(int)
    counts = (
        profile.groupby(["structural_regime", "exposed_top_tercile"], dropna=False)
        .agg(n_countries=("country_iso3_code", "nunique"))
        .reset_index()
    )
    regions = profile.groupby(["structural_regime", "wb_region"], dropna=False).size().reset_index(name="n_countries")
    coverage = (
        panel.groupby("structural_regime", dropna=False)
        .agg(
            n_country_years=("country_iso3_code", "size"),
            tiva_observed=("tiva_fexgr_dva_share", lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum())),
            export_recovery_observed=("log_export_recovery", lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum())),
            diversification_observed=("partner_diversification_excl_us_china", lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum())),
        )
        .reset_index()
    )
    counts.to_csv(out_dir / "structural_regime_exposure_counts.csv", index=False)
    regions.to_csv(out_dir / "structural_regime_geographic_composition.csv", index=False)
    coverage.to_csv(out_dir / "structural_regime_outcome_coverage.csv", index=False)
    return counts


def run_leave_one_out(panel: pd.DataFrame, regimes: list[str], out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    countries = sorted(panel["country_iso3_code"].dropna().astype(str).unique())
    for outcome_key, (outcome, outcome_label) in OUTCOMES.items():
        for excluded in countries:
            subset = panel.loc[panel["country_iso3_code"].ne(excluded)].copy()
            frame, terms, vectors = add_regime_terms(subset, regimes)
            try:
                fit = fit_terms(frame, outcome, terms)
                for regime in regimes:
                    row = contrast_statistics(fit, vectors[regime], alternative="two-sided")
                    rows.append(
                        {
                            "outcome_key": outcome_key,
                            "outcome": outcome_label,
                            "excluded_country": excluded,
                            "regime": regime,
                            "estimate": row["estimate"],
                            "se": row["se"],
                            "p_cluster": row["p_cluster"],
                            "n_obs": fit.n_obs,
                            "n_countries": fit.n_countries,
                        }
                    )
            except Exception as exc:
                rows.append({"outcome_key": outcome_key, "excluded_country": excluded, "error": str(exc)})
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "structural_regime_leave_one_country_out.csv", index=False)
    return result


def run_omitted_confounding(adjustments: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for outcome_key in adjustments["outcome_key"].dropna().unique():
        for focal in ["eci_exposure_post", "eci_exposure_post_x_moderator"]:
            short = adjustments.loc[(adjustments["outcome_key"] == outcome_key) & (adjustments["model"] == "A_FE") & (adjustments["focal_term"] == focal)]
            full = adjustments.loc[(adjustments["outcome_key"] == outcome_key) & (adjustments["model"] == "B_structural_x_year") & (adjustments["focal_term"] == focal)]
            if short.empty or full.empty or "estimate" not in short or "estimate" not in full:
                continue
            short_row = short.iloc[0]
            full_row = full.iloc[0]
            b_short = safe_float(short_row.get("estimate"))
            b_full = safe_float(full_row.get("estimate"))
            r_short = safe_float(short_row.get("within_r2"))
            r_full = safe_float(full_row.get("within_r2"))
            r_max = min(1.0, 1.3 * r_full) if pd.notna(r_full) else np.nan
            denominator = (0.0 - b_full) * (r_full - r_short) if pd.notna(r_full) and pd.notna(r_short) else np.nan
            delta = ((b_full - b_short) * (r_max - r_full) / denominator) if pd.notna(denominator) and denominator != 0 else np.nan
            tvalue = safe_float(short_row.get("tvalue"))
            partial_r2 = (tvalue**2 / (tvalue**2 + max(safe_float(short_row.get("n_countries")) - 1, 1))) if pd.notna(tvalue) else np.nan
            rows.append(
                {
                    "outcome_key": outcome_key,
                    "focal_term": focal,
                    "short_model": "A_FE",
                    "full_model": "B_structural_x_year",
                    "estimate_short": b_short,
                    "estimate_full": b_full,
                    "within_r2_short": r_short,
                    "within_r2_full": r_full,
                    "oster_delta_to_zero_approx": delta,
                    "partial_r2_short_cluster_t_approx": partial_r2,
                    "triggered_by_full_family_survival": 0,
                    "interpretation": "screening diagnostic; no result is promoted unless the full decision rule is met",
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "omitted_confounding_sensitivity.csv", index=False)
    return result


def apply_family(tables: list[pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    usable = []
    for table in tables:
        if table is None or table.empty or "pvalue_for_fdr" not in table:
            continue
        part = table.loc[table["pvalue_for_fdr"].notna()].copy()
        if not part.empty:
            usable.append(part)
    family = pd.concat(usable, ignore_index=True, sort=False) if usable else pd.DataFrame()
    if not family.empty:
        pvalues = pd.to_numeric(family["pvalue_for_fdr"], errors="coerce")
        order = pvalues.sort_values().index
        adjusted = pd.Series(np.nan, index=family.index, dtype=float)
        running = 1.0
        m = int(pvalues.notna().sum())
        for rank, index in enumerate(reversed(order), start=1):
            value = float(pvalues.loc[index]) * m / (m - rank + 1)
            running = min(running, value)
            adjusted.loc[index] = running
        family["qvalue_bh_full_structural_family"] = adjusted
        family["fdr_significant_0_05"] = (adjusted < 0.05).astype(int)
    family.to_csv(out_dir / "full_structural_multiplicity_family.csv", index=False)
    return family


def make_plots(profile: pd.DataFrame, coefficients: pd.DataFrame, out_dir: Path) -> None:
    profile_plot = profile.groupby("structural_regime")[STRUCTURAL_VARS].mean()
    standardized = (profile_plot - profile[STRUCTURAL_VARS].mean()) / profile[STRUCTURAL_VARS].std(ddof=0)
    ax = standardized.T.plot(kind="bar", figsize=(12, 6))
    ax.set_ylabel("Standardized pre-shock structural mean")
    ax.set_xlabel("")
    ax.set_title("Outcome-independent structural regime profiles")
    ax.legend(title="Regime")
    plt.tight_layout()
    plt.savefig(out_dir / "figure_structural_regime_profiles.png", dpi=180)
    plt.close()

    if coefficients.empty:
        return
    plot = coefficients.copy()
    plot["label"] = plot["outcome_key"].astype(str) + " / " + plot["regime"].astype(str)
    plot = plot.sort_values(["outcome_key", "regime"])
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(10, max(5, len(plot) * 0.35)))
    ax.errorbar(plot["estimate"], y, xerr=1.96 * plot["se"], fmt="o", color="#1f4e79", capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["label"])
    ax.set_xlabel("Regime-specific ECI x exposure x post coefficient")
    ax.set_title("Structural-regime heterogeneity estimates")
    plt.tight_layout()
    plt.savefig(out_dir / "figure_structural_regime_coefficients.png", dpi=180)
    plt.close()


def write_protocol(out_dir: Path, commit: str) -> None:
    text = f"""# Structural-Regime Pre-Analysis Protocol

Fixed repository commit: `{commit}`.

This extension is frozen before outcome-regime models are estimated. The country-level structural profile uses only 2012-2017 information. ECI, COI, treatment exposure, resilience outcomes, GPR, and post-2017 classifications are excluded from the clustering feature matrix. Region is retained for composition diagnostics but is not one-hot encoded into the numeric clustering matrix. ECI remains the focal explanatory variable.

Primary outcomes:

- adverse deviation from the 2015-2017 forward-GVC linkage mean;
- log export recovery relative to the 2015-2017 export baseline;
- partner diversification excluding the United States and China.

Structural candidates:

`{", ".join(STRUCTURAL_VARS)}`.

The primary continuous exposure is the frozen pre-shock US-China trade-nexus measure (`exposure_pre`), standardized across the eligible third-country sample. The primary post indicator begins in 2018; 2019 onward is a sensitivity period. Model controls are country and year fixed effects, with structural-variable-by-year adjustment, region-by-year adjustment, country-specific linear trends, and continuous GPS overlap weights as separate sensitivity models. Contemporaneous post-shock controls are not used.

Missing structural cells are median-imputed only for the unsupervised clustering step, with imputation counts retained in the assignments audit. The inferential family includes continuous structural moderator coefficients, progressive-adjustment focal coefficients, pooled regime-specific coefficients and contrasts, event-study pretrend joint tests, and placebo tests. Descriptive exposure-balance, correlation, VIF, and regime-composition diagnostics are not treated as confirmatory hypothesis tests. All model tests use one Benjamini-Hochberg correction. Wild-cluster bootstrap inference uses 999 Rademacher replications.

Decision rule: a structural regime enters the main theory only if the pooled regime-difference or moderator test survives the complete-family correction, pretrends and placebo tests are acceptable, signs are stable across adjustment models, no single country or region drives the estimate, the regime has adequate exposed-country counts and power, and the pattern appears in a corrected outcome. Otherwise it remains exploratory or appendix-only.
"""
    (out_dir / "structural_design_protocol.md").write_text(text, encoding="utf-8")


def write_summary(
    out_dir: Path,
    profile: pd.DataFrame,
    selected_k: int,
    family: pd.DataFrame,
    coefficients: pd.DataFrame,
    differences: pd.DataFrame,
    selection: pd.DataFrame,
) -> None:
    significant = int((family.get("fdr_significant_0_05", pd.Series(dtype=int)) == 1).sum()) if not family.empty else 0
    regime_diff_sig = int((differences.get("qvalue_bh_full_structural_family", pd.Series(dtype=float)) < 0.05).sum()) if not differences.empty else 0
    selected_score = selection.loc[selection["k"].eq(selected_k), "selection_score"].iloc[0]
    lines = [
        "# Structural-Regime Analysis Completion",
        "",
        "This package implements the attached ten-step structural-regime checklist as a separate extension of the completed tariff-weighted design.",
        "",
        "## Frozen Design",
        f"- Structural profile countries: {profile['country_iso3_code'].nunique()}; source period: 2012-2017.",
        f"- Candidate structural variables: {len(STRUCTURAL_VARS)}; clustering excludes ECI, COI, treatment, outcomes, GPR, and post-shock classifications.",
        f"- Selected structural solution: k={selected_k}; outcome-independent selection score={selected_score:.3f}.",
        "- Primary post period: 2018 onward; 2019 onward is sensitivity.",
        "",
        "## Diagnostics and Inference",
        f"- Full structural model family: {len(family)} tests.",
        f"- BH-significant tests at q<.05: {significant}.",
        f"- BH-significant pooled regime differences: {regime_diff_sig}.",
        "- Continuous moderator tests, pooled regime differences, placebo tests, and event-study pretrend joints are reported separately and linked through the full family.",
        "",
        "## Interpretation",
        "The structural-regime argument is not promoted automatically from a positive subgroup coefficient. The decision rule requires pooled differences, multiplicity-adjusted evidence, acceptable identification diagnostics, adjustment stability, adequate power, and influence/geographic checks.",
        "",
        "## Outputs",
        "- `structural_profile_2012_2017.csv` and `structural_profile_audit.csv`",
        "- `exposure_balance.csv`, `exposure_smd.csv`, `exposure_correlations.csv`, `pre_outcome_process_tests.csv`, and `structural_vif.csv`",
        "- `continuous_structural_moderator_tests.csv` and `progressive_adjustment_models.csv`",
        "- `structural_regime_selection.csv`, `structural_regime_profiles.csv`, and `structural_regime_assignments.csv`",
        "- `structural_regime_specific_coefficients.csv`, `structural_regime_difference_tests.csv`, and `structural_regime_power_mde.csv`",
        "- `structural_regime_event_study_coefficients.csv`, `structural_regime_event_study_pretrend_tests.csv`, `structural_regime_placebo_tests.csv`, and `structural_regime_leave_one_country_out.csv`",
        "- `omitted_confounding_sensitivity.csv` and `full_structural_multiplicity_family.csv`",
    ]
    (out_dir / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen structural-regime analysis extension.")
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_DEFAULT)
    parser.add_argument("--stability-reps", type=int, default=STABILITY_DEFAULT)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_dir = base_dir / "reports" / "structural_regime_completion"
    ensure_dir(out_dir)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base_dir, text=True).strip()
    write_protocol(out_dir, commit)

    panel = pd.read_csv(base_dir / "reports" / "final_design_completion" / "panel_with_completed_design_constructs.csv")
    wdi, wdi_path, downloaded = load_wdi_structural_source(base_dir)
    profile, profile_audit, outcome_audit = build_structural_profile(panel, wdi)
    profile.to_csv(out_dir / "structural_profile_2012_2017.csv", index=False)
    profile_audit.to_csv(out_dir / "structural_profile_audit.csv", index=False)
    outcome_audit.to_csv(out_dir / "pre_outcome_coverage_audit.csv", index=False)

    diagnostic_tables = run_exposure_diagnostics(profile, out_dir)
    model_panel = panel.merge(
        profile[["country_iso3_code", "wb_region", "z_eci_pre", "z_coi_pre", "z_exposure_pre"] + [f"z_{v}" for v in STRUCTURAL_VARS]],
        on="country_iso3_code",
        how="left",
        suffixes=("", "_profile"),
    )
    model_panel["post_2018"] = model_panel["year"].ge(PRIMARY_POST_START).astype(float)
    model_panel["post_2019"] = model_panel["year"].ge(SENSITIVITY_POST_START).astype(float)

    continuous = run_continuous_moderators(model_panel, out_dir, args.bootstrap_reps, args.seed)
    adjustments = run_adjustment_models(model_panel, profile, out_dir)
    profile, selection, assignments, selected_k, regimes = choose_structural_regimes(profile, out_dir, args.seed, args.stability_reps)
    model_panel = model_panel.drop(columns=["structural_regime"], errors="ignore").merge(
        profile[["country_iso3_code", "structural_regime"]], on="country_iso3_code", how="left"
    )
    model_panel.to_csv(out_dir / "panel_with_structural_regimes.csv", index=False)
    regime_composition = run_regime_composition(profile, model_panel, regimes, out_dir)
    coefficients, differences, fits = run_regime_models(model_panel, regimes, out_dir, args.bootstrap_reps, args.seed + 50000)
    power = run_regime_power(profile, coefficients, out_dir)
    event, pretrend, placebo = run_event_study_and_placebo(model_panel, regimes, out_dir, args.bootstrap_reps, args.seed + 60000)
    loo = run_leave_one_out(model_panel, regimes, out_dir)
    omitted = run_omitted_confounding(adjustments, out_dir)
    family = apply_family([continuous, adjustments, coefficients, differences, pretrend, placebo], out_dir)

    # Reattach the authoritative full-family q-values to model-specific tables.
    for filename in [
        "continuous_structural_moderator_tests.csv",
        "progressive_adjustment_models.csv",
        "structural_regime_specific_coefficients.csv",
        "structural_regime_difference_tests.csv",
        "structural_regime_event_study_pretrend_tests.csv",
        "structural_regime_placebo_tests.csv",
    ]:
        path = out_dir / filename
        table = pd.read_csv(path)
        if "test_id" in table and not family.empty:
            q = family[["test_id", "qvalue_bh_full_structural_family", "fdr_significant_0_05"]].drop_duplicates("test_id")
            table = table.drop(columns=["qvalue_bh_full_structural_family", "fdr_significant_0_05"], errors="ignore").merge(q, on="test_id", how="left")
            table.to_csv(path, index=False)

    make_plots(profile, coefficients, out_dir)
    audit = pd.DataFrame(
        [
            {"item": "repository_commit", "value": commit},
            {"item": "wdi_source_path", "value": str(wdi_path)},
            {"item": "wdi_source_sha256", "value": sha256(wdi_path)},
            {"item": "wdi_downloaded_this_run", "value": int(downloaded)},
            {"item": "baseline_start", "value": BASELINE_START},
            {"item": "baseline_end", "value": BASELINE_END},
            {"item": "primary_post_start", "value": PRIMARY_POST_START},
            {"item": "sensitivity_post_start", "value": SENSITIVITY_POST_START},
            {"item": "selected_k", "value": selected_k},
            {"item": "selected_regimes", "value": ",".join(regimes)},
            {"item": "bootstrap_reps", "value": args.bootstrap_reps},
            {"item": "stability_reps", "value": args.stability_reps},
            {"item": "structural_profile_countries", "value": profile["country_iso3_code"].nunique()},
            {"item": "full_family_tests", "value": len(family)},
            {"item": "full_family_fdr_significant", "value": int(family["fdr_significant_0_05"].sum()) if not family.empty else 0},
        ]
    )
    audit.to_csv(out_dir / "structural_analysis_metadata.csv", index=False)
    write_summary(out_dir, profile, selected_k, family, coefficients, differences, selection)
    print(f"Wrote structural-regime analysis to {out_dir}")
    print(f"Selected k={selected_k}; family tests={len(family)}; q<.05={int(family['fdr_significant_0_05'].sum()) if not family.empty else 0}")


if __name__ == "__main__":
    main()

