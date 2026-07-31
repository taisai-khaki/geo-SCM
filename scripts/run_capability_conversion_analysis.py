from __future__ import annotations

"""Run the theory-refining capability-conversion (H6) analysis package.

This script deliberately keeps the legacy analysis outputs intact. It creates a
separate, reproducible redesign that freezes capability measures before the
tariff shock, uses continuous pre-shock nexus exposure, and treats H6 as
exploratory/theory-refining rather than confirmatory.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats


BASELINE_START = 2015
BASELINE_END = 2017
POST_START = 2019
H6_TERMS = [
    "post_x_exposure",
    "post_x_eci_below",
    "post_x_eci_above",
    "h6_low_slope",
    "h6_high_slope",
]


@dataclass
class FEFit:
    term_names: list[str]
    beta: np.ndarray
    covariance: np.ndarray
    standard_errors: np.ndarray
    pvalues_cluster: np.ndarray
    x_resid: np.ndarray
    y_resid: np.ndarray
    y_hat: np.ndarray
    residuals: np.ndarray
    inv_xx: np.ndarray
    cluster_codes: np.ndarray
    n_clusters: int
    n_countries: int
    n_years: int
    n_obs: int
    rss: float
    r2_within: float
    df_clusters: int
    k_full: int
    sample_index: np.ndarray

    def term_index(self, term: str) -> int:
        return self.term_names.index(term)

    def term_row(self, term: str) -> dict[str, float]:
        idx = self.term_index(term)
        return {
            "coef": float(self.beta[idx]),
            "se": float(self.standard_errors[idx]),
            "p_cluster_two_sided": float(self.pvalues_cluster[idx]),
        }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    sd = values.std(skipna=True, ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (values - values.mean(skipna=True)) / sd


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    lo = values.quantile(lower)
    hi = values.quantile(upper)
    return values.clip(lower=lo, upper=hi)


def bh_fdr_adjust(pvalues: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = pvalues.notna()
    if valid.sum() == 0:
        return out
    ordered = pvalues.loc[valid].sort_values()
    m = len(ordered)
    ranks = np.arange(1, m + 1, dtype=float)
    adjusted = ordered.to_numpy(dtype=float) * m / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out.loc[ordered.index] = np.minimum(adjusted, 1.0)
    return out


def _weighted_group_demean(
    values: np.ndarray,
    codes: np.ndarray,
    n_groups: int,
    weights: np.ndarray | None,
) -> np.ndarray:
    totals = np.zeros((n_groups, values.shape[1]), dtype=float)
    if weights is None:
        np.add.at(totals, codes, values)
        denom = np.bincount(codes, minlength=n_groups).astype(float)
    else:
        np.add.at(totals, codes, values * weights[:, None])
        denom = np.bincount(codes, weights=weights, minlength=n_groups).astype(float)
    means = totals / np.maximum(denom[:, None], 1e-12)
    return values - means[codes]


def two_way_residualize(
    values: np.ndarray,
    entity_codes: np.ndarray,
    time_codes: np.ndarray,
    weights: np.ndarray | None = None,
    max_iter: int = 200,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Alternating projections remove country and year fixed effects."""

    result = np.asarray(values, dtype=float)
    if result.ndim == 1:
        result = result[:, None]
    result = result.copy()
    n_entity = int(entity_codes.max()) + 1
    n_time = int(time_codes.max()) + 1

    for _ in range(max_iter):
        old = result.copy()
        result = _weighted_group_demean(result, entity_codes, n_entity, weights)
        result = _weighted_group_demean(result, time_codes, n_time, weights)
        if np.max(np.abs(result - old)) < tolerance:
            break
    return result


def cluster_covariance(
    x: np.ndarray,
    residuals: np.ndarray,
    cluster_codes: np.ndarray,
    n_clusters: int,
    k_full: int,
    inv_xx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if inv_xx is None:
        inv_xx = np.linalg.pinv(x.T @ x)
    scores = np.zeros((n_clusters, x.shape[1]), dtype=float)
    np.add.at(scores, cluster_codes, x * residuals[:, None])
    meat = scores.T @ scores
    n_obs = x.shape[0]
    denom = max(n_obs - k_full, 1)
    correction = (n_clusters / max(n_clusters - 1, 1)) * ((n_obs - 1) / denom)
    covariance = correction * (inv_xx @ meat @ inv_xx)
    covariance = (covariance + covariance.T) / 2.0
    return covariance, inv_xx


def fit_twfe(
    df: pd.DataFrame,
    outcome_col: str,
    term_names: list[str],
    weight_col: str | None = None,
    entity_col: str = "country_iso3_code",
    time_col: str = "year",
    cluster_col: str = "country_iso3_code",
) -> FEFit:
    required = list(dict.fromkeys([outcome_col, entity_col, time_col, cluster_col] + term_names))
    if weight_col is not None:
        required.append(weight_col)
    work = df[required].copy().replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        raise ValueError(f"No observations available for {outcome_col}.")

    entity_codes, entity_levels = pd.factorize(work[entity_col].astype(str), sort=True)
    time_codes, time_levels = pd.factorize(work[time_col], sort=True)
    cluster_codes, cluster_levels = pd.factorize(work[cluster_col].astype(str), sort=True)
    y = work[outcome_col].to_numpy(dtype=float)
    x = work[term_names].to_numpy(dtype=float)

    weights: np.ndarray | None = None
    if weight_col is not None:
        weights = work[weight_col].to_numpy(dtype=float)
        if np.any(weights <= 0):
            raise ValueError(f"Non-positive weights in {weight_col}.")

    y_resid = two_way_residualize(y, entity_codes, time_codes, weights).ravel()
    x_resid = two_way_residualize(x, entity_codes, time_codes, weights)
    if weights is not None:
        sqrt_w = np.sqrt(weights)
        y_resid = y_resid * sqrt_w
        x_resid = x_resid * sqrt_w[:, None]

    keep = np.nanstd(x_resid, axis=0) > 1e-11
    if not keep.all():
        dropped = [name for name, keep_flag in zip(term_names, keep) if not keep_flag]
        raise ValueError(f"Collinear after fixed-effect residualization: {dropped}")

    beta = np.linalg.pinv(x_resid.T @ x_resid) @ (x_resid.T @ y_resid)
    fitted = x_resid @ beta
    residuals = y_resid - fitted
    n_obs = len(work)
    n_countries = len(entity_levels)
    n_years = len(time_levels)
    n_clusters = len(cluster_levels)
    k_full = len(term_names) + n_countries + n_years - 1
    covariance, inv_xx = cluster_covariance(
        x_resid, residuals, cluster_codes, n_clusters=n_clusters, k_full=k_full
    )
    ses = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = beta / ses
    df_clusters = max(n_clusters - 1, 1)
    pvals = 2.0 * stats.t.sf(np.abs(tvals), df=df_clusters)
    ss_total = float(np.sum((y_resid - y_resid.mean()) ** 2))
    rss = float(np.sum(residuals**2))
    r2 = np.nan if ss_total <= 0 else 1.0 - rss / ss_total

    return FEFit(
        term_names=term_names,
        beta=beta,
        covariance=covariance,
        standard_errors=ses,
        pvalues_cluster=pvals,
        x_resid=x_resid,
        y_resid=y_resid,
        y_hat=fitted,
        residuals=residuals,
        inv_xx=inv_xx,
        cluster_codes=cluster_codes,
        n_clusters=n_clusters,
        n_countries=n_countries,
        n_years=n_years,
        n_obs=n_obs,
        rss=rss,
        r2_within=float(r2),
        df_clusters=df_clusters,
        k_full=k_full,
        sample_index=work.index.to_numpy(),
    )


def contrast_statistics(
    fit: FEFit,
    contrast: np.ndarray,
    alternative: str = "two-sided",
) -> dict[str, float]:
    estimate = float(contrast @ fit.beta)
    variance = float(contrast @ fit.covariance @ contrast)
    se = float(np.sqrt(max(variance, 0.0)))
    tvalue = estimate / se if se > 0 else np.nan
    if alternative == "less":
        pvalue = float(stats.t.cdf(tvalue, df=fit.df_clusters))
    elif alternative == "greater":
        pvalue = float(stats.t.sf(tvalue, df=fit.df_clusters))
    else:
        pvalue = float(2.0 * stats.t.sf(abs(tvalue), df=fit.df_clusters))
    return {
        "estimate": estimate,
        "se": se,
        "tvalue": float(tvalue),
        "p_cluster": pvalue,
        "ci_low_95": estimate - stats.t.ppf(0.975, fit.df_clusters) * se,
        "ci_high_95": estimate + stats.t.ppf(0.975, fit.df_clusters) * se,
    }


def wild_cluster_bootstrap_contrast(
    fit: FEFit,
    contrast: np.ndarray,
    reps: int,
    seed: int,
    alternative: str,
) -> dict[str, float | int]:
    """Wild-cluster bootstrap-t with a restriction imposed under the null."""

    observed = contrast_statistics(fit, contrast, alternative=alternative)
    if not np.isfinite(observed["tvalue"]):
        return {
            "p_wild_bootstrap": np.nan,
            "bootstrap_reps_requested": int(reps),
            "bootstrap_reps_success": 0,
        }

    restriction_variance = float(contrast @ fit.inv_xx @ contrast)
    if restriction_variance <= 0:
        return {
            "p_wild_bootstrap": np.nan,
            "bootstrap_reps_requested": int(reps),
            "bootstrap_reps_success": 0,
        }

    restricted_beta = fit.beta - (
        (fit.inv_xx @ contrast) * (float(contrast @ fit.beta) / restriction_variance)
    )
    yhat_null = fit.x_resid @ restricted_beta
    residual_null = fit.y_resid - yhat_null
    rng = np.random.default_rng(seed)
    tstars: list[float] = []
    for _ in range(reps):
        signs = rng.choice(np.array([-1.0, 1.0]), size=fit.n_clusters)
        ystar = yhat_null + residual_null * signs[fit.cluster_codes]
        beta_star = fit.inv_xx @ (fit.x_resid.T @ ystar)
        residual_star = ystar - fit.x_resid @ beta_star
        cov_star, _ = cluster_covariance(
            fit.x_resid,
            residual_star,
            fit.cluster_codes,
            n_clusters=fit.n_clusters,
            k_full=fit.k_full,
            inv_xx=fit.inv_xx,
        )
        se_star = float(np.sqrt(max(float(contrast @ cov_star @ contrast), 0.0)))
        if se_star > 0:
            tstars.append(float((contrast @ beta_star) / se_star))

    tarr = np.asarray(tstars, dtype=float)
    if len(tarr) == 0:
        p_boot = np.nan
    elif alternative == "less":
        p_boot = float((1 + np.sum(tarr <= observed["tvalue"])) / (len(tarr) + 1))
    elif alternative == "greater":
        p_boot = float((1 + np.sum(tarr >= observed["tvalue"])) / (len(tarr) + 1))
    else:
        p_boot = float(
            (1 + np.sum(np.abs(tarr) >= abs(observed["tvalue"]))) / (len(tarr) + 1)
        )
    return {
        "p_wild_bootstrap": p_boot,
        "bootstrap_reps_requested": int(reps),
        "bootstrap_reps_success": int(len(tarr)),
    }


def make_pretrend_adjusted_recovery(panel: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=panel.index, dtype=float)
    for _, group in panel.groupby("country_iso3_code"):
        usable = group[(group["year"] <= BASELINE_END) & (group["export_value"] > 0)].copy()
        if usable["year"].nunique() < 4:
            continue
        log_exports = np.log(usable["export_value"].to_numpy(dtype=float))
        slope, intercept = np.polyfit(usable["year"].to_numpy(dtype=float), log_exports, 1)
        all_usable = group[group["export_value"] > 0]
        expected = intercept + slope * all_usable["year"].to_numpy(dtype=float)
        result.loc[all_usable.index] = np.log(
            all_usable["export_value"].to_numpy(dtype=float)
        ) - expected
    return result


def build_channel_exposures(bilateral: pd.DataFrame) -> pd.DataFrame:
    work = bilateral.copy()
    work = work[
        work["year"].between(BASELINE_START, BASELINE_END)
        & (work["country_iso3_code"] != work["partner_iso3_code"])
    ].copy()
    for col in ["export_value", "import_value"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    annual = (
        work.groupby(["country_iso3_code", "year"], as_index=False)
        .agg(total_exports=("export_value", "sum"), total_imports=("import_value", "sum"))
    )
    us_exports = (
        work[work["partner_iso3_code"] == "USA"]
        .groupby(["country_iso3_code", "year"], as_index=False)["export_value"]
        .sum()
        .rename(columns={"export_value": "exports_to_usa"})
    )
    china_exports = (
        work[work["partner_iso3_code"] == "CHN"]
        .groupby(["country_iso3_code", "year"], as_index=False)["export_value"]
        .sum()
        .rename(columns={"export_value": "exports_to_china"})
    )
    china_imports = (
        work[work["partner_iso3_code"] == "CHN"]
        .groupby(["country_iso3_code", "year"], as_index=False)["import_value"]
        .sum()
        .rename(columns={"import_value": "imports_from_china"})
    )
    result = annual.merge(us_exports, on=["country_iso3_code", "year"], how="left")
    result = result.merge(china_exports, on=["country_iso3_code", "year"], how="left")
    result = result.merge(china_imports, on=["country_iso3_code", "year"], how="left")
    result[["exports_to_usa", "exports_to_china", "imports_from_china"]] = result[
        ["exports_to_usa", "exports_to_china", "imports_from_china"]
    ].fillna(0.0)
    result["us_export_market_dependence_pre_year"] = np.where(
        result["total_exports"] > 0,
        result["exports_to_usa"] / result["total_exports"],
        np.nan,
    )
    result["china_export_market_dependence_pre_year"] = np.where(
        result["total_exports"] > 0,
        result["exports_to_china"] / result["total_exports"],
        np.nan,
    )
    result["china_import_dependence_pre_year"] = np.where(
        result["total_imports"] > 0,
        result["imports_from_china"] / result["total_imports"],
        np.nan,
    )
    return (
        result.groupby("country_iso3_code", as_index=False)[
            [
                "us_export_market_dependence_pre_year",
                "china_export_market_dependence_pre_year",
                "china_import_dependence_pre_year",
            ]
        ]
        .mean()
        .rename(
            columns={
                "us_export_market_dependence_pre_year": "us_export_market_dependence_pre",
                "china_export_market_dependence_pre_year": "china_export_market_dependence_pre",
                "china_import_dependence_pre_year": "china_import_dependence_pre",
            }
        )
    )


def build_partner_outcomes(bilateral: pd.DataFrame) -> pd.DataFrame:
    """Create non-US/China diversification and direct destination-entry measures."""

    work = bilateral.copy()
    work = work[
        (work["country_iso3_code"] != work["partner_iso3_code"])
        & ~work["partner_iso3_code"].isin(["USA", "CHN"])
    ].copy()
    work["export_value"] = pd.to_numeric(work["export_value"], errors="coerce").fillna(0.0)
    work = work[work["export_value"] > 0].copy()

    totals = (
        work.groupby(["country_iso3_code", "year"], as_index=False)["export_value"]
        .sum()
        .rename(columns={"export_value": "exports_excl_us_china"})
    )
    work = work.merge(totals, on=["country_iso3_code", "year"], how="left")
    work["share"] = work["export_value"] / work["exports_excl_us_china"]
    concentration = (
        work.assign(
            share_sq=work["share"] ** 2,
            share_log_share=work["share"] * np.log(work["share"]),
        )
        .groupby(["country_iso3_code", "year"], as_index=False)
        .agg(
            hhi_excl_us_china=("share_sq", "sum"),
            destination_entropy_excl_us_china=("share_log_share", lambda x: -x.sum()),
            n_non_uschina_destinations=("partner_iso3_code", "nunique"),
        )
    )
    concentration["partner_diversification_excl_us_china"] = (
        1.0 - concentration["hhi_excl_us_china"]
    )
    concentration["effective_destinations_excl_us_china"] = np.where(
        concentration["hhi_excl_us_china"] > 0,
        1.0 / concentration["hhi_excl_us_china"],
        np.nan,
    )

    entry_rows: list[dict[str, Any]] = []
    for country, group in work.groupby("country_iso3_code"):
        by_year: dict[int, pd.DataFrame] = {
            int(year): year_group.copy() for year, year_group in group.groupby("year")
        }
        for year, current in by_year.items():
            prior_partners: set[str] = set()
            for prior_year in [year - 1, year - 2, year - 3]:
                if prior_year in by_year:
                    prior_partners.update(by_year[prior_year]["partner_iso3_code"].astype(str))
            current_partners = set(current["partner_iso3_code"].astype(str))
            new_partners = current_partners - prior_partners
            new_value = float(
                current[current["partner_iso3_code"].astype(str).isin(new_partners)][
                    "export_value"
                ].sum()
            )
            total_value = float(current["export_value"].sum())
            next_partners = (
                set(by_year[year + 1]["partner_iso3_code"].astype(str))
                if year + 1 in by_year
                else set()
            )
            persistent = new_partners & next_partners
            entry_rows.append(
                {
                    "country_iso3_code": country,
                    "year": year,
                    "new_non_uschina_destination_count": len(new_partners),
                    "new_non_uschina_destination_export_share": (
                        new_value / total_value if total_value > 0 else np.nan
                    ),
                    "persistent_new_non_uschina_destination_count": len(persistent)
                    if year + 1 in by_year
                    else np.nan,
                }
            )
    entries = pd.DataFrame(entry_rows)
    return concentration.merge(entries, on=["country_iso3_code", "year"], how="outer")


def freeze_pre_shock_constructs(
    panel: pd.DataFrame,
    bilateral: pd.DataFrame,
    income_reference: pd.DataFrame,
    regions: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pre = panel[panel["year"].between(BASELINE_START, BASELINE_END)].copy()
    pre = (
        pre.groupby("country_iso3_code", as_index=False)
        .agg(
            eci_pre_raw=("eci", "mean"),
            coi_pre_raw=("coi", "mean"),
            baseline_log_gdp_pc=("log_gdp_pc", "mean"),
            baseline_trade_open=("wdi_trade_openness_pct_gdp", "mean"),
            baseline_wgi=("wgi_institutional_quality_composite", "mean"),
            baseline_rents=("wdi_natural_resource_rents_pct_gdp", "mean"),
            baseline_export_value=("baseline_export_2015_2017", "mean"),
            nexus_exposure_pre=("us_china_trade_intensity_pre", "mean"),
        )
    )
    for col in [
        "eci_pre_raw",
        "coi_pre_raw",
        "baseline_log_gdp_pc",
        "baseline_trade_open",
        "baseline_wgi",
        "baseline_rents",
        "nexus_exposure_pre",
    ]:
        pre[f"z_{col}"] = zscore(pre[col])

    valid_exposure = pre["nexus_exposure_pre"].dropna()
    pre["exposed_top_tercile_pre"] = (
        pre["nexus_exposure_pre"] >= valid_exposure.quantile(2.0 / 3.0)
    ).astype(int)
    pre["exposed_top_quartile_pre"] = (
        pre["nexus_exposure_pre"] >= valid_exposure.quantile(0.75)
    ).astype(int)
    pre["exposure_tercile_threshold"] = valid_exposure.quantile(2.0 / 3.0)
    pre["exposure_quartile_threshold"] = valid_exposure.quantile(0.75)

    channels = build_channel_exposures(bilateral)
    pre = pre.merge(channels, on="country_iso3_code", how="left")
    income_cols = [
        c for c in ["country_iso3_code", "wb_income_id", "wb_income_name"] if c in income_reference
    ]
    if income_cols:
        pre = pre.merge(
            income_reference[income_cols].drop_duplicates(),
            on="country_iso3_code",
            how="left",
        )
    if regions is not None and {"country_iso3_code", "wb_region"}.issubset(regions.columns):
        pre = pre.merge(
            regions[["country_iso3_code", "wb_region"]].drop_duplicates(),
            on="country_iso3_code",
            how="left",
        )
    if "wb_region" not in pre:
        pre["wb_region"] = np.nan

    result = panel.merge(pre, on="country_iso3_code", how="left")
    result["post_tariff"] = (result["year"] >= POST_START).astype(int)
    result["gvc_signed_change"] = result["delta_tiva_fexgr_dva_share"]
    positive_exports = (result["export_value"] > 0) & (result["baseline_export_2015_2017"] > 0)
    result["log_export_recovery"] = np.where(
        positive_exports,
        np.log(result["export_value"]) - np.log(result["baseline_export_2015_2017"]),
        np.nan,
    )
    result["export_recovery_winsor_1_99"] = winsorize(result["export_recovery_index"])
    result["pretrend_adjusted_export_recovery"] = make_pretrend_adjusted_recovery(result)

    q10 = pre["baseline_export_value"].quantile(0.10)
    result["not_bottom_decile_export_size"] = (
        result["baseline_export_value"] > q10
    ).astype(int)
    sqrt_export = np.sqrt(result["baseline_export_value"].clip(lower=1.0))
    scaled_weight = sqrt_export / sqrt_export.median(skipna=True)
    result["export_size_weight"] = scaled_weight.clip(
        lower=scaled_weight.quantile(0.01),
        upper=scaled_weight.quantile(0.99),
    )

    partner_outcomes = build_partner_outcomes(bilateral)
    result = result.merge(partner_outcomes, on=["country_iso3_code", "year"], how="left")
    return result, pre


def prepare_piecewise_data(
    df: pd.DataFrame,
    threshold: float,
    exposure_col: str,
) -> pd.DataFrame:
    out = df.copy()
    eci_centered = out["eci_pre_raw"] - threshold
    out["eci_below_threshold"] = np.minimum(eci_centered, 0.0)
    out["eci_above_threshold"] = np.maximum(eci_centered, 0.0)
    post = out["post_tariff"]
    exposure = out[exposure_col]
    out["post_x_exposure"] = post * exposure
    out["post_x_eci_below"] = post * out["eci_below_threshold"]
    out["post_x_eci_above"] = post * out["eci_above_threshold"]
    out["h6_low_slope"] = post * exposure * out["eci_below_threshold"]
    out["h6_high_slope"] = post * exposure * out["eci_above_threshold"]
    return out


def select_h6_threshold(
    df: pd.DataFrame,
    outcome_col: str,
    exposure_col: str,
    quantiles: Iterable[float],
) -> tuple[float, pd.DataFrame, dict[float, FEFit]]:
    required = [outcome_col, "eci_pre_raw", exposure_col, "country_iso3_code", "year"]
    sample = df.dropna(subset=required).copy()
    country_eci = sample[["country_iso3_code", "eci_pre_raw"]].drop_duplicates()
    thresholds = np.unique(
        np.round(country_eci["eci_pre_raw"].quantile(list(quantiles)).to_numpy(dtype=float), 8)
    )
    fits: dict[float, FEFit] = {}
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        low_countries = int((country_eci["eci_pre_raw"] < threshold).sum())
        high_countries = int((country_eci["eci_pre_raw"] > threshold).sum())
        if min(low_countries, high_countries) < 20:
            continue
        model_df = prepare_piecewise_data(sample, float(threshold), exposure_col)
        fit = fit_twfe(model_df, outcome_col, H6_TERMS)
        fits[float(threshold)] = fit
        rows.append(
            {
                "threshold_eci_pre": float(threshold),
                "rss": fit.rss,
                "r2_within": fit.r2_within,
                "n_obs": fit.n_obs,
                "n_countries": fit.n_countries,
                "countries_below_threshold": low_countries,
                "countries_above_threshold": high_countries,
            }
        )
    table = pd.DataFrame(rows).sort_values("threshold_eci_pre").reset_index(drop=True)
    if table.empty:
        raise ValueError("No feasible threshold candidates.")
    best_threshold = float(table.loc[table["rss"].idxmin(), "threshold_eci_pre"])
    return best_threshold, table, fits


def wild_bootstrap_threshold_distribution(
    fits: dict[float, FEFit],
    best_threshold: float,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    best_fit = fits[best_threshold]
    thresholds = sorted(fits)
    for threshold in thresholds:
        other = fits[threshold]
        if not np.array_equal(other.sample_index, best_fit.sample_index):
            raise ValueError("Threshold candidates do not share an identical estimation sample.")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for rep in range(1, reps + 1):
        signs = rng.choice(np.array([-1.0, 1.0]), size=best_fit.n_clusters)
        ystar = best_fit.y_hat + best_fit.residuals * signs[best_fit.cluster_codes]
        rss_values: list[tuple[float, float]] = []
        for threshold in thresholds:
            fit = fits[threshold]
            beta = fit.inv_xx @ (fit.x_resid.T @ ystar)
            resid = ystar - fit.x_resid @ beta
            rss_values.append((threshold, float(np.sum(resid**2))))
        selected = min(rss_values, key=lambda item: item[1])
        rows.append(
            {
                "replication": rep,
                "threshold_eci_pre": float(selected[0]),
                "rss_selected": float(selected[1]),
            }
        )
    return pd.DataFrame(rows)


def h6_test_rows(
    fit: FEFit,
    spec: str,
    outcome_label: str,
    exposure_label: str,
    threshold: float,
    reps: int,
    seed: int,
) -> list[dict[str, Any]]:
    low = np.zeros(len(fit.term_names), dtype=float)
    high = np.zeros(len(fit.term_names), dtype=float)
    low[fit.term_index("h6_low_slope")] = 1.0
    high[fit.term_index("h6_high_slope")] = 1.0
    contrasts = [
        ("H6a_beta_low_lt_0", low, "less", "negative"),
        ("H6b_beta_high_minus_low_gt_0", high - low, "greater", "positive"),
    ]
    rows: list[dict[str, Any]] = []
    for offset, (test_name, contrast, alternative, expected_direction) in enumerate(contrasts):
        stats_row = contrast_statistics(fit, contrast, alternative=alternative)
        boot = wild_cluster_bootstrap_contrast(
            fit,
            contrast,
            reps=reps,
            seed=seed + offset,
            alternative=alternative,
        )
        rows.append(
            {
                "spec": spec,
                "outcome": outcome_label,
                "exposure": exposure_label,
                "threshold_eci_pre": float(threshold),
                "test": test_name,
                "alternative": alternative,
                "expected_direction": expected_direction,
                **stats_row,
                **boot,
                "n_obs": fit.n_obs,
                "n_countries": fit.n_countries,
                "n_years": fit.n_years,
            }
        )
    return rows


def add_pre_control_year_terms(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    controls = [
        "z_baseline_log_gdp_pc",
        "z_baseline_trade_open",
        "z_baseline_wgi",
        "z_baseline_rents",
    ]
    terms: list[str] = []
    years = sorted(int(y) for y in out["year"].dropna().unique())
    if not years:
        return out, terms
    base_year = years[0]
    for control in controls:
        if control not in out:
            continue
        for year in years:
            if year == base_year:
                continue
            name = f"{control}_x_year_{year}"
            out[name] = out[control] * (out["year"] == year).astype(int)
            terms.append(name)
    return out, terms


def run_h6_suite(
    panel: pd.DataFrame,
    bootstrap_reps: int,
    threshold_bootstrap_reps: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    float,
    FEFit,
]:
    quantiles = np.arange(0.20, 0.801, 0.05)
    primary_outcome = "log_export_recovery"
    primary_exposure = "z_nexus_exposure_pre"
    best_threshold, search_table, fits = select_h6_threshold(
        panel, primary_outcome, primary_exposure, quantiles
    )
    threshold_boot = wild_bootstrap_threshold_distribution(
        fits,
        best_threshold=best_threshold,
        reps=threshold_bootstrap_reps,
        seed=seed + 100,
    )
    threshold_summary = pd.DataFrame(
        [
            {
                "primary_outcome": primary_outcome,
                "primary_exposure": primary_exposure,
                "threshold_eci_pre": best_threshold,
                "threshold_ci_low_95_wild": float(
                    threshold_boot["threshold_eci_pre"].quantile(0.025)
                ),
                "threshold_ci_high_95_wild": float(
                    threshold_boot["threshold_eci_pre"].quantile(0.975)
                ),
                "threshold_bootstrap_reps": int(threshold_bootstrap_reps),
                "threshold_grid_quantiles": "0.20 to 0.80 by 0.05",
            }
        ]
    )
    primary_fit = fits[best_threshold]
    rows: list[dict[str, Any]] = h6_test_rows(
        primary_fit,
        spec="primary_log_continuous_exposure",
        outcome_label="Log export recovery",
        exposure_label="Continuous pre-shock US-China nexus exposure (z)",
        threshold=best_threshold,
        reps=bootstrap_reps,
        seed=seed + 200,
    )

    specs: list[dict[str, Any]] = [
        {
            "spec": "ratio_continuous_exposure",
            "outcome": "export_recovery_index",
            "outcome_label": "Ratio export recovery",
            "exposure": "z_nexus_exposure_pre",
            "exposure_label": "Continuous pre-shock US-China nexus exposure (z)",
        },
        {
            "spec": "winsorized_ratio_continuous_exposure",
            "outcome": "export_recovery_winsor_1_99",
            "outcome_label": "Winsorized ratio export recovery (1/99)",
            "exposure": "z_nexus_exposure_pre",
            "exposure_label": "Continuous pre-shock US-China nexus exposure (z)",
        },
        {
            "spec": "pretrend_adjusted_log_continuous_exposure",
            "outcome": "pretrend_adjusted_export_recovery",
            "outcome_label": "Pretrend-adjusted log export deviation",
            "exposure": "z_nexus_exposure_pre",
            "exposure_label": "Continuous pre-shock US-China nexus exposure (z)",
        },
        {
            "spec": "log_continuous_exposure_excl_bottom_export_decile",
            "outcome": "log_export_recovery",
            "outcome_label": "Log export recovery",
            "exposure": "z_nexus_exposure_pre",
            "exposure_label": "Continuous pre-shock US-China nexus exposure (z)",
            "subset": "not_bottom_decile_export_size",
        },
        {
            "spec": "log_continuous_exposure_export_size_weighted",
            "outcome": "log_export_recovery",
            "outcome_label": "Log export recovery",
            "exposure": "z_nexus_exposure_pre",
            "exposure_label": "Continuous pre-shock US-China nexus exposure (z)",
            "weight_col": "export_size_weight",
        },
        {
            "spec": "log_top_tercile_exposure",
            "outcome": "log_export_recovery",
            "outcome_label": "Log export recovery",
            "exposure": "exposed_top_tercile_pre",
            "exposure_label": "Top-tercile nexus exposure",
        },
        {
            "spec": "log_top_quartile_exposure",
            "outcome": "log_export_recovery",
            "outcome_label": "Log export recovery",
            "exposure": "exposed_top_quartile_pre",
            "exposure_label": "Top-quartile nexus exposure",
        },
    ]

    controls_df, control_terms = add_pre_control_year_terms(panel)
    specs.append(
        {
            "spec": "log_continuous_exposure_pre_shock_controls_x_year",
            "outcome": "log_export_recovery",
            "outcome_label": "Log export recovery",
            "exposure": "z_nexus_exposure_pre",
            "exposure_label": "Continuous pre-shock US-China nexus exposure (z)",
            "data": controls_df,
            "extra_terms": control_terms,
        }
    )

    model_rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(specs, start=1):
        spec_df = spec.get("data", panel)
        if "subset" in spec:
            spec_df = spec_df[spec_df[str(spec["subset"])] == 1].copy()
        model_df = prepare_piecewise_data(
            spec_df, threshold=best_threshold, exposure_col=str(spec["exposure"])
        )
        term_names = H6_TERMS + list(spec.get("extra_terms", []))
        fit = fit_twfe(
            model_df,
            outcome_col=str(spec["outcome"]),
            term_names=term_names,
            weight_col=spec.get("weight_col"),
        )
        rows.extend(
            h6_test_rows(
                fit,
                spec=str(spec["spec"]),
                outcome_label=str(spec["outcome_label"]),
                exposure_label=str(spec["exposure_label"]),
                threshold=best_threshold,
                reps=bootstrap_reps,
                seed=seed + 1000 + idx * 20,
            )
        )
        for term in H6_TERMS:
            model_rows.append(
                {
                    "spec": str(spec["spec"]),
                    "outcome": str(spec["outcome_label"]),
                    "term": term,
                    **fit.term_row(term),
                    "n_obs": fit.n_obs,
                    "n_countries": fit.n_countries,
                    "n_years": fit.n_years,
                    "r2_within": fit.r2_within,
                }
            )

    tests = pd.DataFrame(rows)
    tests["qvalue_bh_full_h6_family"] = bh_fdr_adjust(tests["p_wild_bootstrap"])
    tests["fdr_significant_0_05"] = (tests["qvalue_bh_full_h6_family"] < 0.05).astype(int)
    return (
        tests,
        pd.DataFrame(model_rows),
        threshold_summary,
        search_table,
        threshold_boot,
        best_threshold,
        primary_fit,
    )


def run_holdout_validation(
    panel: pd.DataFrame,
    strata_col: str,
    splits: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [
        "country_iso3_code",
        "log_export_recovery",
        "eci_pre_raw",
        "z_nexus_exposure_pre",
        strata_col,
    ]
    sample = panel.dropna(subset=required).copy()
    country_strata = sample[["country_iso3_code", strata_col]].drop_duplicates()
    country_strata[strata_col] = country_strata[strata_col].fillna("Unknown").astype(str)
    quantiles = np.arange(0.20, 0.801, 0.05)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for split in range(1, splits + 1):
        test_countries: list[str] = []
        for _, group in country_strata.groupby(strata_col):
            countries = group["country_iso3_code"].to_numpy(dtype=str)
            if len(countries) < 4:
                continue
            n_test = min(max(1, int(round(len(countries) * 0.30))), len(countries) - 1)
            test_countries.extend(rng.choice(countries, size=n_test, replace=False).tolist())
        test_set = set(test_countries)
        train = sample[~sample["country_iso3_code"].isin(test_set)].copy()
        test = sample[sample["country_iso3_code"].isin(test_set)].copy()
        try:
            selected, _, _ = select_h6_threshold(
                train,
                "log_export_recovery",
                "z_nexus_exposure_pre",
                quantiles,
            )
            fit_test = fit_twfe(
                prepare_piecewise_data(test, selected, "z_nexus_exposure_pre"),
                "log_export_recovery",
                H6_TERMS,
            )
            low = np.zeros(len(fit_test.term_names), dtype=float)
            high = np.zeros(len(fit_test.term_names), dtype=float)
            low[fit_test.term_index("h6_low_slope")] = 1.0
            high[fit_test.term_index("h6_high_slope")] = 1.0
            low_stats = contrast_statistics(fit_test, low, alternative="less")
            diff_stats = contrast_statistics(fit_test, high - low, alternative="greater")
            rows.append(
                {
                    "split": split,
                    "selected_threshold_train": selected,
                    "n_train_countries": int(train["country_iso3_code"].nunique()),
                    "n_test_countries": int(test["country_iso3_code"].nunique()),
                    "beta_low_test": low_stats["estimate"],
                    "p_low_test_cluster_one_sided": low_stats["p_cluster"],
                    "beta_high_minus_low_test": diff_stats["estimate"],
                    "p_difference_test_cluster_one_sided": diff_stats["p_cluster"],
                    "low_direction_matches": int(low_stats["estimate"] < 0),
                    "difference_direction_matches": int(diff_stats["estimate"] > 0),
                    "both_direction_match": int(
                        (low_stats["estimate"] < 0) and (diff_stats["estimate"] > 0)
                    ),
                }
            )
        except (ValueError, np.linalg.LinAlgError):
            rows.append(
                {
                    "split": split,
                    "selected_threshold_train": np.nan,
                    "n_train_countries": int(train["country_iso3_code"].nunique()),
                    "n_test_countries": int(test["country_iso3_code"].nunique()),
                    "beta_low_test": np.nan,
                    "p_low_test_cluster_one_sided": np.nan,
                    "beta_high_minus_low_test": np.nan,
                    "p_difference_test_cluster_one_sided": np.nan,
                    "low_direction_matches": np.nan,
                    "difference_direction_matches": np.nan,
                    "both_direction_match": np.nan,
                }
            )
    detail = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "stratification": strata_col,
                "splits_requested": int(splits),
                "splits_successful": int(detail["selected_threshold_train"].notna().sum()),
                "median_train_threshold": float(detail["selected_threshold_train"].median()),
                "threshold_iqr_low": float(detail["selected_threshold_train"].quantile(0.25)),
                "threshold_iqr_high": float(detail["selected_threshold_train"].quantile(0.75)),
                "share_low_negative": float(detail["low_direction_matches"].mean()),
                "share_difference_positive": float(detail["difference_direction_matches"].mean()),
                "share_both_directions": float(detail["both_direction_match"].mean()),
                "share_both_cluster_p_lt_0_05": float(
                    (
                        (detail["p_low_test_cluster_one_sided"] < 0.05)
                        & (detail["p_difference_test_cluster_one_sided"] < 0.05)
                    ).mean()
                ),
            }
        ]
    )
    return detail, summary


def add_linear_did_terms(df: pd.DataFrame, exposure_col: str) -> pd.DataFrame:
    out = df.copy()
    out["post_x_exposure"] = out["post_tariff"] * out[exposure_col]
    out["post_x_eci_pre"] = out["post_tariff"] * out["z_eci_pre_raw"]
    out["eci_pre_x_exposure_post"] = (
        out["post_tariff"] * out[exposure_col] * out["z_eci_pre_raw"]
    )
    return out


def equivalence_and_mde(fit: FEFit, term: str, bound: float = 0.10) -> dict[str, float | int]:
    contrast = np.zeros(len(fit.term_names), dtype=float)
    contrast[fit.term_index(term)] = 1.0
    result = contrast_statistics(fit, contrast, alternative="two-sided")
    estimate = float(result["estimate"])
    se = float(result["se"])
    if se <= 0:
        return {
            "equivalence_bound_abs_sd": bound,
            "tost_p_lower": np.nan,
            "tost_p_upper": np.nan,
            "tost_pvalue": np.nan,
            "equivalent_within_bound_0_05": 0,
            "mde_80pct_power_abs_sd": np.nan,
        }
    t_lower = (estimate + bound) / se
    t_upper = (estimate - bound) / se
    p_lower = float(stats.t.sf(t_lower, df=fit.df_clusters))
    p_upper = float(stats.t.cdf(t_upper, df=fit.df_clusters))
    tost_p = max(p_lower, p_upper)
    mde = (
        stats.t.ppf(0.975, fit.df_clusters) + stats.t.ppf(0.80, fit.df_clusters)
    ) * se
    return {
        "equivalence_bound_abs_sd": bound,
        "tost_p_lower": p_lower,
        "tost_p_upper": p_upper,
        "tost_pvalue": tost_p,
        "equivalent_within_bound_0_05": int(tost_p < 0.05),
        "mde_80pct_power_abs_sd": float(mde),
    }


def run_redesigned_main_models(
    panel: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = [
        ("H1", "Signed forward-GVC linkage change", "gvc_signed_change", "positive"),
        ("H2", "Log export recovery", "log_export_recovery", "positive"),
        (
            "H3",
            "Partner diversification excluding US and China",
            "partner_diversification_excl_us_china",
            "positive",
        ),
    ]
    rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    terms = ["post_x_exposure", "post_x_eci_pre", "eci_pre_x_exposure_post"]
    for offset, (hypothesis, label, outcome, expected_sign) in enumerate(outcomes):
        model_df = add_linear_did_terms(panel, "z_nexus_exposure_pre")
        fit = fit_twfe(model_df, outcome, terms)
        target = np.zeros(len(fit.term_names), dtype=float)
        target[fit.term_index("eci_pre_x_exposure_post")] = 1.0
        base = contrast_statistics(fit, target, alternative="two-sided")
        bootstrap = wild_cluster_bootstrap_contrast(
            fit,
            target,
            reps=bootstrap_reps,
            seed=seed + offset * 10,
            alternative="two-sided",
        )
        rows.append(
            {
                "hypothesis": hypothesis,
                "outcome": label,
                "outcome_col": outcome,
                "expected_sign": expected_sign,
                "term": "eci_pre_x_exposure_post",
                **base,
                **bootstrap,
                "n_obs": fit.n_obs,
                "n_countries": fit.n_countries,
                "n_years": fit.n_years,
                "r2_within": fit.r2_within,
            }
        )

        standardized = model_df.copy()
        standardized[f"z_{outcome}"] = zscore(standardized[outcome])
        fit_std = fit_twfe(standardized, f"z_{outcome}", terms)
        equivalence_rows.append(
            {
                "hypothesis": hypothesis,
                "outcome": label,
                "outcome_col": outcome,
                "standardized_term": "eci_pre_x_exposure_post",
                "coef_standardized_outcome": fit_std.term_row("eci_pre_x_exposure_post")[
                    "coef"
                ],
                "se_standardized_outcome": fit_std.term_row("eci_pre_x_exposure_post")["se"],
                **equivalence_and_mde(fit_std, "eci_pre_x_exposure_post"),
                "n_obs": fit_std.n_obs,
                "n_countries": fit_std.n_countries,
            }
        )
    result = pd.DataFrame(rows)
    result["qvalue_bh_h1_h3_family"] = bh_fdr_adjust(result["p_wild_bootstrap"])
    return result, pd.DataFrame(equivalence_rows)


def add_h4_terms(df: pd.DataFrame, exposure_col: str) -> pd.DataFrame:
    out = df.copy()
    post = out["post_tariff"]
    exposure = out[exposure_col]
    eci = out["z_eci_pre_raw"]
    coi = out["z_coi_pre_raw"]
    out["post_x_exposure"] = post * exposure
    out["post_x_eci_pre"] = post * eci
    out["post_x_coi_pre"] = post * coi
    out["post_x_exposure_x_eci_pre"] = post * exposure * eci
    out["post_x_exposure_x_coi_pre"] = post * exposure * coi
    out["post_x_eci_pre_x_coi_pre"] = post * eci * coi
    out["h4_eci_exposure_coi_post"] = post * exposure * eci * coi
    return out


def run_h4_models(
    panel: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = [
        ("H1 outcome", "gvc_signed_change"),
        ("H2 outcome", "log_export_recovery"),
        ("H3 outcome", "partner_diversification_excl_us_china"),
    ]
    terms = [
        "post_x_exposure",
        "post_x_eci_pre",
        "post_x_coi_pre",
        "post_x_exposure_x_eci_pre",
        "post_x_exposure_x_coi_pre",
        "post_x_eci_pre_x_coi_pre",
        "h4_eci_exposure_coi_post",
    ]
    prepared = add_h4_terms(panel, "z_nexus_exposure_pre")
    rows: list[dict[str, Any]] = []
    long_parts: list[pd.DataFrame] = []
    for offset, (label, outcome) in enumerate(outcomes):
        fit = fit_twfe(prepared, outcome, terms)
        contrast = np.zeros(len(fit.term_names), dtype=float)
        contrast[fit.term_index("h4_eci_exposure_coi_post")] = 1.0
        base = contrast_statistics(fit, contrast, alternative="greater")
        boot = wild_cluster_bootstrap_contrast(
            fit,
            contrast,
            reps=bootstrap_reps,
            seed=seed + offset * 10,
            alternative="greater",
        )
        rows.append(
            {
                "hypothesis": "H4",
                "outcome": label,
                "outcome_col": outcome,
                "term": "h4_eci_exposure_coi_post",
                "expected_direction": "positive",
                **base,
                **boot,
                "n_obs": fit.n_obs,
                "n_countries": fit.n_countries,
                "n_years": fit.n_years,
            }
        )

        part = prepared[
            [
                "country_iso3_code",
                "year",
                outcome,
            ]
            + terms
        ].dropna().copy()
        part["outcome_name"] = label
        part["stacked_outcome"] = zscore(part[outcome])
        long_parts.append(part)

    long_df = pd.concat(long_parts, ignore_index=True)
    stacked_terms: list[str] = []
    for label, _ in outcomes:
        label_key = label.replace(" ", "_")
        for term in terms:
            name = f"{label_key}__{term}"
            long_df[name] = np.where(long_df["outcome_name"] == label, long_df[term], 0.0)
            stacked_terms.append(name)
    long_df["entity_outcome"] = (
        long_df["country_iso3_code"].astype(str) + "__" + long_df["outcome_name"].astype(str)
    )
    long_df["year_outcome"] = (
        long_df["year"].astype(str) + "__" + long_df["outcome_name"].astype(str)
    )
    stacked_fit = fit_twfe(
        long_df,
        "stacked_outcome",
        stacked_terms,
        entity_col="entity_outcome",
        time_col="year_outcome",
        cluster_col="country_iso3_code",
    )
    target_terms = [
        f"{label.replace(' ', '_')}__h4_eci_exposure_coi_post" for label, _ in outcomes
    ]
    idx = [stacked_fit.term_index(term) for term in target_terms]
    beta = stacked_fit.beta[idx]
    cov = stacked_fit.covariance[np.ix_(idx, idx)]
    wald_stat = float(beta.T @ np.linalg.pinv(cov) @ beta)
    wald_p = float(stats.chi2.sf(wald_stat, df=len(idx)))
    omnibus = pd.DataFrame(
        [
            {
                "hypothesis": "H4",
                "test": "joint_wald_all_three_outcome_specific_interactions_zero",
                "wald_chi2": wald_stat,
                "df": len(idx),
                "pvalue_cluster_wald": wald_p,
                "n_obs_stacked": stacked_fit.n_obs,
                "n_country_clusters": stacked_fit.n_clusters,
            }
        ]
    )
    detail = pd.DataFrame(rows)
    detail["qvalue_bh_h4_outcome_family"] = bh_fdr_adjust(detail["p_wild_bootstrap"])
    return detail, omnibus


def run_observed_gpr_models(
    panel: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = panel[panel["gpr_country_annual"].notna()].copy()
    observed["z_gpr_observed"] = zscore(observed["gpr_country_annual"])
    outcomes = [
        ("H1 outcome", "gvc_signed_change"),
        ("H2 outcome", "log_export_recovery"),
        ("H3 outcome", "partner_diversification_excl_us_china"),
    ]
    rows: list[dict[str, Any]] = []
    for offset, (label, outcome) in enumerate(outcomes):
        d = observed.copy()
        p = d["post_tariff"]
        e = d["z_nexus_exposure_pre"]
        c = d["z_eci_pre_raw"]
        g = d["z_gpr_observed"]
        d["post_x_exposure"] = p * e
        d["post_x_eci"] = p * c
        d["post_x_gpr"] = p * g
        d["post_x_exposure_x_eci"] = p * e * c
        d["post_x_exposure_x_gpr"] = p * e * g
        d["post_x_eci_x_gpr"] = p * c * g
        d["h5_eci_exposure_gpr_post"] = p * e * c * g
        terms = [
            "post_x_exposure",
            "post_x_eci",
            "post_x_gpr",
            "post_x_exposure_x_eci",
            "post_x_exposure_x_gpr",
            "post_x_eci_x_gpr",
            "h5_eci_exposure_gpr_post",
        ]
        try:
            fit = fit_twfe(d, outcome, terms)
            contrast = np.zeros(len(fit.term_names), dtype=float)
            contrast[fit.term_index("h5_eci_exposure_gpr_post")] = 1.0
            base = contrast_statistics(fit, contrast, alternative="less")
            boot = wild_cluster_bootstrap_contrast(
                fit,
                contrast,
                reps=bootstrap_reps,
                seed=seed + offset * 10,
                alternative="less",
            )
            rows.append(
                {
                    "hypothesis": "H5 exploratory observed-GPR-only",
                    "outcome": label,
                    "outcome_col": outcome,
                    "expected_direction": "negative",
                    **base,
                    **boot,
                    "n_obs": fit.n_obs,
                    "n_countries": fit.n_countries,
                    "n_years": fit.n_years,
                }
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            rows.append(
                {
                    "hypothesis": "H5 exploratory observed-GPR-only",
                    "outcome": label,
                    "outcome_col": outcome,
                    "expected_direction": "negative",
                    "error": str(exc),
                }
            )
    coverage = (
        observed.groupby("year", as_index=False)
        .agg(
            observed_country_gpr_rows=("country_iso3_code", "size"),
            observed_country_gpr_countries=("country_iso3_code", "nunique"),
        )
    )
    coverage["all_panel_rows"] = panel.groupby("year").size().reindex(coverage["year"]).to_numpy()
    coverage["observed_share"] = (
        coverage["observed_country_gpr_rows"] / coverage["all_panel_rows"]
    )
    return pd.DataFrame(rows), coverage


def run_income_heterogeneity(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = panel.dropna(
        subset=[
            "log_export_recovery",
            "z_nexus_exposure_pre",
            "z_eci_pre_raw",
            "wb_income_id",
        ]
    ).copy()
    groups = sorted(d["wb_income_id"].astype(str).unique())
    reference = "HIC" if "HIC" in groups else groups[0]
    p = d["post_tariff"]
    e = d["z_nexus_exposure_pre"]
    c = d["z_eci_pre_raw"]
    d["post_x_exposure"] = p * e
    d["post_x_eci"] = p * c
    d["base_eci_exposure_post"] = p * e * c
    terms = ["post_x_exposure", "post_x_eci", "base_eci_exposure_post"]
    modifier_terms: list[str] = []
    for group in groups:
        if group == reference:
            continue
        flag = (d["wb_income_id"].astype(str) == group).astype(int)
        for short_name, value in [
            ("post", p),
            ("post_exposure", p * e),
            ("post_eci", p * c),
            ("eci_exposure_post", p * e * c),
        ]:
            name = f"{group}_{short_name}_modifier"
            d[name] = flag * value
            terms.append(name)
            if short_name == "eci_exposure_post":
                modifier_terms.append(name)
    fit = fit_twfe(d, "log_export_recovery", terms)
    rows: list[dict[str, Any]] = []
    base_idx = fit.term_index("base_eci_exposure_post")
    for group in groups:
        contrast = np.zeros(len(fit.term_names), dtype=float)
        contrast[base_idx] = 1.0
        if group != reference:
            contrast[fit.term_index(f"{group}_eci_exposure_post_modifier")] = 1.0
        result = contrast_statistics(fit, contrast, alternative="two-sided")
        rows.append(
            {
                "income_group": group,
                "reference_group": reference,
                "group_specific_eci_exposure_effect": result["estimate"],
                "se": result["se"],
                "pvalue_two_sided": result["p_cluster"],
                "n_obs": fit.n_obs,
                "n_countries": fit.n_countries,
            }
        )
    if modifier_terms:
        idx = [fit.term_index(term) for term in modifier_terms]
        beta = fit.beta[idx]
        cov = fit.covariance[np.ix_(idx, idx)]
        stat = float(beta.T @ np.linalg.pinv(cov) @ beta)
        pvalue = float(stats.chi2.sf(stat, df=len(idx)))
    else:
        stat = np.nan
        pvalue = np.nan
    omnibus = pd.DataFrame(
        [
            {
                "outcome": "Log export recovery",
                "reference_group": reference,
                "test": "joint_wald_income_group_differences_in_eci_exposure_post_effect",
                "wald_chi2": stat,
                "df": len(modifier_terms),
                "pvalue_cluster_wald": pvalue,
                "n_obs": fit.n_obs,
                "n_countries": fit.n_countries,
            }
        ]
    )
    return pd.DataFrame(rows), omnibus


def run_mechanism_models(
    panel: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    outcomes = [
        (
            "New non-US/China destination entries",
            "new_non_uschina_destination_count",
            "positive",
        ),
        (
            "New non-US/China destination export share",
            "new_non_uschina_destination_export_share",
            "positive",
        ),
        (
            "Persistent new non-US/China destination entries",
            "persistent_new_non_uschina_destination_count",
            "positive",
        ),
    ]
    terms = ["post_x_exposure", "post_x_eci_pre", "eci_pre_x_exposure_post"]
    d = add_linear_did_terms(panel, "z_nexus_exposure_pre")
    rows: list[dict[str, Any]] = []
    for offset, (label, outcome, expected) in enumerate(outcomes):
        fit = fit_twfe(d, outcome, terms)
        contrast = np.zeros(len(fit.term_names), dtype=float)
        contrast[fit.term_index("eci_pre_x_exposure_post")] = 1.0
        base = contrast_statistics(fit, contrast, alternative="greater")
        boot = wild_cluster_bootstrap_contrast(
            fit,
            contrast,
            reps=bootstrap_reps,
            seed=seed + offset * 10,
            alternative="greater",
        )
        rows.append(
            {
                "mechanism": "Market search and redirection",
                "outcome": label,
                "outcome_col": outcome,
                "expected_direction": expected,
                **base,
                **boot,
                "n_obs": fit.n_obs,
                "n_countries": fit.n_countries,
                "n_years": fit.n_years,
            }
        )
    table = pd.DataFrame(rows)
    table["qvalue_bh_mechanism_family"] = bh_fdr_adjust(table["p_wild_bootstrap"])
    return table


def run_temporal_h6(panel: pd.DataFrame, threshold: float) -> pd.DataFrame:
    d = panel.dropna(
        subset=["log_export_recovery", "eci_pre_raw", "z_nexus_exposure_pre"]
    ).copy()
    d = prepare_piecewise_data(d, threshold, "z_nexus_exposure_pre")
    period_defs = {
        "tariff_implementation_2019": d["year"].eq(2019).astype(int),
        "covid_2020_2021": d["year"].between(2020, 2021).astype(int),
        "later_shock_2022": d["year"].eq(2022).astype(int),
    }
    terms: list[str] = []
    for name, indicator in period_defs.items():
        for source in [
            "z_nexus_exposure_pre",
            "eci_below_threshold",
            "eci_above_threshold",
            "h6_low_slope",
            "h6_high_slope",
        ]:
            term = f"{name}__{source}"
            if source == "z_nexus_exposure_pre":
                d[term] = indicator * d[source]
            elif source in ["eci_below_threshold", "eci_above_threshold"]:
                d[term] = indicator * d[source]
            else:
                d[term] = indicator * d[source] / d["post_tariff"].replace(0, np.nan)
                d[term] = d[term].fillna(0.0)
            terms.append(term)
    fit = fit_twfe(d, "log_export_recovery", terms)
    rows: list[dict[str, Any]] = []
    for name in period_defs:
        low = np.zeros(len(fit.term_names), dtype=float)
        high = np.zeros(len(fit.term_names), dtype=float)
        low[fit.term_index(f"{name}__h6_low_slope")] = 1.0
        high[fit.term_index(f"{name}__h6_high_slope")] = 1.0
        for test, contrast, alternative in [
            ("H6a_beta_low_lt_0", low, "less"),
            ("H6b_beta_high_minus_low_gt_0", high - low, "greater"),
        ]:
            rows.append(
                {
                    "period": name,
                    "test": test,
                    **contrast_statistics(fit, contrast, alternative=alternative),
                    "n_obs": fit.n_obs,
                    "n_countries": fit.n_countries,
                }
            )
    table = pd.DataFrame(rows)
    table["qvalue_bh_temporal_exploration"] = bh_fdr_adjust(table["p_cluster"])
    return table


def _knn_prediction(
    recipient: pd.Series,
    donors: pd.DataFrame,
    features: list[str],
    k: int,
) -> float | None:
    valid_donors = donors.dropna(subset=["gpr_country_annual"]).copy()
    if valid_donors.empty:
        return None
    scales = valid_donors[features].std(skipna=True, ddof=0).replace(0, np.nan)
    distances: list[tuple[float, float]] = []
    for _, donor in valid_donors.iterrows():
        pieces: list[float] = []
        for feature in features:
            rv = recipient.get(feature, np.nan)
            dv = donor.get(feature, np.nan)
            scale = scales.get(feature, np.nan)
            if pd.notna(rv) and pd.notna(dv) and pd.notna(scale) and scale > 0:
                pieces.append(((float(rv) - float(dv)) / float(scale)) ** 2)
        if len(pieces) >= 2:
            distances.append((float(np.sqrt(np.mean(pieces))), float(donor["gpr_country_annual"])))
    if not distances:
        return None
    distances = sorted(distances, key=lambda item: item[0])[: min(k, len(distances))]
    distance_values = np.array([item[0] for item in distances], dtype=float)
    gpr_values = np.array([item[1] for item in distances], dtype=float)
    weights = 1.0 / (distance_values + 1e-6)
    return float(np.sum(weights * gpr_values) / np.sum(weights))


def run_gpr_imputation_audit(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = [
        "eci_pre_raw",
        "coi_pre_raw",
        "z_nexus_exposure_pre",
        "baseline_log_gdp_pc",
        "baseline_trade_open",
        "baseline_wgi",
    ]
    observed = panel.dropna(subset=["gpr_country_annual"]).copy()
    rows: list[dict[str, Any]] = []
    for k in [3, 5, 10]:
        for year, group in observed.groupby("year"):
            for idx, row in group.iterrows():
                donors = group[group.index != idx]
                prediction = _knn_prediction(row, donors, features, k=k)
                if prediction is None:
                    continue
                rows.append(
                    {
                        "k": k,
                        "year": int(year),
                        "country_iso3_code": row["country_iso3_code"],
                        "observed_gpr": float(row["gpr_country_annual"]),
                        "predicted_gpr": prediction,
                        "squared_error": (prediction - float(row["gpr_country_annual"])) ** 2,
                        "absolute_error": abs(prediction - float(row["gpr_country_annual"])),
                    }
                )
    detail = pd.DataFrame(rows)
    if detail.empty:
        validation = pd.DataFrame(
            columns=["k", "n_masked", "rmse", "mae", "correlation"]
        )
    else:
        validation_rows: list[dict[str, Any]] = []
        for k, group in detail.groupby("k"):
            corr = group["observed_gpr"].corr(group["predicted_gpr"])
            validation_rows.append(
                {
                    "k": int(k),
                    "n_masked": int(len(group)),
                    "rmse": float(np.sqrt(group["squared_error"].mean())),
                    "mae": float(group["absolute_error"].mean()),
                    "correlation": float(corr) if pd.notna(corr) else np.nan,
                }
            )
        validation = pd.DataFrame(validation_rows)
    return detail, validation


def plot_h6_curve(
    fit: FEFit,
    threshold: float,
    panel: pd.DataFrame,
    out_dir: Path,
) -> None:
    values = panel["eci_pre_raw"].dropna()
    x = np.linspace(values.quantile(0.02), values.quantile(0.98), 250)
    low = np.minimum(x - threshold, 0.0)
    high = np.maximum(x - threshold, 0.0)
    effect = np.zeros(len(x), dtype=float)
    se = np.zeros(len(x), dtype=float)
    idx_pe = fit.term_index("post_x_exposure")
    idx_low = fit.term_index("h6_low_slope")
    idx_high = fit.term_index("h6_high_slope")
    for i, (lo, hi) in enumerate(zip(low, high)):
        contrast = np.zeros(len(fit.term_names), dtype=float)
        contrast[idx_pe] = 1.0
        contrast[idx_low] = lo
        contrast[idx_high] = hi
        effect[i] = float(contrast @ fit.beta)
        se[i] = np.sqrt(max(float(contrast @ fit.covariance @ contrast), 0.0))
    crit = stats.t.ppf(0.975, fit.df_clusters)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(x, effect, color="#1b5e20", linewidth=2.4)
    ax.fill_between(x, effect - crit * se, effect + crit * se, color="#a5d6a7", alpha=0.55)
    ax.axhline(0, color="#444444", linewidth=0.9)
    ax.axvline(threshold, color="#c62828", linewidth=1.4, linestyle="--")
    ax.text(
        threshold,
        ax.get_ylim()[1],
        f" Threshold = {threshold:.2f}",
        color="#b71c1c",
        va="top",
        ha="left",
        fontsize=9,
    )
    ax.set_xlabel("Pre-shock Economic Complexity Index (2015-2017 average)")
    ax.set_ylabel(
        "Effect of +1 SD US-China nexus exposure on log export recovery"
    )
    ax.set_title("Capability-conversion threshold: estimated exposure effect by baseline ECI")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_h6_capability_conversion_threshold.png", dpi=220)
    fig.savefig(out_dir / "figure_h6_capability_conversion_threshold.pdf")
    plt.close(fig)


def plot_redesigned_estimates(
    main_models: pd.DataFrame,
    h4_models: pd.DataFrame,
    mechanism_models: pd.DataFrame,
    out_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for _, row in main_models.iterrows():
        rows.append(
            {
                "label": str(row["hypothesis"]) + ": " + str(row["outcome"]),
                "estimate": row["estimate"],
                "low": row["ci_low_95"],
                "high": row["ci_high_95"],
                "group": "H1-H3",
            }
        )
    for _, row in h4_models.iterrows():
        rows.append(
            {
                "label": "H4: " + str(row["outcome"]),
                "estimate": row["estimate"],
                "low": row["ci_low_95"],
                "high": row["ci_high_95"],
                "group": "H4",
            }
        )
    for _, row in mechanism_models.iterrows():
        rows.append(
            {
                "label": "Mechanism: " + str(row["outcome"]),
                "estimate": row["estimate"],
                "low": row["ci_low_95"],
                "high": row["ci_high_95"],
                "group": "Mechanism",
            }
        )
    plot_df = pd.DataFrame(rows).dropna()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10.4, max(4.8, 0.52 * len(plot_df) + 1.4)))
    y = np.arange(len(plot_df))
    colors = plot_df["group"].map(
        {"H1-H3": "#1565c0", "H4": "#6a1b9a", "Mechanism": "#ef6c00"}
    )
    ax.errorbar(
        plot_df["estimate"],
        y,
        xerr=[
            plot_df["estimate"] - plot_df["low"],
            plot_df["high"] - plot_df["estimate"],
        ],
        fmt="none",
        ecolor='#546e7a',
        elinewidth=1.6,
        capsize=3,
        zorder=1,
    )
    ax.scatter(plot_df["estimate"], y, s=42, c=colors.tolist(), zorder=2)
    ax.axvline(0, color="#444444", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Estimate with 95% country-clustered confidence interval")
    ax.set_title("Redesigned pre-shock capability and exposure estimates")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_redesigned_estimates.png", dpi=220)
    fig.savefig(out_dir / "figure_redesigned_estimates.pdf")
    plt.close(fig)


def format_test_line(row: pd.Series) -> str:
    return (
        f"{row['test']}: estimate={row['estimate']:.4f}, "
        f"wild-bootstrap p={row['p_wild_bootstrap']:.4g}, "
        f"FDR q={row['qvalue_bh_full_h6_family']:.4g}"
    )


def write_summary(
    out_dir: Path,
    threshold_summary: pd.DataFrame,
    h6_tests: pd.DataFrame,
    holdout_summary: pd.DataFrame,
    h4_omnibus: pd.DataFrame,
    gpr_coverage: pd.DataFrame,
    validation_strata: str,
) -> None:
    primary = h6_tests[h6_tests["spec"] == "primary_log_continuous_exposure"]
    threshold = threshold_summary.iloc[0]
    h6a = primary[primary["test"] == "H6a_beta_low_lt_0"].iloc[0]
    h6b = primary[primary["test"] == "H6b_beta_high_minus_low_gt_0"].iloc[0]
    direction_primary = bool((h6a["estimate"] < 0) and (h6b["estimate"] > 0))
    fdr_primary = bool(
        (h6a["qvalue_bh_full_h6_family"] < 0.05)
        and (h6b["qvalue_bh_full_h6_family"] < 0.05)
    )
    robustness = h6_tests[h6_tests["spec"] != "primary_log_continuous_exposure"]
    robustness_direction = float(
        (
            ((robustness["test"] == "H6a_beta_low_lt_0") & (robustness["estimate"] < 0))
            | (
                (robustness["test"] == "H6b_beta_high_minus_low_gt_0")
                & (robustness["estimate"] > 0)
            )
        ).mean()
    )
    holdout = holdout_summary.iloc[0]

    lines = [
        "# Capability-Conversion Threshold Redesign",
        "",
        "## Classification",
        "",
        "H6 is a theory-refining, exploratory analysis. The low-complexity pattern was discovered in this dataset, so this package does not relabel it as confirmatory evidence.",
        "",
        "## H6 Statement",
        "",
        "H6: Following tariff-induced disruption, increases in productive complexity are associated with weaker export recovery among countries below a minimum level of baseline productive complexity; this negative relationship attenuates once countries possess a sufficiently broad productive capability base.",
        "",
        "## Design Corrections",
        "",
        "- ECI and COI are frozen at their 2015-2017 country averages.",
        "- The primary exposure is continuous pre-shock US-China trade-nexus intensity, standardized across countries. Top-tercile and top-quartile exposure are robustness checks.",
        "- The primary recovery outcome is log export recovery. Ratio, winsorized-ratio, and pretrend-adjusted outcomes are robustness checks.",
        "- The primary model has country and year fixed effects and does not condition on contemporaneous controls that may be post-treatment. A pre-shock-controls-by-year sensitivity is included.",
        "- H5 is not used as primary evidence because country GPR is directly observed for only a limited country subset and no deterministic imputation is used here.",
        "",
        "## Threshold",
        "",
        f"- Selected baseline-ECI breakpoint: {threshold['threshold_eci_pre']:.4f}.",
        f"- Wild-bootstrap 95% percentile interval over the threshold grid: {threshold['threshold_ci_low_95_wild']:.4f} to {threshold['threshold_ci_high_95_wild']:.4f}.",
        f"- Threshold search grid: {threshold['threshold_grid_quantiles']}.",
        "",
        "## Primary H6 Tests",
        "",
        f"- {format_test_line(h6a)}.",
        f"- {format_test_line(h6b)}.",
        f"- Directional pattern (both H6 conditions): {direction_primary}.",
        f"- Both primary tests survive FDR across the complete reported H6 exploration family: {fdr_primary}.",
        "",
        "## Robustness and Internal Validation",
        "",
        f"- Share of non-primary H6 test rows with the predicted directional sign: {robustness_direction:.3f}.",
        f"- Repeated {validation_strata}-stratified country holdouts: {int(holdout['splits_successful'])}/{int(holdout['splits_requested'])} successful splits.",
        f"- Holdout share with negative low-regime slope: {holdout['share_low_negative']:.3f}.",
        f"- Holdout share with positive high-minus-low contrast: {holdout['share_difference_positive']:.3f}.",
        f"- Holdout share with both predicted directions: {holdout['share_both_directions']:.3f}.",
        "",
        "## Redesigned H1-H5 Context",
        "",
        "The redesigned H1-H3 table uses frozen pre-shock capability measures, continuous exposure, signed GVC change, log export recovery, and diversification that excludes the United States and China. The equivalence/MDE table distinguishes precise near-zero estimates from imprecise ones.",
        f"- H4 omnibus joint test across outcome-specific interactions: chi2={h4_omnibus.iloc[0]['wald_chi2']:.4f}, p={h4_omnibus.iloc[0]['pvalue_cluster_wald']:.4g}.",
        f"- Directly observed country-GPR coverage ranges from {gpr_coverage['observed_share'].min():.3f} to {gpr_coverage['observed_share'].max():.3f} of country-year rows by year.",
        "",
        "## Measurement Scope",
        "",
        "- The local bilateral source is country-partner-year data, not HS product-level data. It supports continuous nexus exposure, US-market dependence, China import dependence, diversification excluding US/China, and destination-entry mechanisms, but not a valid tariff-weighted product exposure measure.",
        "- The direct mechanism test therefore focuses on market search and redirection (new and persistent non-US/China export destinations). Product reallocation and input-substitution mechanisms require a product-level trade/input dataset.",
        "- GPR profile validation uses only frozen pre-shock covariates; it does not feed imputed values into the primary models.",
        "",
        "## Files",
        "",
        "- h6_threshold_search.csv and h6_threshold_bootstrap_distribution.csv: breakpoint search and uncertainty.",
        "- h6_primary_and_robustness_tests.csv: all H6 directional tests, 999-replication wild-bootstrap p-values, and full-family BH q-values.",
        "- h6_holdout_validation.csv: repeated country-holdout validation.",
        "- redesigned_h1_h3_tests.csv, h4_redesigned_tests.csv, and h4_omnibus_joint_test.csv: redesigned baseline and moderation analyses.",
        "- mechanism_destination_entry_tests.csv: direct market-search/redirection tests.",
        "- gpr_observed_coverage.csv and gpr_pre_shock_profile_validation.csv: GPR coverage and no-outcome-leakage imputation audit.",
        "- figure_h6_capability_conversion_threshold.png: marginal exposure effect over baseline ECI.",
    ]
    (out_dir / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")

    tex_lines = [
        "% Theory-refining H6 addendum generated from the capability-conversion redesign.",
        "\\subsubsection{Capability-Conversion Threshold}",
        "",
        "The aggregate resilience effect of productive complexity may be non-linear. A country can develop isolated sophisticated export activities before it has the broad supplier, labor, and adjacent-product systems needed to convert those activities into economy-wide adjustment capacity. In this intermediate state, greater complexity may increase exposure to demanding international production networks without yet providing sufficient redundancy, substitutability, or reallocation capacity. Once a broader capability base is present, additional complexity should become more convertible into export recovery.",
        "",
        "\\begin{quote}",
        "H6: Following tariff-induced disruption, increases in productive complexity are associated with weaker export recovery among countries below a minimum level of baseline productive complexity; this negative relationship attenuates once countries possess a sufficiently broad productive capability base.",
        "\\end{quote}",
        "",
        "H6 is theory-refining rather than confirmatory because the motivating low-ECI pattern was identified in the current dataset. We operationalize baseline productive complexity as the 2015--2017 average ECI and estimate a pooled segmented country and year fixed-effects model with continuous pre-shock US--China nexus exposure. The selected breakpoint is "
        + f"{threshold['threshold_eci_pre']:.3f}"
        + " (wild-bootstrap grid interval "
        + f"{threshold['threshold_ci_low_95_wild']:.3f}"
        + " to "
        + f"{threshold['threshold_ci_high_95_wild']:.3f}"
        + "). The primary outcome is log export recovery, with ratio-based, winsorized, pretrend-adjusted, small-economy exclusion, export-size-weighted, and alternative-exposure specifications reported as robustness checks.",
        "",
        "The decisive tests are a negative low-regime slope and a positive difference between the high- and low-regime slopes. We report 999-replication country wild-bootstrap p-values, Benjamini--Hochberg adjusted q-values over the complete reported H6 exploration family, and repeated stratified country-holdout validation.",
    ]
    (out_dir / "h6_theory_methods_addendum.tex").write_text(
        "\n".join(tex_lines), encoding="utf-8"
    )


def write_metadata(
    out_dir: Path,
    panel: pd.DataFrame,
    frozen: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    bootstrap_reps: int,
    threshold_bootstrap_reps: int,
    holdout_splits: int,
    validation_strata: str,
) -> None:
    metadata = {
        "analysis_type": "theory_refining_exploratory_H6",
        "input_panel": "data/processed/regression_panel_2012_2022.csv",
        "input_bilateral": "data/processed/source_atlas_country_country_year_2012_2022.csv",
        "baseline_years": [BASELINE_START, BASELINE_END],
        "post_start": POST_START,
        "countries_in_panel": int(panel["country_iso3_code"].nunique()),
        "rows_in_panel": int(len(panel)),
        "countries_with_frozen_eci": int(frozen["eci_pre_raw"].notna().sum()),
        "primary_exposure": "z_nexus_exposure_pre",
        "primary_outcome": "log_export_recovery",
        "gvc_construct": "signed annual change in TiVA forward-linkage share",
        "partner_diversification_construct": "1-HHI after excluding US and China",
        "h6_threshold": threshold_summary.iloc[0].to_dict(),
        "wild_cluster_bootstrap_reps": int(bootstrap_reps),
        "threshold_wild_bootstrap_reps": int(threshold_bootstrap_reps),
        "holdout_splits": int(holdout_splits),
        "holdout_stratification": validation_strata,
        "product_level_tariff_weighted_exposure_available": False,
        "product_level_tariff_weighted_exposure_reason": (
            "Stored bilateral data are country-partner-year aggregates without product identifiers."
        ),
        "h5_primary_status": "not_primary_due_to_limited_observed_country_gpr_coverage",
    }
    (out_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen pre-shock capability-conversion H6 analysis package."
    )
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root directory.",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=999)
    parser.add_argument("--threshold-bootstrap-reps", type=int, default=999)
    parser.add_argument("--holdout-splits", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_dir = base_dir / "reports" / "capability_conversion_redesign"
    ensure_dir(out_dir)

    panel = pd.read_csv(base_dir / "data" / "processed" / "regression_panel_2012_2022.csv")
    bilateral = pd.read_csv(
        base_dir / "data" / "processed" / "source_atlas_country_country_year_2012_2022.csv"
    )
    income_reference = pd.read_csv(
        base_dir / "reports" / "class_based_tests" / "world_bank_income_groups_reference.csv"
    )
    regions_path = base_dir / "data" / "processed" / "world_bank_country_regions.csv"
    regions = pd.read_csv(regions_path) if regions_path.exists() else None
    if regions is not None and "wb_region" in regions:
        regions["wb_region"] = regions["wb_region"].astype("string").str.strip()

    analysis_panel, frozen = freeze_pre_shock_constructs(
        panel, bilateral, income_reference, regions
    )
    analysis_panel.to_csv(out_dir / "panel_with_frozen_pre_shock_constructs.csv", index=False)
    frozen.to_csv(out_dir / "frozen_country_constructs.csv", index=False)

    audit = pd.DataFrame(
        [
            {"metric": "panel_rows", "value": len(analysis_panel)},
            {"metric": "countries", "value": analysis_panel["country_iso3_code"].nunique()},
            {
                "metric": "countries_with_frozen_eci",
                "value": frozen["eci_pre_raw"].notna().sum(),
            },
            {
                "metric": "log_export_recovery_rows",
                "value": analysis_panel["log_export_recovery"].notna().sum(),
            },
            {
                "metric": "ratio_export_recovery_max",
                "value": analysis_panel["export_recovery_index"].max(),
            },
            {
                "metric": "gpr_observed_rows",
                "value": analysis_panel["gpr_country_annual"].notna().sum(),
            },
            {
                "metric": "gpr_observed_countries",
                "value": analysis_panel.loc[
                    analysis_panel["gpr_country_annual"].notna(), "country_iso3_code"
                ].nunique(),
            },
        ]
    )
    audit.to_csv(out_dir / "data_construction_audit.csv", index=False)

    (
        h6_tests,
        h6_coefficients,
        threshold_summary,
        threshold_search,
        threshold_distribution,
        best_threshold,
        primary_h6_fit,
    ) = run_h6_suite(
        analysis_panel,
        bootstrap_reps=int(args.bootstrap_reps),
        threshold_bootstrap_reps=int(args.threshold_bootstrap_reps),
        seed=int(args.seed),
    )
    h6_tests.to_csv(out_dir / "h6_primary_and_robustness_tests.csv", index=False)
    h6_coefficients.to_csv(out_dir / "h6_model_coefficients.csv", index=False)
    threshold_summary.to_csv(out_dir / "h6_threshold_summary.csv", index=False)
    threshold_search.to_csv(out_dir / "h6_threshold_search.csv", index=False)
    threshold_distribution.to_csv(
        out_dir / "h6_threshold_bootstrap_distribution.csv", index=False
    )

    validation_strata = "World Bank region" if regions is not None else "World Bank income group"
    strata_col = "wb_region" if regions is not None else "wb_income_id"
    holdout_detail, holdout_summary = run_holdout_validation(
        analysis_panel,
        strata_col=strata_col,
        splits=int(args.holdout_splits),
        seed=int(args.seed) + 3000,
    )
    holdout_detail.to_csv(out_dir / "h6_holdout_validation.csv", index=False)
    holdout_summary.to_csv(out_dir / "h6_holdout_validation_summary.csv", index=False)

    main_models, equivalence = run_redesigned_main_models(
        analysis_panel, bootstrap_reps=int(args.bootstrap_reps), seed=int(args.seed) + 4000
    )
    main_models.to_csv(out_dir / "redesigned_h1_h3_tests.csv", index=False)
    equivalence.to_csv(out_dir / "equivalence_mde.csv", index=False)

    h4_models, h4_omnibus = run_h4_models(
        analysis_panel, bootstrap_reps=int(args.bootstrap_reps), seed=int(args.seed) + 5000
    )
    h4_models.to_csv(out_dir / "h4_redesigned_tests.csv", index=False)
    h4_omnibus.to_csv(out_dir / "h4_omnibus_joint_test.csv", index=False)

    h5_observed, gpr_coverage = run_observed_gpr_models(
        analysis_panel, bootstrap_reps=int(args.bootstrap_reps), seed=int(args.seed) + 6000
    )
    h5_observed.to_csv(out_dir / "h5_observed_gpr_exploratory.csv", index=False)
    gpr_coverage.to_csv(out_dir / "gpr_observed_coverage.csv", index=False)
    gpr_validation_detail, gpr_validation = run_gpr_imputation_audit(analysis_panel)
    gpr_validation_detail.to_csv(out_dir / "gpr_pre_shock_profile_validation_detail.csv", index=False)
    gpr_validation.to_csv(out_dir / "gpr_pre_shock_profile_validation.csv", index=False)

    income_detail, income_omnibus = run_income_heterogeneity(analysis_panel)
    income_detail.to_csv(out_dir / "income_heterogeneity_log_recovery.csv", index=False)
    income_omnibus.to_csv(out_dir / "income_heterogeneity_log_recovery_omnibus.csv", index=False)

    mechanism_models = run_mechanism_models(
        analysis_panel, bootstrap_reps=int(args.bootstrap_reps), seed=int(args.seed) + 7000
    )
    mechanism_models.to_csv(out_dir / "mechanism_destination_entry_tests.csv", index=False)

    temporal_h6 = run_temporal_h6(analysis_panel, best_threshold)
    temporal_h6.to_csv(out_dir / "h6_temporal_regime_exploration.csv", index=False)

    plot_h6_curve(primary_h6_fit, best_threshold, analysis_panel, out_dir)
    plot_redesigned_estimates(main_models, h4_models, mechanism_models, out_dir)
    write_summary(
        out_dir,
        threshold_summary,
        h6_tests,
        holdout_summary,
        h4_omnibus,
        gpr_coverage,
        validation_strata,
    )
    write_metadata(
        out_dir,
        analysis_panel,
        frozen,
        threshold_summary,
        bootstrap_reps=int(args.bootstrap_reps),
        threshold_bootstrap_reps=int(args.threshold_bootstrap_reps),
        holdout_splits=int(args.holdout_splits),
        validation_strata=validation_strata,
    )

    print(f"Wrote capability-conversion redesign package: {out_dir}")
    print(f"Selected H6 threshold: {best_threshold:.4f}")
    print(f"H6 test rows: {len(h6_tests)}")


if __name__ == "__main__":
    main()
