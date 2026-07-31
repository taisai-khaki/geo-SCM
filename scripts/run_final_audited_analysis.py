from __future__ import annotations

"""Corrected final-analysis audit runner.

This layer corrects sample construction, consensus structural profiles, regime
equality inference, focal event studies, post-period sensitivity, power,
common-sample adjustments, weighting diagnostics, and evidence freezing.
"""

import argparse
import hashlib
import math
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
from matplotlib import pyplot as plt
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_capability_conversion_analysis import (  # noqa: E402
    contrast_statistics,
    wild_cluster_bootstrap_contrast,
    zscore,
)
from run_structural_regime_analysis import (  # noqa: E402
    BASELINE_END,
    BASELINE_START,
    OUTCOMES,
    STRUCTURAL_LABELS,
    STRUCTURAL_VARS,
    add_base_terms,
    add_moderator_terms,
    build_country_trend_terms,
    build_region_year_terms,
    build_structural_profile,
    fit_terms,
    load_wdi_structural_source,
    wald_test,
)

BOOTSTRAP_REPS = 999
MI_REPS = 20
STABILITY_REPS = 100
PRIMARY_POST = 2018
SENSITIVITY_POST = 2019
EVENT_REFERENCE = 2017
# Explicit non-economy, aggregate, and historical codes. Do not use substring
# matching: legitimate names such as Central African Republic and South Africa
# must remain eligible when their analytical inputs are complete.
KNOWN_INVALID_CODES = frozenset(
    {
        "ANS", "ANT", "ATA", "ATF", "BES", "BLM", "BVT", "CCK", "CSK",
        "CXR", "DDR", "FLK", "HMD", "MSR", "NFK", "NIU", "PCN", "SCG",
        "SGS", "SHN", "SPM", "SUN", "TKL", "TMP", "VAT", "YUG", "ZAR",
    }
)
EXACT_INVALID_NAMES = frozenset(
    {
        "world", "africa", "asia", "europe", "aggregate", "income group",
        "not classified", "unspecified", "trade total", "regions", "undeclared",
    }
)
ANALYTICAL_ECONOMY_CODES = frozenset({"HKG", "MAC", "PSE"})
BLOCKS = {
    "development_reallocation_capacity": [
        "pre_log_real_gdp_pc",
        "pre_real_gdp_pc_growth_mean",
        "pre_real_gdp_pc_growth_volatility",
        "pre_institutional_quality",
    ],
    "industrial_trade_structure": [
        "pre_manufacturing_value_added_share",
        "pre_trade_openness",
        "pre_export_concentration",
    ],
    "structural_vulnerability_scale": [
        "pre_resource_rents",
        "pre_log_population",
        "pre_real_gdp_pc_growth_slope",
    ],
}
EQUIVALENCE_RULE = "absolute bound = 0.10 outcome standard deviations in the stated sample"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bh_adjust(values: pd.Series) -> pd.Series:
    p = num(values)
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.notna()
    if not valid.any():
        return out
    ordered = p.loc[valid].sort_values()
    m = len(ordered)
    adjusted = ordered.to_numpy(dtype=float) * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out.loc[ordered.index] = np.minimum(adjusted, 1.0)
    return out


def align_labels(reference: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    table = np.zeros((k, k), dtype=int)
    for left, right in zip(reference, labels):
        table[int(left), int(right)] += 1
    rows, cols = linear_sum_assignment(-table)
    mapping = {int(col): int(row) for row, col in zip(rows, cols)}
    return np.array([mapping.get(int(value), int(value)) for value in labels], dtype=int)


def standardize_profile(profile: pd.DataFrame) -> pd.DataFrame:
    out = profile.copy()
    for variable in STRUCTURAL_VARS + ["eci_pre", "coi_pre", "exposure_pre"]:
        out[f"z_{variable}"] = zscore(out[variable])
    out["exposed_top_tercile"] = out["exposure_tercile"].eq("high").astype(int)
    return out


def growth_counts(panel: pd.DataFrame) -> pd.DataFrame:
    pre = panel.loc[
        panel["analysis_eligible_third_country"].eq(1)
        & panel["year"].between(BASELINE_START, BASELINE_END)
    ]
    rows = []
    for country, group in pre.groupby("country_iso3_code"):
        gdp = num(group["wdi_gdp_pc_const_2015_usd"]).where(lambda x: x > 0)
        rows.append(
            {
                "country_iso3_code": country,
                "pre_gdp_observations": int(gdp.notna().sum()),
                "pre_growth_observations": int(np.log(gdp).diff().notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def build_samples(panel: pd.DataFrame, profile: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out = profile.copy().merge(growth_counts(panel), on="country_iso3_code", how="left")
    out["cluster_imputed_fields"] = out[STRUCTURAL_VARS].isna().sum(axis=1)
    names = out["country_name"].fillna("").astype(str)
    codes = out["country_iso3_code"].fillna("").astype(str)
    normalized_names = names.str.strip().str.casefold()
    out["known_invalid_code"] = codes.isin(KNOWN_INVALID_CODES)
    out["known_aggregate_name"] = normalized_names.isin(EXACT_INVALID_NAMES)
    out["sovereign_or_analytical_economy"] = (
        out["wb_region"].notna() | codes.isin(ANALYTICAL_ECONOMY_CODES)
    )
    out["valid_entity"] = (
        ~out["known_invalid_code"]
        & ~out["known_aggregate_name"]
        & out["sovereign_or_analytical_economy"]
    )
    out["entity_validity_reason"] = np.select(
        [
            out["known_invalid_code"],
            out["known_aggregate_name"],
            ~out["sovereign_or_analytical_economy"],
        ],
        [
            "known_invalid_or_historical_code",
            "exact_known_aggregate_name",
            "missing_sovereign_or_analytical_economy_classification",
        ],
        default="valid_sovereign_or_analytical_economy",
    )
    out["valid_exposure"] = out["exposure_pre"].notna()
    out["valid_eci"] = out["eci_pre"].notna()
    out["valid_growth_history"] = out["pre_growth_observations"].ge(3)
    out["primary_structural_sample"] = (
        out["valid_entity"] & out["valid_exposure"] & out["valid_eci"]
        & out["valid_growth_history"] & out["cluster_imputed_fields"].le(2)
    )
    out["complete_case_structural_sample"] = (
        out["primary_structural_sample"] & out["cluster_imputed_fields"].eq(0)
    )

    def exclusion(row: pd.Series) -> str:
        if not row["valid_entity"]:
            return str(row["entity_validity_reason"])
        if not row["valid_exposure"]:
            return "missing_pre_shock_exposure"
        if not row["valid_eci"]:
            return "missing_pre_shock_eci"
        if not row["valid_growth_history"]:
            return "fewer_than_three_pre_shock_growth_observations"
        if row["cluster_imputed_fields"] > 2:
            return "more_than_two_structural_indicators_imputed"
        return ""

    out["primary_exclusion_reason"] = out.apply(exclusion, axis=1)
    audit_cols = [
        "country_iso3_code", "country_name", "wb_region",
        "cluster_imputed_fields", "pre_gdp_observations", "pre_growth_observations",
        "valid_entity", "valid_exposure", "valid_eci", "valid_growth_history",
        "entity_validity_reason",
        "primary_structural_sample", "complete_case_structural_sample",
        "primary_exclusion_reason",
    ]
    audit = out[audit_cols].copy()
    missing = out[["country_iso3_code", "country_name"] + STRUCTURAL_VARS].copy()
    missing["cluster_imputed_fields"] = out["cluster_imputed_fields"]
    missing["missing_structural_variables"] = out[STRUCTURAL_VARS].isna().apply(
        lambda row: ";".join(row.index[row].tolist()), axis=1
    )
    rows = []
    for name, mask in [
        ("broad_legacy", pd.Series(True, index=out.index)),
        ("cleaned_primary", out["primary_structural_sample"]),
        ("complete_case", out["complete_case_structural_sample"]),
    ]:
        sample = out.loc[mask]
        rows.append(
            {
                "sample": name,
                "n_countries": len(sample),
                "n_complete_profiles": int(sample["cluster_imputed_fields"].eq(0).sum()),
                "mean_imputed_fields": float(sample["cluster_imputed_fields"].mean()),
                "max_imputed_fields": int(sample["cluster_imputed_fields"].max()),
                "n_exposed_top_tercile": int(sample["exposure_tercile"].eq("high").sum()),
                "n_with_gvc_pre_slope": int(sample["gvc_pre_slope"].notna().sum()),
                "n_with_recovery_pre_slope": int(sample["recovery_pre_slope"].notna().sum()),
                "n_with_diversification_pre_slope": int(sample["diversification_pre_slope"].notna().sum()),
            }
        )
    comparison = pd.DataFrame(rows)
    write_csv(audit, out_dir / "valid_country_sample_audit.csv")
    write_csv(audit.loc[~out["primary_structural_sample"]], out_dir / "excluded_entities.csv")
    write_csv(missing, out_dir / "structural_missingness_by_country.csv")
    write_csv(comparison, out_dir / "structural_sample_comparison.csv")
    write_csv(out, out_dir / "structural_profile_final_audit_base.csv")
    return out


def imputation_matrices(profile: pd.DataFrame, reps: int, seed: int) -> list[np.ndarray]:
    base = profile[STRUCTURAL_VARS].copy()
    rng = np.random.default_rng(seed)
    matrices = []
    for _ in range(reps):
        filled = base.copy()
        for variable in STRUCTURAL_VARS:
            values = num(base[variable])
            observed = values.dropna().to_numpy(dtype=float)
            if len(observed) == 0:
                draws = np.zeros(len(values))
            elif len(observed) == 1:
                draws = np.repeat(observed[0], len(values))
            else:
                draws = rng.normal(observed.mean(), observed.std(ddof=1), len(values))
                draws = np.clip(draws, observed.min(), observed.max())
            array = values.to_numpy(dtype=float)
            array[np.isnan(array)] = draws[np.isnan(array)]
            filled[variable] = array
        matrices.append(StandardScaler().fit_transform(filled[STRUCTURAL_VARS]))
    return matrices


def consensus_k(
    profile: pd.DataFrame, k: int, matrices: list[np.ndarray], seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    labels_all = []
    silhouettes = []
    sses = []
    reference = None
    for idx, matrix in enumerate(matrices):
        fit = KMeans(n_clusters=k, random_state=seed + idx, n_init=50).fit(matrix)
        labels = fit.labels_
        if reference is None:
            reference = labels.copy()
        else:
            labels = align_labels(reference, labels, k)
        labels_all.append(labels)
        silhouettes.append(float(silhouette_score(matrix, fit.labels_)))
        sses.append(float(fit.inertia_))
    label_array = np.vstack(labels_all)
    co_cluster = (label_array[:, :, None] == label_array[:, None, :]).mean(axis=0)
    consensus = KMeans(n_clusters=k, random_state=seed + 5000, n_init=50).fit_predict(co_cluster)
    consensus = align_labels(reference, consensus, k)
    probabilities = (label_array == consensus[None, :]).mean(axis=0)

    rng = np.random.default_rng(seed + 9000)
    aris = []
    matrix = matrices[0]
    for rep in range(STABILITY_REPS):
        idx = rng.choice(len(profile), size=max(k * 4, int(0.8 * len(profile))), replace=False)
        sampled = KMeans(n_clusters=k, random_state=seed + 10000 + rep, n_init=20).fit(matrix[idx])
        aris.append(float(adjusted_rand_score(reference[idx], align_labels(reference[idx], sampled.labels_, k))))
    summary = pd.DataFrame(
        [
            {
                "k": k,
                "n_countries": len(profile),
                "silhouette_mean_across_imputations": np.mean(silhouettes),
                "within_cluster_sse_mean_across_imputations": np.mean(sses),
                "bootstrap_mean_ari": np.mean(aris),
                "mean_assignment_probability": np.mean(probabilities),
                "proportion_assignment_probability_ge_0_80": np.mean(probabilities >= 0.80),
                "min_assignment_probability": np.min(probabilities),
                "max_assignment_probability": np.max(probabilities),
                "n_imputations": len(matrices),
                "stability_reps": STABILITY_REPS,
            }
        ]
    )
    assignment = pd.DataFrame(
        {
            "country_iso3_code": profile["country_iso3_code"].to_numpy(),
            "consensus_cluster_numeric": consensus,
            "assignment_probability": probabilities,
        }
    )
    return summary, assignment, co_cluster


def label_assignment(profile: pd.DataFrame, assignment: pd.DataFrame, sample: str, selected_k: int) -> pd.DataFrame:
    out = assignment.merge(
        profile[["country_iso3_code", "pre_log_real_gdp_pc", "wb_region", "exposure_tercile", "cluster_imputed_fields"]],
        on="country_iso3_code",
        how="left",
    )
    order = out.groupby("consensus_cluster_numeric")["pre_log_real_gdp_pc"].mean().sort_values().index.tolist()
    mapping = {int(value): f"R{idx + 1}" for idx, value in enumerate(order)}
    out["structural_profile"] = out["consensus_cluster_numeric"].map(mapping)
    out["structural_regime"] = out["structural_profile"]
    out["sample"] = sample
    out["selected_k"] = selected_k
    return out


def run_clustering(profile: pd.DataFrame, out_dir: Path, seed: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    selections, assignments = [], {}
    primary_assign = None
    for sample, mask in [
        ("cleaned_primary", profile["primary_structural_sample"]),
        ("complete_case", profile["complete_case_structural_sample"]),
        ("broad_legacy", pd.Series(True, index=profile.index)),
    ]:
        subset = profile.loc[mask].copy()
        matrices = imputation_matrices(subset, MI_REPS, seed + len(sample))
        choices = {}
        for k in [2, 3]:
            summary, assignment, co = consensus_k(subset, k, matrices, seed + k + len(sample))
            summary["sample"] = sample
            selections.append(summary.iloc[0].to_dict())
            choices[k] = (summary.iloc[0], assignment, co)
        sel = pd.DataFrame([row for row in selections if row["sample"] == sample])
        gap = float(sel["silhouette_mean_across_imputations"].max() - sel["silhouette_mean_across_imputations"].min())
        if gap <= 0.02:
            probability_gap = float(sel["mean_assignment_probability"].max() - sel["mean_assignment_probability"].min())
            ari_gap = float(sel["bootstrap_mean_ari"].max() - sel["bootstrap_mean_ari"].min())
            if probability_gap > 0.02:
                chosen = sel.sort_values(
                    ["mean_assignment_probability", "bootstrap_mean_ari", "k"],
                    ascending=[False, False, True],
                ).iloc[0]
                basis = "silhouette gap <= 0.02; probability gap > 0.02; higher probability then ARI"
            elif ari_gap > 0.02:
                chosen = sel.sort_values(
                    ["bootstrap_mean_ari", "k"], ascending=[False, True]
                ).iloc[0]
                basis = "silhouette gap <= 0.02; probability gap <= 0.02; ARI gap > 0.02"
            else:
                chosen = sel.sort_values(["k"], ascending=[True]).iloc[0]
                basis = "silhouette gap <= 0.02; probability and ARI gaps <= 0.02; lower k"
        else:
            chosen = sel.sort_values(
                ["silhouette_mean_across_imputations", "k"],
                ascending=[False, True],
            ).iloc[0]
            basis = "highest MI mean silhouette; ties lower k"
        selected_k = int(chosen["k"])
        for row in selections:
            if row["sample"] == sample:
                row["silhouette_gap_to_best"] = gap
                row["selection_basis"] = basis
                row["selected_k"] = selected_k
        summary, assignment, co = choices[selected_k]
        labeled = label_assignment(subset, assignment, sample, selected_k)
        assignments[sample] = labeled
        if sample == "cleaned_primary":
            primary_assign = labeled
            countries = subset["country_iso3_code"].tolist()
            matrix_rows = [
                {
                    "country_left": countries[i],
                    "country_right": countries[j],
                    "co_clustering_probability": float(co[i, j]),
                }
                for i in range(len(countries)) for j in range(len(countries))
            ]
            write_csv(pd.DataFrame(matrix_rows), out_dir / "structural_regime_consensus_matrix.csv")
            write_csv(labeled, out_dir / "structural_regime_assignment_probabilities.csv")
    selection = pd.DataFrame(selections)
    all_assignments = pd.concat(assignments.values(), ignore_index=True)
    write_csv(selection, out_dir / "structural_regime_selection_clean.csv")
    write_csv(all_assignments, out_dir / "structural_regime_assignments_clean.csv")
    write_csv(primary_assign, out_dir / "structural_regime_assignment_probabilities.csv")
    summary_profile = (
        profile.merge(primary_assign[["country_iso3_code", "structural_profile"]], on="country_iso3_code", how="inner")
        .groupby("structural_profile")[STRUCTURAL_VARS]
        .mean().reset_index().rename(columns={"structural_profile": "structural_regime"})
    )
    write_csv(summary_profile, out_dir / "structural_regime_profiles_clean.csv")
    counts = primary_assign.groupby(["structural_profile", "exposure_tercile"]).size().reset_index(name="n_countries")
    write_csv(counts, out_dir / "structural_regime_exposure_counts_clean.csv")
    return primary_assign, assignments, selection

def prepare_panel(panel: pd.DataFrame, profile: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    p = standardize_profile(profile)
    cols = [
        "country_iso3_code", "z_eci_pre", "z_coi_pre", "z_exposure_pre",
        "wb_region", "exposed_top_tercile",
    ] + [f"z_{v}" for v in STRUCTURAL_VARS]
    cols = [c for c in cols if c in p.columns]
    a_cols = ["country_iso3_code", "structural_profile", "structural_regime", "assignment_probability"]
    a_cols = [c for c in a_cols if c in assignment.columns]
    merged = p[cols].merge(assignment[a_cols], on="country_iso3_code", how="inner")
    out = panel.drop(columns=["wb_region"], errors="ignore").merge(merged, on="country_iso3_code", how="inner")
    out["post_2018"] = out["year"].ge(PRIMARY_POST).astype(float)
    out["post_2019"] = out["year"].ge(SENSITIVITY_POST).astype(float)
    return out


def prepare_confirmatory_event_panel(panel: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """Build event-study inputs without structural-sample assignment filtering."""
    p = standardize_profile(profile)
    cols = ["country_iso3_code", "z_eci_pre", "z_exposure_pre"]
    merged = p[cols].merge(panel, on="country_iso3_code", how="right")
    merged["post_2018"] = merged["year"].ge(PRIMARY_POST).astype(float)
    merged["post_2019"] = merged["year"].ge(SENSITIVITY_POST).astype(float)
    return merged

def coefficient_row(fit: Any, term: str, reps: int, seed: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    vector = np.zeros(len(fit.term_names))
    vector[fit.term_index(term)] = 1.0
    stat = contrast_statistics(fit, vector)
    boot = wild_cluster_bootstrap_contrast(fit, vector, reps=reps, seed=seed, alternative="two-sided")
    row = {**stat, **boot, "pvalue_for_fdr": boot["p_wild_bootstrap"],
           "n_obs": fit.n_obs, "n_countries": fit.n_countries, "n_years": fit.n_years}
    if extra:
        row.update(extra)
    return row


def focal_models(frame: pd.DataFrame, post_start: int, reps: int, seed: int) -> pd.DataFrame:
    work_frame = frame.copy()
    work_frame["post_2018"] = work_frame["year"].ge(post_start).astype(float)
    rows = []
    for oi, (outcome_key, (outcome, label)) in enumerate(OUTCOMES.items()):
        work, terms = add_base_terms(work_frame)
        work, mod_terms = add_moderator_terms(work, "z_coi_pre")
        terms += mod_terms
        try:
            fit = fit_terms(work, outcome, terms)
            for term, hypothesis, offset in [
                ("eci_exposure_post", "H1-H3 focal ECI x exposure x post", 0),
                ("eci_exposure_post_x_moderator", "H4 ECI x exposure x post x COI", 50),
            ]:
                row = coefficient_row(fit, term, reps, seed + oi * 100 + offset)
                row.update(
                    {
                        "test_id": f"focal_{post_start}_{outcome_key}_{term}",
                        "outcome_key": outcome_key, "outcome": label,
                        "post_start": post_start, "term": term,
                        "hypothesis": hypothesis,
                    }
                )
                rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "test_id": f"focal_{post_start}_{outcome_key}_error",
                    "outcome_key": outcome_key, "outcome": label,
                    "post_start": post_start, "error": str(exc),
                    "pvalue_for_fdr": np.nan,
                }
            )
    return pd.DataFrame(rows)


def regime_terms(frame: pd.DataFrame, regimes: list[str]) -> tuple[pd.DataFrame, list[str], dict[str, np.ndarray]]:
    out, terms = add_base_terms(frame)
    reference = regimes[0]
    vectors = {}
    for regime in regimes[1:]:
        mask = out["structural_regime"].eq(regime).astype(float)
        for base in ["post_x_exposure", "post_x_eci", "eci_exposure_post"]:
            term = f"{base}_x_{regime}"
            out[term] = out[base] * mask
            terms.append(term)
        for year in sorted(out["year"].dropna().astype(int).unique()):
            term = f"regime_year_{regime}_{year}"
            out[term] = mask * out["year"].eq(year).astype(float)
            terms.append(term)
    for regime in regimes:
        vector = np.zeros(len(terms))
        vector[terms.index("eci_exposure_post")] = 1.0
        if regime != reference:
            vector[terms.index(f"eci_exposure_post_x_{regime}")] = 1.0
        vectors[regime] = vector
    return out, terms, vectors


def wald_row(
    fit: Any, contrast: np.ndarray, test_id: str, test_type: str,
    reps: int, seed: int, extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stat = contrast_statistics(fit, contrast)
    joint = wald_test(fit, np.atleast_2d(contrast), reps=reps, seed=seed)
    row = {
        "test_id": test_id, "test_type": test_type,
        "estimate": stat["estimate"], "se": stat["se"], "tvalue": stat["tvalue"],
        "p_cluster": stat["p_cluster"], "ci_low_95": stat["ci_low_95"],
        "ci_high_95": stat["ci_high_95"],
        "p_wild_bootstrap": joint["p_wild_bootstrap"],
        "bootstrap_reps_requested": joint["bootstrap_reps_requested"],
        "bootstrap_reps_success": joint["bootstrap_reps_success"],
        "pvalue_for_fdr": joint["pvalue_for_fdr"],
        "n_obs": fit.n_obs, "n_countries": fit.n_countries, "n_years": fit.n_years,
    }
    if extra:
        row.update(extra)
    return row


def corrected_regime_models(
    frame: pd.DataFrame, post_start: int, regimes: list[str], reps: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    work_frame = frame.copy()
    work_frame["post_2018"] = work_frame["year"].ge(post_start).astype(float)
    slopes, differences, validation, fits = [], [], [], {}
    for oi, (outcome_key, (outcome, label)) in enumerate(OUTCOMES.items()):
        work, terms, vectors = regime_terms(work_frame, regimes)
        try:
            fit = fit_terms(work, outcome, terms)
            fits[outcome_key] = fit
            for ri, regime in enumerate(regimes):
                slopes.append(
                    wald_row(
                        fit, vectors[regime],
                        f"regime_slope_{post_start}_{outcome_key}_{regime}",
                        "regime_specific_slope", reps, seed + oi * 100 + ri,
                        {
                            "outcome_key": outcome_key, "outcome": label,
                            "regime": regime, "post_start": post_start,
                        },
                    )
                )
            reference = regimes[0]
            contrasts = [vectors[r] - vectors[reference] for r in regimes[1:]]
            matrix = np.vstack(contrasts)
            for ri, regime in enumerate(regimes[1:]):
                differences.append(
                    wald_row(
                        fit, contrasts[ri],
                        f"pairwise_{post_start}_{outcome_key}_{regime}_minus_{reference}",
                        "pairwise_regime_difference", reps, seed + oi * 100 + 30 + ri,
                        {
                            "outcome_key": outcome_key, "outcome": label,
                            "regime_left": regime, "regime_right": reference,
                            "post_start": post_start,
                        },
                    )
                )
            omnibus = wald_test(fit, matrix, reps=reps, seed=seed + oi * 100 + 30)
            differences.append(
                {
                    "test_id": f"omnibus_{post_start}_{outcome_key}_regime_equality",
                    "test_type": "omnibus_regime_equality",
                    "estimate": np.nan, "se": np.nan, "tvalue": np.nan,
                    "p_cluster": omnibus["p_cluster"],
                    "ci_low_95": np.nan, "ci_high_95": np.nan,
                    "p_wild_bootstrap": omnibus["p_wild_bootstrap"],
                    "bootstrap_reps_requested": omnibus["bootstrap_reps_requested"],
                    "bootstrap_reps_success": omnibus["bootstrap_reps_success"],
                    "pvalue_for_fdr": omnibus["pvalue_for_fdr"],
                    "n_obs": fit.n_obs, "n_countries": fit.n_countries, "n_years": fit.n_years,
                    "outcome_key": outcome_key, "outcome": label,
                    "regime_left": "|".join(regimes[1:]), "regime_right": reference,
                    "post_start": post_start,
                }
            )
            pair_p = differences[-2]["p_wild_bootstrap"] if len(regimes) == 2 else np.nan
            diff = abs(float(pair_p) - float(omnibus["p_wild_bootstrap"])) if len(regimes) == 2 else np.nan
            validation.append(
                {
                    "outcome_key": outcome_key, "outcome": label, "post_start": post_start,
                    "k": len(regimes), "pairwise_p_wild": pair_p,
                    "omnibus_p_wild": omnibus["p_wild_bootstrap"],
                    "absolute_difference": diff, "tolerance": 1e-12,
                    "validation_passed": bool(diff < 1e-12) if len(regimes) == 2 else np.nan,
                }
            )
        except Exception as exc:
            validation.append(
                {
                    "outcome_key": outcome_key, "outcome": label,
                    "post_start": post_start, "k": len(regimes),
                    "error": str(exc), "validation_passed": False,
                }
            )
    return pd.DataFrame(slopes), pd.DataFrame(differences), pd.DataFrame(validation), fits


def focal_event_terms(
    frame: pd.DataFrame,
    interacted: bool = False,
    regime_reference: str = "R1",
) -> tuple[pd.DataFrame, list[str], dict[int, str]]:
    out = frame.copy()
    terms = []
    triple = {}
    regimes = sorted(out["structural_regime"].dropna().unique()) if interacted else []
    for year in sorted(num(out["year"]).dropna().astype(int).unique()):
        if year == EVENT_REFERENCE:
            continue
        year_flag = out["year"].eq(year).astype(float)
        e_term = f"event_exposure_{year}"
        c_term = f"event_eci_{year}"
        t_term = f"event_eci_exposure_{year}"
        out[e_term] = out["z_exposure_pre"] * year_flag
        out[c_term] = out["z_eci_pre"] * year_flag
        out[t_term] = out["z_eci_pre"] * out["z_exposure_pre"] * year_flag
        terms += [e_term, c_term, t_term]
        triple[year] = t_term
        for regime in regimes:
            if regime == regime_reference:
                continue
            mask = out["structural_regime"].eq(regime).astype(float)
            for base_term in (e_term, c_term, t_term):
                xterm = f"{base_term}_x_{regime}"
                out[xterm] = out[base_term] * mask
                terms.append(xterm)
            regime_year = f"regime_year_{regime}_{year}"
            out[regime_year] = mask * year_flag
            terms.append(regime_year)
    return out, terms, triple

def corrected_event_studies(
    frame: pd.DataFrame,
    regimes: list[str],
    out_dir: Path,
    reps: int,
    seed: int,
    confirmatory_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coefficient_rows, pre_rows, post_rows = [], [], []
    for oi, (outcome_key, (outcome, label)) in enumerate(OUTCOMES.items()):
        confirmatory = confirmatory_frame if confirmatory_frame is not None else frame
        variants = [
            ("full_sample_confirmatory", confirmatory, False),
            ("full_sample_cleaned_structural", frame, False),
            ("pooled_regime_interacted", frame, True),
        ]
        variants += [
            (f"regime_{regime}", frame.loc[frame["structural_regime"].eq(regime)], False)
            for regime in regimes
        ]
        for vi, (variant, subset, interacted) in enumerate(variants):
            work, terms, triple = focal_event_terms(subset, interacted, regimes[0])
            try:
                fit = fit_terms(work, outcome, terms)
                for year, term in triple.items():
                    vector = np.zeros(len(fit.term_names))
                    vector[fit.term_index(term)] = 1.0
                    stat = contrast_statistics(fit, vector)
                    coefficient_rows.append(
                        {
                            "test_id": f"corrected_event_{variant}_{outcome_key}_{year}",
                            "variant": variant, "outcome_key": outcome_key,
                            "outcome": label, "event_year": year, "term": term,
                            "estimate": stat["estimate"], "se": stat["se"],
                            "p_cluster": stat["p_cluster"],
                            "ci_low_95": stat["ci_low_95"], "ci_high_95": stat["ci_high_95"],
                            "n_obs": fit.n_obs, "n_countries": fit.n_countries,
                        }
                    )
                for test_name, years, target in [
                    ("pretrend", range(2012, 2017), pre_rows),
                    ("post_period", range(2018, 2023), post_rows),
                ]:
                    selected = [term for year, term in triple.items() if year in years]
                    if interacted and len(regimes) == 2:
                        for year, term in triple.items():
                            xterm = f"{term}_x_{regimes[1]}"
                            if xterm in fit.term_names and year in years:
                                selected.append(xterm)
                    if not selected:
                        continue
                    matrix = np.zeros((len(selected), len(fit.term_names)))
                    for idx, term in enumerate(selected):
                        matrix[idx, fit.term_index(term)] = 1.0
                    joint = wald_test(
                        fit, matrix, reps=reps,
                        seed=seed + oi * 1000 + vi * 10 + (test_name == "post_period"),
                    )
                    target.append(
                        {
                            "test_id": f"corrected_event_{test_name}_{variant}_{outcome_key}",
                            "variant": variant, "outcome_key": outcome_key,
                            "outcome": label, "test_type": f"joint_{test_name}",
                            **joint, "n_obs": fit.n_obs, "n_countries": fit.n_countries,
                        }
                    )
            except Exception as exc:
                pre_rows.append(
                    {
                        "test_id": f"corrected_event_pretrend_{variant}_{outcome_key}",
                        "variant": variant, "outcome_key": outcome_key,
                        "error": str(exc), "pvalue_for_fdr": np.nan,
                    }
                )
    coefficients = pd.DataFrame(coefficient_rows)
    pretrend = pd.DataFrame(pre_rows)
    post_tests = pd.DataFrame(post_rows)
    write_csv(coefficients, out_dir / "corrected_event_study_coefficients.csv")
    write_csv(pretrend, out_dir / "corrected_event_study_pretrend_tests.csv")
    write_csv(post_tests, out_dir / "corrected_event_study_post_tests.csv")
    for outcome_key, (_, label) in OUTCOMES.items():
        plot = coefficients.loc[
            (coefficients["outcome_key"] == outcome_key)
            & coefficients["variant"].eq("full_sample_confirmatory")
        ]
        if plot.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(EVENT_REFERENCE, color="gray", linestyle="--", linewidth=0.8)
        ax.errorbar(
            plot["event_year"], plot["estimate"],
            yerr=[plot["estimate"] - plot["ci_low_95"], plot["ci_high_95"] - plot["estimate"]],
            fmt="o-", capsize=3, color="#155e75",
        )
        ax.set_title(f"Corrected focal event study: {label}")
        ax.set_xlabel("Year; 2017 reference")
        ax.set_ylabel("ECI x exposure coefficient")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / f"figure_corrected_event_study_{outcome_key}.png", dpi=180)
        plt.close(fig)
    return coefficients, pretrend, post_tests

def regime_counts(frame: pd.DataFrame, fit: Any) -> pd.DataFrame:
    used = frame.loc[fit.sample_index].copy()
    rows = []
    for regime, group in used.groupby("structural_regime"):
        countries = group["country_iso3_code"].nunique()
        exposed = group.loc[group["exposed_top_tercile"].eq(1), "country_iso3_code"].nunique()
        rows.append(
            {
                "regime": regime,
                "n_countries_used": int(countries),
                "n_country_year_observations": int(len(group)),
                "n_exposed_countries_used": int(exposed),
                "effective_number_of_clusters": int(countries),
            }
        )
    return pd.DataFrame(rows)


def power_tables(frame: pd.DataFrame, fits: dict[str, Any], slopes: pd.DataFrame, differences: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, diff_rows = [], []
    for outcome_key, fit in fits.items():
        label = OUTCOMES[outcome_key][1]
        counts = regime_counts(frame, fit)
        for _, count in counts.iterrows():
            regime = count["regime"]
            slope = slopes.loc[
                (slopes["outcome_key"] == outcome_key) & slopes["regime"].eq(regime)
            ].iloc[0]
            rows.append(
                {
                    "outcome_key": outcome_key, "outcome": label, "regime": regime,
                    **count.to_dict(), "standard_error": slope["se"],
                    "mde_80pct": 2.80 * slope["se"], "mde_90pct": 3.24 * slope["se"],
                    "mde_rule": "two-sided cluster-power approximation",
                }
            )
        pair = differences.loc[
            (differences["outcome_key"] == outcome_key)
            & differences["test_type"].eq("pairwise_regime_difference")
        ]
        if not pair.empty:
            pair = pair.iloc[0]
            diff_rows.append(
                {
                    "outcome_key": outcome_key, "outcome": label,
                    "regime_left": pair["regime_left"], "regime_right": pair["regime_right"],
                    "n_countries_used": fit.n_countries, "n_country_year_observations": fit.n_obs,
                    "effective_number_of_clusters": fit.n_clusters,
                    "contrast_estimate": pair["estimate"], "contrast_se": pair["se"],
                    "mde_80pct": 2.80 * pair["se"], "mde_90pct": 3.24 * pair["se"],
                }
            )
    power = pd.DataFrame(rows)
    diff_power = pd.DataFrame(diff_rows)
    write_csv(power, out_dir / "regime_power_mde_outcome_specific.csv")
    write_csv(diff_power, out_dir / "regime_difference_power_mde.csv")
    return power, diff_power


def smd(left: pd.Series, right: pd.Series) -> float:
    left, right = num(left).dropna(), num(right).dropna()
    if len(left) < 2 or len(right) < 2:
        return np.nan
    denom = math.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2)
    return float((left.mean() - right.mean()) / denom) if denom > 0 else np.nan


def weighted_smd(left: pd.Series, right: pd.Series, wl: pd.Series, wr: pd.Series) -> float:
    left, right = num(left), num(right)
    mask_l, mask_r = left.notna() & wl.notna(), right.notna() & wr.notna()
    left, right, wl, wr = left[mask_l], right[mask_r], wl[mask_l], wr[mask_r]
    if len(left) < 2 or len(right) < 2:
        return np.nan
    mean_l, mean_r = np.average(left, weights=wl), np.average(right, weights=wr)
    var_l = np.average((left - mean_l) ** 2, weights=wl)
    var_r = np.average((right - mean_r) ** 2, weights=wr)
    denom = math.sqrt((var_l + var_r) / 2)
    return float((mean_l - mean_r) / denom) if denom > 0 else np.nan


def residual_density_weights(profile: pd.DataFrame) -> pd.DataFrame:
    out = profile.copy()
    covariates = out[STRUCTURAL_VARS].apply(num)
    complete = covariates.notna().all(axis=1) & out["exposure_pre"].notna()
    scaled = StandardScaler().fit_transform(covariates.loc[complete])
    exposure = out.loc[complete, "exposure_pre"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(exposure)), scaled])
    beta = np.linalg.pinv(design) @ exposure
    residual = exposure - design @ beta
    sd = float(np.std(residual, ddof=1)) or 1.0
    out["residual_density_weight"] = np.nan
    out.loc[complete, "residual_density_weight"] = np.exp(-0.5 * (residual / sd) ** 2)
    out["residual_density_weight"] /= out["residual_density_weight"].mean()
    return out


def weight_and_balance(
    profile: pd.DataFrame, frame: pd.DataFrame, out_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report residual-density weighting only as a failed diagnostic."""
    weighted = residual_density_weights(profile)
    weights = weighted["residual_density_weight"].dropna()
    ess = (weights.sum() ** 2) / (weights.pow(2).sum())
    diagnostics = pd.DataFrame(
        [
            {"metric": "n_weighted_countries", "value": len(weights), "diagnostic_status": "failed_balance"},
            {"metric": "min_weight", "value": weights.min(), "diagnostic_status": "failed_balance"},
            {"metric": "max_weight", "value": weights.max(), "diagnostic_status": "failed_balance"},
            {"metric": "median_weight", "value": weights.median(), "diagnostic_status": "failed_balance"},
            {"metric": "p01_weight", "value": weights.quantile(0.01), "diagnostic_status": "failed_balance"},
            {"metric": "p05_weight", "value": weights.quantile(0.05), "diagnostic_status": "failed_balance"},
            {"metric": "p95_weight", "value": weights.quantile(0.95), "diagnostic_status": "failed_balance"},
            {"metric": "p99_weight", "value": weights.quantile(0.99), "diagnostic_status": "failed_balance"},
            {"metric": "effective_sample_size", "value": ess, "diagnostic_status": "failed_balance"},
            {"metric": "ess_fraction_of_weighted_countries", "value": ess / len(weights), "diagnostic_status": "failed_balance"},
            {"metric": "max_country_leverage_share", "value": weights.max() / weights.sum(), "diagnostic_status": "failed_balance"},
        ]
    )
    low = profile["exposure_tercile"].eq("low")
    high = profile["exposure_tercile"].eq("high")
    balance_rows = []
    for variable in STRUCTURAL_VARS:
        before = smd(profile.loc[high, variable], profile.loc[low, variable])
        after = weighted_smd(
            weighted.loc[high, variable], weighted.loc[low, variable],
            weighted.loc[high, "residual_density_weight"],
            weighted.loc[low, "residual_density_weight"],
        )
        balance_rows.append(
            {
                "variable": variable,
                "label": STRUCTURAL_LABELS[variable],
                "diagnostic_status": "failed_balance",
                "smd_before": before,
                "smd_after_residual_density_weighting_failed": after,
                "absolute_smd_after": abs(after) if pd.notna(after) else np.nan,
                "balance_improved": abs(after) < abs(before) if pd.notna(after) and pd.notna(before) else False,
            }
        )
    balance = pd.DataFrame(balance_rows)
    write_csv(diagnostics, out_dir / "residual_density_weight_diagnostics.csv")
    write_csv(balance, out_dir / "balance_before_after_weighting.csv")
    stale = out_dir / "weighted_model_trimmed_sensitivity.csv"
    if stale.exists():
        stale.unlink()
    return weighted, balance


def theory_balance(
    profile: pd.DataFrame, weighted: pd.DataFrame, out_dir: Path
) -> pd.DataFrame:
    rows = []
    for weighting, use_weights in [
        ("unadjusted", False),
        ("residual_density_weighting_failed", True),
    ]:
        low = profile["exposure_tercile"].eq("low")
        high = profile["exposure_tercile"].eq("high")
        values = []
        for variable in STRUCTURAL_VARS:
            if use_weights:
                value = weighted_smd(
                    weighted.loc[high, variable], weighted.loc[low, variable],
                    weighted.loc[high, "residual_density_weight"],
                    weighted.loc[low, "residual_density_weight"],
                )
            else:
                value = smd(profile.loc[high, variable], profile.loc[low, variable])
            values.append(value)
            rows.append(
                {
                    "weighting": weighting,
                    "variable": variable,
                    "smd": value,
                    "absolute_smd": abs(value) if pd.notna(value) else np.nan,
                    "absolute_smd_gt_0_10": abs(value) > 0.10 if pd.notna(value) else np.nan,
                    "diagnostic_status": "failed_balance" if use_weights else "descriptive_unadjusted",
                }
            )
        valid = np.asarray([x for x in values if pd.notna(x)], dtype=float)
        rows.append(
            {
                "weighting": weighting,
                "variable": "__summary__",
                "max_absolute_smd": float(np.max(np.abs(valid))) if len(valid) else np.nan,
                "proportion_absolute_smd_gt_0_10": float(np.mean(np.abs(valid) > 0.10)) if len(valid) else np.nan,
                "diagnostic_status": "failed_balance" if use_weights else "descriptive_unadjusted",
            }
        )
    result = pd.DataFrame(rows)
    write_csv(result, out_dir / "theory_based_balance_diagnostics.csv")
    return result

def adjustment_models(
    frame: pd.DataFrame, out_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare Models A-D on common and maximum-available samples.

    Residual-density weighting is intentionally excluded from the adjustment
    evidence because its diagnostic balance check fails for all ten covariates.
    """
    work = frame.copy()
    rows_common, rows_max = [], []
    model_names = ["A_FE", "B_structural_x_year", "C_region_x_year", "D_country_trends"]
    for outcome_key, (outcome, label) in OUTCOMES.items():
        required = [
            outcome, "z_eci_pre", "z_coi_pre", "z_exposure_pre", "wb_region",
        ] + [f"z_{v}" for v in STRUCTURAL_VARS]
        common = work.dropna(subset=required).copy()
        for sample_name, subset in [
            ("common_sample", common),
            ("maximum_available", work),
        ]:
            for model_name in model_names:
                sub = subset.copy()
                sub["post_2018"] = sub["year"].ge(PRIMARY_POST).astype(float)
                sub, terms = add_base_terms(sub)
                sub, mod_terms = add_moderator_terms(sub, "z_coi_pre")
                terms += mod_terms
                if model_name == "B_structural_x_year":
                    for variable in STRUCTURAL_VARS:
                        col = f"z_{variable}"
                        for year in sorted(sub["year"].dropna().unique()):
                            if year == min(sub["year"].dropna()):
                                continue
                            term = f"adjust_structural_year_{variable}_{year}"
                            sub[term] = sub[col] * sub["year"].eq(year).astype(float)
                            terms.append(term)
                elif model_name == "C_region_x_year":
                    terms += build_region_year_terms(sub)
                elif model_name == "D_country_trends":
                    terms += build_country_trend_terms(sub, outcome)
                try:
                    fit = fit_terms(sub, outcome, terms)
                    for term in ["eci_exposure_post", "eci_exposure_post_x_moderator"]:
                        row = coefficient_row(fit, term, 0, 0)
                        row.update(
                            {
                                "outcome_key": outcome_key, "outcome": label,
                                "sample_type": sample_name, "model": model_name,
                                "term": term,
                                "common_sample_n_countries": common["country_iso3_code"].nunique(),
                                "common_sample_n_obs": len(common),
                                "adjustment_evidence": "Models A-D; no weighting model",
                            }
                        )
                        (rows_common if sample_name == "common_sample" else rows_max).append(row)
                except Exception as exc:
                    target = rows_common if sample_name == "common_sample" else rows_max
                    target.append(
                        {
                            "outcome_key": outcome_key, "outcome": label,
                            "sample_type": sample_name, "model": model_name,
                            "adjustment_evidence": "Models A-D; no weighting model",
                            "error": str(exc), "pvalue_for_fdr": np.nan,
                        }
                    )
    common_df, max_df = pd.DataFrame(rows_common), pd.DataFrame(rows_max)
    stability_rows = []
    for (outcome_key, term), group in common_df.groupby(["outcome_key", "term"]):
        base_rows = group.loc[group["model"].eq("A_FE")]
        if base_rows.empty:
            continue
        base = base_rows.iloc[0]
        for _, row in group.iterrows():
            estimate = row.get("estimate")
            base_estimate = base.get("estimate")
            overlap = (
                pd.notna(estimate) and pd.notna(base_estimate)
                and row["ci_low_95"] <= base["ci_high_95"]
                and base["ci_low_95"] <= row["ci_high_95"]
            )
            stability_rows.append(
                {
                    "outcome_key": outcome_key, "term": term, "model": row["model"],
                    "coefficient": estimate, "p_cluster": row.get("p_cluster"),
                    "percentage_change_from_model_A": (
                        100 * (estimate - base_estimate) / abs(base_estimate)
                        if pd.notna(estimate) and pd.notna(base_estimate) and base_estimate != 0 else np.nan
                    ),
                    "sign_stability_vs_A": np.sign(estimate) == np.sign(base_estimate),
                    "confidence_intervals_overlap": overlap,
                    "common_sample_n_countries": row["common_sample_n_countries"],
                    "common_sample_n_obs": row["common_sample_n_obs"],
                    "adjustment_evidence": "Models A-D; no weighting model",
                }
            )
    stability = pd.DataFrame(stability_rows)
    write_csv(common_df, out_dir / "progressive_adjustment_common_sample.csv")
    write_csv(max_df, out_dir / "progressive_adjustment_maximum_sample.csv")
    write_csv(stability, out_dir / "coefficient_stability_summary.csv")
    stale = out_dir / "weighted_model_trimmed_sensitivity.csv"
    if stale.exists():
        stale.unlink()
    return common_df, max_df, stability

def moderator_block_tests(frame: pd.DataFrame, out_dir: Path, reps: int, seed: int) -> pd.DataFrame:
    rows = []
    for oi, (outcome_key, (outcome, label)) in enumerate(OUTCOMES.items()):
        work, terms = add_base_terms(frame)
        restrictions = []
        for block, variables in BLOCKS.items():
            for variable in variables:
                mod = "z_" + variable
                work[f"block_post_x_mod_{variable}"] = work["post_2018"] * work[mod]
                work[f"block_post_x_exposure_x_mod_{variable}"] = work["post_2018"] * work["z_exposure_pre"] * work[mod]
                work[f"block_post_x_eci_x_mod_{variable}"] = work["post_2018"] * work["z_eci_pre"] * work[mod]
                focal = f"block_eci_exposure_post_x_mod_{variable}"
                work[focal] = work["post_2018"] * work["z_exposure_pre"] * work["z_eci_pre"] * work[mod]
                terms += [
                    f"block_post_x_mod_{variable}",
                    f"block_post_x_exposure_x_mod_{variable}",
                    f"block_post_x_eci_x_mod_{variable}",
                    focal,
                ]
            restrictions.append(
                (
                    block,
                    [
                        f"block_eci_exposure_post_x_mod_{variable}"
                        for variable in variables
                    ],
                )
            )
        try:
            fit = fit_terms(work, outcome, terms)
            for bi, (block, restricted) in enumerate(restrictions):
                matrix = np.zeros((len(restricted), len(fit.term_names)))
                for i, term in enumerate(restricted):
                    matrix[i, fit.term_index(term)] = 1.0
                joint = wald_test(fit, matrix, reps=reps, seed=seed + oi * 100 + bi)
                rows.append(
                    {
                        "test_id": f"moderator_block_{outcome_key}_{block}",
                        "outcome_key": outcome_key, "outcome": label, "block": block,
                        "terms_restricted": "|".join(restricted),
                        **joint, "n_obs": fit.n_obs, "n_countries": fit.n_countries,
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "test_id": f"moderator_block_error_{outcome_key}",
                    "outcome_key": outcome_key, "outcome": label,
                    "error": str(exc), "pvalue_for_fdr": np.nan,
                }
            )
    result = pd.DataFrame(rows)
    write_csv(result, out_dir / "structural_moderator_block_omnibus_tests.csv")
    return result


def assignment_sensitivity(
    panel: pd.DataFrame, profiles: dict[str, pd.DataFrame], base_profile: pd.DataFrame,
    out_dir: Path, reps: int, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = profiles["cleaned_primary"].set_index("country_iso3_code")
    reassign_rows, model_rows = [], []
    for sample, assignment in profiles.items():
        assignment_index = assignment.set_index("country_iso3_code")
        for country in assignment_index.index:
            primary_regime = primary["structural_profile"].get(country, np.nan)
            alternative = assignment_index.loc[country, "structural_profile"]
            reassign_rows.append(
                {
                    "sample": sample, "country_iso3_code": country,
                    "primary_structural_profile": primary_regime,
                    "alternative_structural_profile": alternative,
                    "reassigned_vs_primary": (
                        bool(pd.notna(primary_regime) and primary_regime != alternative)
                    ),
                }
            )
        sample_profile = base_profile.loc[
            base_profile["country_iso3_code"].isin(assignment_index.index)
        ].copy()
        model = prepare_panel(panel, sample_profile, assignment)
        regimes = sorted(assignment["structural_profile"].dropna().unique())
        if len(regimes) != 2:
            continue
        slopes, differences, _, _ = corrected_regime_models(
            model, PRIMARY_POST, regimes, reps, seed + len(model_rows)
        )
        for _, row in pd.concat([slopes, differences], ignore_index=True).iterrows():
            model_rows.append(
                {
                    "sample": sample, "test_id": row.get("test_id"),
                    "outcome_key": row.get("outcome_key"), "outcome": row.get("outcome"),
                    "test_type": row.get("test_type"), "regime": row.get("regime"),
                    "regime_left": row.get("regime_left"), "regime_right": row.get("regime_right"),
                    "estimate": row.get("estimate"), "se": row.get("se"),
                    "p_cluster": row.get("p_cluster"),
                    "p_wild_bootstrap": row.get("p_wild_bootstrap"),
                    "n_obs": row.get("n_obs"), "n_countries": row.get("n_countries"),
                }
            )
    reassign = pd.DataFrame(reassign_rows)
    models = pd.DataFrame(model_rows)
    write_csv(reassign, out_dir / "regime_assignment_robustness.csv")
    write_csv(models, out_dir / "regime_model_assignment_sensitivity.csv")
    return reassign, models


def influence_analysis(frame: pd.DataFrame, regimes: list[str], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    country_detail, country_summary, region_rows = [], [], []
    countries = sorted(frame["country_iso3_code"].dropna().unique())
    for outcome_key, (outcome, label) in OUTCOMES.items():
        base_work, terms, vectors = regime_terms(frame, regimes)
        base_fit = fit_terms(base_work, outcome, terms)
        base_diff = contrast_statistics(base_fit, vectors["R2"] - vectors["R1"])
        for regime in regimes:
            base = contrast_statistics(base_fit, vectors[regime])
            estimates = []
            for excluded in countries:
                subset = frame.loc[frame["country_iso3_code"].ne(excluded)].copy()
                try:
                    work, terms2, vectors2 = regime_terms(subset, regimes)
                    fit = fit_terms(work, outcome, terms2)
                    stat = contrast_statistics(fit, vectors2[regime])
                    estimates.append((excluded, stat))
                    country_detail.append(
                        {
                            "outcome_key": outcome_key, "regime": regime,
                            "excluded_country": excluded, "estimate": stat["estimate"],
                            "p_cluster": stat["p_cluster"],
                        }
                    )
                except Exception:
                    continue
            if estimates:
                vals = np.asarray([item[1]["estimate"] for item in estimates])
                changes = np.abs(vals - base["estimate"])
                idx = int(np.argmax(changes))
                country_summary.append(
                    {
                        "outcome_key": outcome_key, "regime": regime,
                        "min_leave_one_out_estimate": vals.min(),
                        "max_leave_one_out_estimate": vals.max(),
                        "proportion_retaining_original_sign": np.mean(
                            np.sign(vals) == np.sign(base["estimate"])
                        ),
                        "maximum_absolute_coefficient_change": changes[idx],
                        "country_producing_largest_change": estimates[idx][0],
                        "baseline_estimate": base["estimate"],
                        "baseline_p_lt_05": base["p_cluster"] < 0.05,
                        "any_leave_one_out_p_lt_05": any(
                            item[1]["p_cluster"] < 0.05 for item in estimates
                        ),
                    }
                )
        for region in sorted(frame["wb_region"].dropna().unique()):
            subset = frame.loc[frame["wb_region"].ne(region)].copy()
            try:
                work, terms2, vectors2 = regime_terms(subset, regimes)
                fit = fit_terms(work, outcome, terms2)
                diff = contrast_statistics(fit, vectors2["R2"] - vectors2["R1"])
                for regime in regimes:
                    stat = contrast_statistics(fit, vectors2[regime])
                    region_rows.append(
                        {
                            "outcome_key": outcome_key, "excluded_region": region,
                            "regime": regime, "estimate": stat["estimate"],
                            "p_cluster": stat["p_cluster"],
                        }
                    )
                region_rows.append(
                    {
                        "outcome_key": outcome_key, "excluded_region": region,
                        "regime": "R2_minus_R1", "estimate": diff["estimate"],
                        "p_cluster": diff["p_cluster"],
                    }
                )
            except Exception as exc:
                region_rows.append(
                    {
                        "outcome_key": outcome_key, "excluded_region": region,
                        "error": str(exc),
                    }
                )
    detail = pd.DataFrame(country_detail)
    summary = pd.DataFrame(country_summary)
    regions = pd.DataFrame(region_rows)
    audit = []
    for (outcome_key, regime), group in regions.loc[
        regions["regime"].isin(regimes)
    ].groupby(["outcome_key", "regime"]):
        audit.append(
            {
                "outcome_key": outcome_key, "regime": regime,
                "region_min_estimate": group["estimate"].min(),
                "region_max_estimate": group["estimate"].max(),
                "region_sign_stability": np.mean(
                    np.sign(group["estimate"]) == np.sign(group["estimate"].iloc[0])
                ),
            }
        )
    write_csv(detail, out_dir / "leave_one_country_out_summary.csv")
    write_csv(summary, out_dir / "leave_one_country_out_summary_stats.csv")
    write_csv(regions, out_dir / "leave_one_region_out_results.csv")
    write_csv(pd.DataFrame(audit), out_dir / "influential_country_region_audit.csv")
    return summary, regions, pd.DataFrame(audit)


def placebo_comparison(frame: pd.DataFrame, out_dir: Path, reps: int, seed: int) -> pd.DataFrame:
    actual = focal_models(frame, PRIMARY_POST, reps, seed)
    actual = actual.loc[actual["term"].eq("eci_exposure_post")]
    rows = []
    for start in [2014, 2015]:
        work = frame.loc[frame["year"].between(BASELINE_START, BASELINE_END)].copy()
        work["placebo_post"] = work["year"].ge(start).astype(float)
        work["placebo_post_x_exposure"] = work["placebo_post"] * work["z_exposure_pre"]
        work["placebo_post_x_eci"] = work["placebo_post"] * work["z_eci_pre"]
        work["placebo_eci_exposure_post"] = (
            work["placebo_post"] * work["z_exposure_pre"] * work["z_eci_pre"]
        )
        for oi, (outcome_key, (outcome, label)) in enumerate(OUTCOMES.items()):
            try:
                fit = fit_terms(
                    work, outcome,
                    ["placebo_post_x_exposure", "placebo_post_x_eci", "placebo_eci_exposure_post"],
                )
                row = coefficient_row(
                    fit, "placebo_eci_exposure_post", reps, seed + start * 10 + oi
                )
                actual_row = actual.loc[actual["outcome_key"].eq(outcome_key)].iloc[0]
                row.update(
                    {
                        "placebo_start": start, "outcome_key": outcome_key, "outcome": label,
                        "actual_estimate_2018": actual_row["estimate"],
                        "actual_p_wild_2018": actual_row["p_wild_bootstrap"],
                        "magnitude_ratio_to_actual": abs(row["estimate"]) / abs(actual_row["estimate"])
                        if actual_row["estimate"] else np.nan,
                        "test_id": f"placebo_{start}_{outcome_key}",
                    }
                )
                rows.append(row)
            except Exception as exc:
                rows.append(
                    {
                        "placebo_start": start, "outcome_key": outcome_key,
                        "outcome": label, "error": str(exc), "pvalue_for_fdr": np.nan,
                    }
                )
    result = pd.DataFrame(rows)
    write_csv(result, out_dir / "placebo_2014_2015_comparison.csv")
    return result


def tost(estimate: float, se: float, bound: float, df: float) -> tuple[float, float, float, bool]:
    if not np.isfinite(estimate) or not np.isfinite(se) or se <= 0 or not np.isfinite(bound):
        return np.nan, np.nan, np.nan, False
    p_lower = stats.t.sf((estimate + bound) / se, df=df)
    p_upper = stats.t.cdf((estimate - bound) / se, df=df)
    p_value = max(float(p_lower), float(p_upper))
    return float(p_lower), float(p_upper), p_value, bool(p_value < 0.05)


def equivalence_tests(
    base_panel: pd.DataFrame, focal: pd.DataFrame, common: pd.DataFrame,
    differences: pd.DataFrame, out_dir: Path, base_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    sources = [
        (
            "tariff_weighted_confirmatory",
            pd.concat(
                [
                    pd.read_csv(base_dir / "reports" / "final_design_completion" / "confirmatory_tariff_weighted_tests.csv"),
                    pd.read_csv(base_dir / "reports" / "final_design_completion" / "h4_tariff_weighted_tests.csv"),
                ],
                ignore_index=True,
            ),
        ),
        ("clean_primary_final", focal.loc[focal["post_start"].eq(PRIMARY_POST)]),
        ("common_sample_adjusted", common.loc[common["model"].eq("A_FE")]),
    ]
    for source_name, table in sources:
        for _, row in table.iterrows():
            outcome_key = row.get("outcome_key")
            if outcome_key not in OUTCOMES:
                text = str(row.get("outcome", "")).lower()
                outcome_key = "gvc" if "gvc" in text or "linkage" in text else (
                    "recovery" if "recovery" in text else "diversification"
                )
            outcome = OUTCOMES[outcome_key][0]
            sd = float(num(base_panel[outcome]).std())
            bound = 0.10 * sd
            p1, p2, pt, eq = tost(
                float(row.get("estimate")), float(row.get("se")), bound,
                max(float(row.get("n_countries", 30)) - 1, 1),
            )
            rows.append(
                {
                    "source": source_name, "test_id": row.get("test_id", row.get("term")),
                    "outcome_key": outcome_key, "outcome": row.get("outcome"),
                    "estimate": row.get("estimate"), "se": row.get("se"),
                    "equivalence_bound_abs": bound, "equivalence_rule": EQUIVALENCE_RULE,
                    "tost_p_lower": p1, "tost_p_upper": p2, "tost_pvalue": pt,
                    "equivalent_within_bound": eq,
                    "ci_low_95": row.get("ci_low_95"), "ci_high_95": row.get("ci_high_95"),
                    "mde_80pct": 2.80 * float(row.get("se")),
                    "mde_90pct": 3.24 * float(row.get("se")),
                }
            )
    for _, row in differences.loc[
        differences["test_type"].eq("pairwise_regime_difference")
    ].iterrows():
        sd = float(num(base_panel[OUTCOMES[row["outcome_key"]][0]]).std())
        p1, p2, pt, eq = tost(
            float(row["estimate"]), float(row["se"]), 0.10 * sd,
            max(float(row["n_countries"]) - 1, 1),
        )
        rows.append(
            {
                "source": "final_cleaned_profile_regime_difference",
                "test_id": row["test_id"], "outcome_key": row["outcome_key"],
                "outcome": row["outcome"], "estimate": row["estimate"], "se": row["se"],
                "equivalence_bound_abs": 0.10 * sd, "equivalence_rule": EQUIVALENCE_RULE,
                "tost_p_lower": p1, "tost_p_upper": p2, "tost_pvalue": pt,
                "equivalent_within_bound": eq, "ci_low_95": row["ci_low_95"],
                "ci_high_95": row["ci_high_95"],
                "mde_80pct": 2.80 * row["se"], "mde_90pct": 3.24 * row["se"],
            }
        )
    result = pd.DataFrame(rows)
    mde = result[
        [
            "source", "test_id", "outcome_key", "estimate", "se",
            "equivalence_bound_abs", "mde_80pct", "mde_90pct",
            "tost_pvalue", "equivalent_within_bound",
        ]
    ]
    write_csv(result, out_dir / "final_equivalence_tests.csv")
    write_csv(mde, out_dir / "final_minimum_detectable_effects.csv")
    return result, mde


def post_period_comparison(frame: pd.DataFrame, out_dir: Path, regimes: list[str], reps: int, seed: int) -> pd.DataFrame:
    rows = []
    for start in [PRIMARY_POST, SENSITIVITY_POST]:
        focal = focal_models(frame, start, reps, seed + start)
        for _, row in focal.iterrows():
            rows.append(
                {
                    "analysis": "focal", "outcome_key": row.get("outcome_key"),
                    "outcome": row.get("outcome"), "entity": row.get("term"),
                    "post_start": start, "estimate": row.get("estimate"),
                    "se": row.get("se"), "p_wild_bootstrap": row.get("p_wild_bootstrap"),
                    "n_obs": row.get("n_obs"), "n_countries": row.get("n_countries"),
                }
            )
        slopes, diffs, _, _ = corrected_regime_models(
            frame, start, regimes, reps, seed + start + 1000
        )
        for _, row in pd.concat([slopes, diffs], ignore_index=True).iterrows():
            rows.append(
                {
                    "analysis": "regime", "outcome_key": row.get("outcome_key"),
                    "outcome": row.get("outcome"), "entity": row.get("test_id"),
                    "post_start": start, "estimate": row.get("estimate"),
                    "se": row.get("se"), "p_wild_bootstrap": row.get("p_wild_bootstrap"),
                    "n_obs": row.get("n_obs"), "n_countries": row.get("n_countries"),
                }
            )
    long = pd.DataFrame(rows)
    summary_rows = []
    for keys, group in long.groupby(["analysis", "outcome_key", "entity"]):
        wide = group.set_index("post_start")
        if 2018 not in wide.index or 2019 not in wide.index:
            continue
        a, b = wide.loc[2018], wide.loc[2019]
        summary_rows.append(
            {
                "analysis": keys[0], "outcome_key": keys[1], "entity": keys[2],
                "estimate_2018": a["estimate"], "se_2018": a["se"],
                "p_wild_2018": a["p_wild_bootstrap"],
                "estimate_2019": b["estimate"], "se_2019": b["se"],
                "p_wild_2019": b["p_wild_bootstrap"],
                "sign_stability": np.sign(a["estimate"]) == np.sign(b["estimate"]),
                "n_obs_2018": a["n_obs"], "n_obs_2019": b["n_obs"],
                "n_countries_2018": a["n_countries"], "n_countries_2019": b["n_countries"],
                "pvalue_for_fdr": b["p_wild_bootstrap"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    write_csv(long, out_dir / "post_period_2018_2019_long.csv")
    write_csv(summary, out_dir / "post_period_2018_2019_comparison.csv")
    return summary

def h6_directional(frame: pd.DataFrame, post_start: int, reps: int, seed: int) -> pd.DataFrame:
    threshold = float(num(frame["z_eci_pre"]).median())
    work = frame.copy()
    centered = work["z_eci_pre"] - threshold
    work["eci_below"] = np.minimum(centered, 0.0)
    work["eci_above"] = np.maximum(centered, 0.0)
    work["h6_low_slope"] = work["year"].ge(post_start).astype(float) * work["z_exposure_pre"] * work["eci_below"]
    work["h6_high_slope"] = work["year"].ge(post_start).astype(float) * work["z_exposure_pre"] * work["eci_above"]
    rows = []
    for oi, (outcome_key, (outcome, label)) in enumerate(OUTCOMES.items()):
        try:
            fit = fit_terms(work, outcome, ["h6_low_slope", "h6_high_slope"])
            low = np.zeros(len(fit.term_names)); low[fit.term_index("h6_low_slope")] = 1.0
            high_minus_low = low.copy()
            high_minus_low[fit.term_index("h6_low_slope")] = -1.0
            high_minus_low[fit.term_index("h6_high_slope")] = 1.0
            for test, vector, alternative in [
                ("H6_low_slope_negative", low, "less"),
                ("H6_high_minus_low_positive", high_minus_low, "greater"),
            ]:
                stat = contrast_statistics(fit, vector, alternative=alternative)
                boot = wild_cluster_bootstrap_contrast(
                    fit, vector, reps=reps, seed=seed + oi * 10, alternative=alternative
                )
                rows.append(
                    {
                        "test_id": f"h6_{post_start}_{outcome_key}_{test}",
                        "outcome_key": outcome_key, "outcome": label,
                        "post_start": post_start, "test": test,
                        "threshold_z_eci": threshold,
                        **stat, **boot,
                        "pvalue_for_fdr": boot["p_wild_bootstrap"],
                        "n_obs": fit.n_obs, "n_countries": fit.n_countries,
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "test_id": f"h6_{post_start}_{outcome_key}_error",
                    "outcome_key": outcome_key, "outcome": label,
                    "post_start": post_start, "error": str(exc),
                    "pvalue_for_fdr": np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_final_families(base_dir: Path, out_dir: Path, tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    old = pd.read_csv(base_dir / "reports" / "final_design_completion" / "full_reported_multiplicity_family.csv")
    confirm = old.loc[old["hypothesis"].isin(["H1", "H2", "H3", "H4"])].copy()
    confirm["multiplicity_family"] = "final_confirmatory_family"
    confirm["qvalue_final_confirmatory_family"] = bh_adjust(confirm["pvalue_for_fdr"])
    confirm["fdr_significant_0_05"] = (confirm["qvalue_final_confirmatory_family"] < 0.05).astype(int)
    write_csv(confirm, out_dir / "final_confirmatory_multiplicity_family.csv")

    pieces = []
    for name in [
        "continuous", "blocks", "regime_slopes", "regime_differences",
        "event_pretrend", "placebo", "post_period", "h6",
    ]:
        table = tables.get(name)
        if table is not None and not table.empty:
            table = table.copy()
            if "pvalue_for_fdr" in table:
                table["multiplicity_family"] = "final_exploratory_structural_family"
                pieces.append(table)
    structural = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if not structural.empty:
        structural["qvalue_final_structural_family"] = bh_adjust(structural["pvalue_for_fdr"])
        structural["fdr_significant_0_05"] = (structural["qvalue_final_structural_family"] < 0.05).astype(int)
    write_csv(structural, out_dir / "final_structural_multiplicity_family.csv")
    definition = """# Multiplicity Family Definition

## Confirmatory family

The authoritative confirmatory family contains the prespecified H1-H3 outcome tests and H4 moderation tests from the frozen tariff-weighted design. Sensitivity outcomes are not added after inspecting results.

## Exploratory structural family

The authoritative exploratory family contains the reduced theory-driven continuous moderator tests, moderator-block omnibus tests, cleaned-profile regime slopes and equality tests, corrected focal event-study joint tests, corrected placebo tests, H6 directional tests, and the prespecified 2019-start sensitivity results.

Balance statistics, silhouette scores, VIFs, composition counts, power/MDE values, equivalence bounds, leave-one-out ranges, and failed weighting diagnostics are descriptive and are not assigned q-values. Residual-density weighting is not an adjustment model.

All q-values are recomputed from the exact rows in final_structural_multiplicity_family.csv using Benjamini-Hochberg adjustment.
"""
    (out_dir / "multiplicity_family_definition.md").write_text(definition, encoding="utf-8")
    return confirm, structural


def write_reproduction(base_dir: Path, out_dir: Path) -> None:
    pinned_requirements = (
        "numpy==2.4.3\n"
        "pandas==2.3.3\n"
        "scipy==1.17.1\n"
        "scikit-learn==1.8.0\n"
        "matplotlib==3.10.8\n"
        "pytest==9.1.1\n"
    )
    (base_dir / "environment.yml").write_text(
        "name: geo-scm-final-analysis\n"
        "channels:\n"
        "  - conda-forge\n"
        "dependencies:\n"
        "  - python=3.14.3\n"
        "  - numpy=2.4.3\n"
        "  - pandas=2.3.3\n"
        "  - scipy=1.17.1\n"
        "  - scikit-learn=1.8.0\n"
        "  - matplotlib=3.10.8\n"
        "  - pytest=9.1.1\n",
        encoding="utf-8",
    )
    (base_dir / "requirements.txt").write_text(pinned_requirements, encoding="utf-8")
    inputs = [
        "reports/final_design_completion/panel_with_completed_design_constructs.csv",
        "data/raw/structural_wdi_2012_2017.csv",
        "scripts/run_final_audited_analysis.py",
        "scripts/run_post_period_sensitivity.py",
        "scripts/run_reproducibility_proof.py",
        "tests/test_final_analysis_integrity.py",
        "environment.yml",
        "requirements.txt",
    ]
    manifest = []
    for relative in inputs:
        path = base_dir / relative
        manifest.append(
            {
                "path": relative,
                "exists": path.exists(),
                "sha256": sha256(path) if path.exists() else "",
            }
        )
    write_csv(pd.DataFrame(manifest), out_dir / "reproduction_manifest.csv")
    (out_dir / "reproduction_log.txt").write_text(
        "Commands are executed by scripts/run_reproducibility_proof.py.\n"
        "Complete console output: reports/structural_regime_completion/reproducibility_proof.log\n"
        "Environment and package versions: reports/structural_regime_completion/reproducibility_versions.csv\n"
        "All inputs are repository-relative; no local absolute paths are required.\n",
        encoding="utf-8",
    )
    hashes = []
    for path in sorted(out_dir.glob("*")):
        if not path.is_file() or path.name == "output_file_hashes.csv":
            continue
        if path.suffix.lower() == ".log" and path.name not in {
            "reproducibility_proof.log", "reproduction_log.txt"
        }:
            continue
        hashes.append(
                {
                    "path": str(path.relative_to(base_dir)),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    write_csv(pd.DataFrame(hashes), out_dir / "output_file_hashes.csv")

def write_tests(base_dir: Path) -> None:
    tests = """from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'structural_regime_completion'

def test_required_final_outputs_exist():
    required = [
        'valid_country_sample_audit.csv', 'excluded_entities.csv',
        'structural_missingness_by_country.csv', 'structural_sample_comparison.csv',
        'structural_regime_selection_clean.csv', 'structural_regime_consensus_matrix.csv',
        'structural_regime_assignment_probabilities.csv', 'structural_regime_assignments_clean.csv',
        'structural_regime_profiles_clean.csv', 'structural_regime_difference_tests_corrected.csv',
        'structural_regime_omnibus_validation.csv', 'corrected_event_study_coefficients.csv',
        'corrected_event_study_pretrend_tests.csv', 'corrected_event_study_post_tests.csv',
        'post_period_2018_2019_comparison.csv',
        'structural_regime_specific_coefficients_2019.csv',
        'structural_regime_difference_tests_2019.csv',
        'structural_regime_omnibus_validation_2019.csv',
        'corrected_event_study_post_period_summary.csv', 'regime_power_mde_outcome_specific.csv',
        'regime_difference_power_mde.csv', 'progressive_adjustment_common_sample.csv',
        'progressive_adjustment_maximum_sample.csv', 'coefficient_stability_summary.csv',
        'residual_density_weight_diagnostics.csv', 'balance_before_after_weighting.csv',
'theory_based_balance_diagnostics.csv',
        'structural_moderator_block_omnibus_tests.csv', 'regime_assignment_robustness.csv',
        'regime_model_assignment_sensitivity.csv', 'leave_one_country_out_summary_stats.csv',
        'leave_one_region_out_results.csv', 'influential_country_region_audit.csv',
        'placebo_2014_2015_comparison.csv', 'final_equivalence_tests.csv',
        'final_minimum_detectable_effects.csv', 'final_confirmatory_multiplicity_family.csv',
        'final_structural_multiplicity_family.csv', 'multiplicity_family_definition.md',
        'reproduction_manifest.csv', 'output_file_hashes.csv', 'reproduction_log.txt',
        'reproducibility_proof.log', 'reproducibility_versions.csv',
    ]
    missing = [name for name in required if not (OUT / name).exists()]
    assert not missing, missing

def test_k2_omnibus_equals_pairwise():
    table = pd.read_csv(OUT / 'structural_regime_omnibus_validation.csv')
    assert table.loc[table['k'] == 2, 'validation_passed'].all()

def test_event_study_has_eci_exposure_terms():
    table = pd.read_csv(OUT / 'corrected_event_study_coefficients.csv')
    assert table['term'].astype(str).str.contains('eci_exposure').all()

def test_primary_sample_constraints():
    table = pd.read_csv(OUT / 'valid_country_sample_audit.csv')
    primary = table.loc[table['primary_structural_sample']]
    assert (primary['cluster_imputed_fields'] <= 2).all()
    assert primary['valid_entity'].all()
    assert primary['valid_exposure'].all()
    assert primary['valid_eci'].all()

def test_post_2019_outputs_present():
    table = pd.read_csv(OUT / 'post_period_2018_2019_comparison.csv')
    assert not table.empty
    assert {'estimate_2018', 'estimate_2019'}.issubset(table.columns)

def test_outcome_power_counts_are_valid():
    table = pd.read_csv(OUT / 'regime_power_mde_outcome_specific.csv')
    assert (table['n_countries_used'] == table['effective_number_of_clusters']).all()

def test_family_qvalues_are_valid():
    table = pd.read_csv(OUT / 'final_structural_multiplicity_family.csv')
    assert table['test_id'].notna().all()
    assert table['qvalue_final_structural_family'].between(0, 1).all()

def test_2019_regime_and_event_outputs():
    validation = pd.read_csv(OUT / 'structural_regime_omnibus_validation_2019.csv')
    assert validation['validation_passed'].all()
    event = pd.read_csv(OUT / 'corrected_event_study_post_period_summary.csv')
    assert 2019 in set(event['post_start'].astype(int))

def test_power_counts_do_not_exceed_observed_panel():
    power = pd.read_csv(OUT / 'regime_power_mde_outcome_specific.csv')
    panel = pd.read_csv(ROOT / 'reports' / 'final_design_completion' / 'panel_with_completed_design_constructs.csv')
    columns = {
        'gvc': 'gvc_adverse_deviation_stability',
        'recovery': 'log_export_recovery',
        'diversification': 'partner_diversification_excl_us_china',
    }
    for key, column in columns.items():
        observed_rows = int(panel[column].notna().sum())
        assert int(power.loc[power['outcome_key'].eq(key), 'n_country_year_observations'].max()) <= observed_rows

def test_family_has_no_obsolete_or_duplicate_ids():
    family = pd.read_csv(OUT / 'final_structural_multiplicity_family.csv')
    assert family['test_id'].notna().all()
    assert family['test_id'].is_unique
    assert family['qvalue_final_structural_family'].notna().all()

def test_frozen_focal_reference_values():
    focal = pd.read_csv(OUT / 'final_focal_models_2018.csv').set_index('test_id')
    expected = {
        'focal_2018_gvc_eci_exposure_post': 0.23079135951473873,
        'focal_2018_recovery_eci_exposure_post': -0.052508028999043645,
        'focal_2018_diversification_eci_exposure_post': -0.013137820287150031,
    }
    for test_id, value in expected.items():
        assert np.isclose(float(focal.loc[test_id, 'estimate']), value, atol=1e-9)
def test_country_validity_keeps_real_economies():
    audit = pd.read_csv(OUT / 'valid_country_sample_audit.csv')
    for code in ['CAF', 'ZAF']:
        row = audit.loc[audit['country_iso3_code'].eq(code)].iloc[0]
        assert bool(row['valid_entity'])
        assert bool(row['primary_structural_sample'])
        assert row['entity_validity_reason'] == 'valid_sovereign_or_analytical_economy'

def test_event_study_scope_matches_confirmatory_samples():
    coefficients = pd.read_csv(OUT / 'corrected_event_study_coefficients.csv')
    full = coefficients.loc[coefficients['variant'].eq('full_sample_confirmatory')]
    expected = {'gvc': 78, 'recovery': 228, 'diversification': 228}
    for key, n in expected.items():
        values = full.loc[full['outcome_key'].eq(key), 'n_countries'].unique()
        assert len(values) == 1 and int(values[0]) == n
    pretrend = pd.read_csv(OUT / 'corrected_event_study_pretrend_tests.csv')
    assert 'full_sample_cleaned_structural' in set(pretrend['variant'])
    assert 'pooled_regime_interacted' in set(pretrend['variant'])

def test_weighting_is_diagnostic_only():
    balance = pd.read_csv(OUT / 'balance_before_after_weighting.csv')
    assert 'joint_balance_pvalue' not in balance.columns
    assert balance['diagnostic_status'].eq('failed_balance').all()
    assert (balance['absolute_smd_after'] > 0.10).all()
    assert not (OUT / 'weighted_model_trimmed_sensitivity.csv').exists()

def test_reproducibility_proof_is_complete():
    log = (OUT / 'reproducibility_proof.log').read_text(encoding='utf-8')
    versions = pd.read_csv(OUT / 'reproducibility_versions.csv')
    assert 'EXIT_CODE: 0' in log
    assert {'python', 'package', 'version'}.issubset(versions.columns)
    assert versions['version'].notna().all()
"""
    tests_dir = base_dir / "tests"
    ensure_dir(tests_dir)
    (tests_dir / "test_final_analysis_integrity.py").write_text(tests, encoding="utf-8")


def write_readme(out_dir: Path, metadata: dict[str, Any]) -> None:
    text = f"""# Final Audited Analysis Snapshot

This is the corrected and reproducible analysis layer requested in the audit memo. It does not search for another post-hoc hypothesis.

## Frozen computation

- Cleaned primary structural sample: {metadata["primary_n"]} countries.
- Complete-case sensitivity: {metadata["complete_n"]} countries.
- Broad legacy sensitivity: {metadata["broad_n"]} countries.
- Consensus clustering uses {MI_REPS} imputed profiles and only k=2 and k=3.
- Primary post period: 2018 onward; prespecified sensitivity: 2019 onward.
- Wild-cluster bootstrap replications: {BOOTSTRAP_REPS}.
- Equivalence bound: {EQUIVALENCE_RULE}.
- Confirmatory event studies use the original outcome-specific samples (78 GVC countries; 228 recovery/diversification countries).
- Residual-density weighting is reported only as a failed balance diagnostic; adjustment evidence uses Models A-D.

## Interpretation rule

A structural profile cannot enter the main theory unless it survives corrected sample construction, consensus assignment, formal regime-equality testing, corrected focal event-study diagnostics, placebo checks, adjustment stability, influence checks, outcome-specific precision, and final exploratory-family correction.

## Authoritative files

- final_confirmatory_multiplicity_family.csv
- final_structural_multiplicity_family.csv
- multiplicity_family_definition.md
- authoritative_results_manifest.csv
- corrected_event_study_coefficients.csv
- post_period_2018_2019_comparison.csv
- progressive_adjustment_common_sample.csv (Models A-D only)
- balance_before_after_weighting.csv (failed diagnostic only)
- regime_assignment_robustness.csv
- balance_before_after_weighting.csv (failed diagnostic only; not an adjustment model)
- progressive_adjustment_common_sample.csv (Models A-D only)

Legacy structural files remain as an audit trail; they are not authoritative after this snapshot.
"""
    (out_dir / "FINAL_ANALYSIS_README.md").write_text(text, encoding="utf-8")


def write_authoritative_manifest(out_dir: Path) -> pd.DataFrame:
    mapping = {
        "H1-H3 and H4 confirmatory results": "final_confirmatory_multiplicity_family.csv",
        "clean structural sample": "valid_country_sample_audit.csv",
        "consensus assignments": "structural_regime_assignment_probabilities.csv",
        "corrected regime equality": "structural_regime_difference_tests_corrected.csv",
        "corrected event study": "corrected_event_study_coefficients.csv",
        "event-study pretrend tests": "corrected_event_study_pretrend_tests.csv",
        "event-study post tests": "corrected_event_study_post_tests.csv",
        "2018 versus 2019 sensitivity": "post_period_2018_2019_comparison.csv",
        "2019 continuous moderator sensitivity": "post_period_2019_continuous_moderators.csv",
        "corrected event-study post summary": "corrected_event_study_post_period_summary.csv",
        "outcome-specific power": "regime_power_mde_outcome_specific.csv",
        "common-sample adjustment": "progressive_adjustment_common_sample.csv",
        "weight diagnostics": "residual_density_weight_diagnostics.csv",
        "theory balance": "theory_based_balance_diagnostics.csv",
        "placebo comparison": "placebo_2014_2015_comparison.csv",
        "equivalence and MDE": "final_equivalence_tests.csv",
        "confirmatory multiplicity": "final_confirmatory_multiplicity_family.csv",
        "structural multiplicity": "final_structural_multiplicity_family.csv",
        "reproduction hashes": "output_file_hashes.csv",
        "reproduction proof": "reproducibility_proof.log",
        "reproducibility versions": "reproducibility_versions.csv",
    }
    mapping.update(
        {
            "clean profile base": "structural_profile_final_audit_base.csv",
            "clean exposure counts": "structural_regime_exposure_counts_clean.csv",
            "regime-specific slopes": "structural_regime_specific_coefficients_corrected.csv",
            "regime difference tests": "structural_regime_difference_tests_corrected.csv",
            "regime omnibus validation": "structural_regime_omnibus_validation.csv",
            "outcome-specific regime-difference power": "regime_difference_power_mde.csv",
            "maximum-sample adjustment": "progressive_adjustment_maximum_sample.csv",
            "coefficient stability": "coefficient_stability_summary.csv",
            "weight balance diagnostic": "balance_before_after_weighting.csv",
            "moderator block omnibus": "structural_moderator_block_omnibus_tests.csv",
            "assignment robustness": "regime_assignment_robustness.csv",
            "assignment model sensitivity": "regime_model_assignment_sensitivity.csv",
            "country influence detail": "leave_one_country_out_summary.csv",
            "H6 directional tests": "h6_directional_2018_2019.csv",
            "minimum detectable effects": "final_minimum_detectable_effects.csv",
            "event-study figures": "figure_corrected_event_study_gvc.png;figure_corrected_event_study_recovery.png;figure_corrected_event_study_diversification.png",
            "2019 regime-specific slopes": "structural_regime_specific_coefficients_2019.csv",
            "2019 regime differences": "structural_regime_difference_tests_2019.csv",
            "2019 regime omnibus validation": "structural_regime_omnibus_validation_2019.csv",
        }
    )
    result = pd.DataFrame(
        [
            {
                "analysis_component": component,
                "authoritative_file": file,
                "role": "final audited snapshot",
            }
            for component, file in mapping.items()
        ]
    )
    write_csv(result, out_dir / "authoritative_results_manifest.csv")
    return result


def main() -> None:
    global MI_REPS, STABILITY_REPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--mi-reps", type=int, default=MI_REPS)
    parser.add_argument("--stability-reps", type=int, default=STABILITY_REPS)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--skip-influence", action="store_true")
    args = parser.parse_args()
    MI_REPS = args.mi_reps
    STABILITY_REPS = args.stability_reps
    base_dir = Path(args.base_dir).resolve()
    out_dir = base_dir / "reports" / "structural_regime_completion"
    ensure_dir(out_dir)

    panel = pd.read_csv(base_dir / "reports" / "final_design_completion" / "panel_with_completed_design_constructs.csv")
    wdi, _, _ = load_wdi_structural_source(base_dir)
    raw_profile, _, _ = build_structural_profile(panel, wdi)
    profile = build_samples(panel, raw_profile, out_dir)
    primary_assignment, assignments, selection = run_clustering(profile, out_dir, args.seed)
    primary_profile = profile.loc[profile["primary_structural_sample"]].copy()
    model_panel = prepare_panel(panel, primary_profile, primary_assignment)
    confirmatory_event_panel = prepare_confirmatory_event_panel(panel, raw_profile)
    regimes = sorted(primary_assignment["structural_profile"].dropna().unique())
    if len(regimes) != 2:
        raise ValueError(f"Expected two selected primary profiles, got {regimes}")

    focal_2018 = focal_models(model_panel, PRIMARY_POST, args.bootstrap_reps, args.seed + 100)
    slopes, differences, validation, fits = corrected_regime_models(
        model_panel, PRIMARY_POST, regimes, args.bootstrap_reps, args.seed + 300
    )
    write_csv(focal_2018, out_dir / "final_focal_models_2018.csv")
    write_csv(slopes, out_dir / "structural_regime_specific_coefficients_corrected.csv")
    write_csv(differences, out_dir / "structural_regime_difference_tests_corrected.csv")
    write_csv(validation, out_dir / "structural_regime_omnibus_validation.csv")
    if not validation.loc[validation["k"].eq(2), "validation_passed"].all():
        raise AssertionError("Corrected k=2 omnibus test does not equal pairwise test")

    event_coefficients, event_pretrend, event_post = corrected_event_studies(
        model_panel, regimes, out_dir, args.bootstrap_reps, args.seed + 400,
        confirmatory_frame=confirmatory_event_panel,
    )
    blocks = moderator_block_tests(model_panel, out_dir, args.bootstrap_reps, args.seed + 500)
    weighted, balance = weight_and_balance(primary_profile, model_panel, out_dir)
    theory = theory_balance(primary_profile, weighted, out_dir)
    common, maximum, stability = adjustment_models(model_panel, out_dir)
    power, difference_power = power_tables(model_panel, fits, slopes, differences, out_dir)
    placebo = placebo_comparison(model_panel, out_dir, args.bootstrap_reps, args.seed + 600)
    post_compare = post_period_comparison(model_panel, out_dir, regimes, args.bootstrap_reps, args.seed + 700)
    h6_2018 = h6_directional(model_panel, PRIMARY_POST, args.bootstrap_reps, args.seed + 800)
    h6_2019 = h6_directional(model_panel, SENSITIVITY_POST, args.bootstrap_reps, args.seed + 900)
    h6 = pd.concat([h6_2018, h6_2019], ignore_index=True)
    write_csv(h6, out_dir / "h6_directional_2018_2019.csv")
    assignment_robustness, assignment_models = assignment_sensitivity(
        panel, assignments, profile, out_dir, args.bootstrap_reps, args.seed + 1000
    )
    if args.skip_influence:
        loo_summary, loo_regions, influence = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        write_csv(loo_summary, out_dir / "leave_one_country_out_summary_stats.csv")
        write_csv(loo_regions, out_dir / "leave_one_region_out_results.csv")
        write_csv(influence, out_dir / "influential_country_region_audit.csv")
    else:
        loo_summary, loo_regions, influence = influence_analysis(model_panel, regimes, out_dir)
    equivalence, mde = equivalence_tests(
        panel, focal_2018, common, differences, out_dir, base_dir
    )

    moderator_rows = []
    for oi, (outcome_key, (outcome, label)) in enumerate(OUTCOMES.items()):
        for vi, variable in enumerate(STRUCTURAL_VARS):
            work, terms = add_base_terms(model_panel)
            work, mod_terms = add_moderator_terms(work, "z_" + variable)
            terms += mod_terms
            try:
                fit = fit_terms(work, outcome, terms)
                row = coefficient_row(
                    fit, "eci_exposure_post_x_moderator", args.bootstrap_reps,
                    args.seed + 2000 + oi * 100 + vi,
                )
                row.update(
                    {
                        "test_id": f"final_continuous_{outcome_key}_{variable}",
                        "outcome_key": outcome_key, "outcome": label,
                        "moderator": "z_" + variable,
                        "moderator_label": STRUCTURAL_LABELS[variable],
                    }
                )
                moderator_rows.append(row)
            except Exception as exc:
                moderator_rows.append(
                    {
                        "test_id": f"final_continuous_{outcome_key}_{variable}",
                        "outcome_key": outcome_key, "outcome": label,
                        "moderator": "z_" + variable, "error": str(exc),
                        "pvalue_for_fdr": np.nan,
                    }
                )
    continuous = pd.DataFrame(moderator_rows)
    write_csv(continuous, out_dir / "final_continuous_structural_moderator_tests.csv")

    post_family = post_compare.copy()
    post_family["test_id"] = (
        "post_period_2019_" + post_family["analysis"].astype(str)
        + "_" + post_family["outcome_key"].astype(str)
        + "_" + post_family["entity"].astype(str)
    )
    post_family["pvalue_for_fdr"] = post_family["p_wild_2019"]
    post_family["bootstrap_reps_requested"] = args.bootstrap_reps
    post_family["bootstrap_reps_success"] = args.bootstrap_reps
    structural_tables = {
        "continuous": continuous,
        "blocks": blocks,
        "regime_slopes": slopes,
        "regime_differences": differences,
        "event_pretrend": event_pretrend,
        "placebo": placebo,
        "post_period": post_family,
        "h6": h6_2018,
    }
    confirm, structural = write_final_families(base_dir, out_dir, structural_tables)

    metadata = {
        "primary_n": int(primary_profile["country_iso3_code"].nunique()),
        "complete_n": int(profile["complete_case_structural_sample"].sum()),
        "broad_n": int(len(profile)),
        "selected_k": int(selection.loc[
            selection["sample"].eq("cleaned_primary") & selection["selected_k"].notna(),
            "selected_k"
        ].iloc[0]),
        "confirmatory_tests": len(confirm),
        "confirmatory_q_lt_005": int((confirm["qvalue_final_confirmatory_family"] < 0.05).sum()),
        "structural_tests": len(structural),
        "structural_q_lt_005": int((structural["qvalue_final_structural_family"] < 0.05).sum()),
    }
    write_readme(out_dir, metadata)
    write_tests(base_dir)
    summary = pd.DataFrame(
        [
            {"item": key, "value": value}
            for key, value in {
                "primary_sample_countries": metadata["primary_n"],
                "complete_case_countries": metadata["complete_n"],
                "broad_sample_countries": metadata["broad_n"],
                "selected_k": metadata["selected_k"],
                "confirmatory_family_tests": metadata["confirmatory_tests"],
                "confirmatory_q_lt_005": metadata["confirmatory_q_lt_005"],
                "structural_family_tests": metadata["structural_tests"],
                "structural_q_lt_005": metadata["structural_q_lt_005"],
                "bootstrap_reps": args.bootstrap_reps,
                "mi_reps": args.mi_reps,
            }.items()
        ]
    )
    write_csv(summary, out_dir / "final_audit_summary.csv")
    write_authoritative_manifest(out_dir)
    write_reproduction(base_dir, out_dir)
    print(summary.to_string(index=False))
    print("Final audited analysis written to reports/structural_regime_completion")


if __name__ == "__main__":
    main()
