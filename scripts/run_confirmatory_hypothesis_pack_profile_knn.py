from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


OUTCOME_MAP = {
    "H1": ("DV1_GVC_Linkage_Stability", "gvc_linkage_stability"),
    "H2": ("DV2_Export_Recovery", "export_recovery_index"),
    "H3": ("DV3_Partner_Diversification", "partner_diversification_1_minus_hhi"),
}

PROFILE_FEATURES = [
    "eci",
    "coi",
    "log_gdp_pc",
    "wdi_trade_openness_pct_gdp",
    "wgi_institutional_quality_composite",
    "us_china_trade_intensity_pre",
    "wdi_natural_resource_rents_pct_gdp",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def zscore(series: pd.Series) -> pd.Series:
    mu = series.mean(skipna=True)
    sigma = series.std(skipna=True, ddof=0)
    if pd.isna(sigma) or sigma == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - mu) / sigma


def _predict_knn_for_row(
    recipient: pd.Series,
    donors: pd.DataFrame,
    feature_scales: pd.Series,
    k: int,
    min_common_features: int,
) -> tuple[float | None, int, float]:
    dists: list[tuple[int, float, int]] = []
    for didx, d in donors.iterrows():
        vals = []
        for f in PROFILE_FEATURES:
            rv = recipient.get(f, np.nan)
            dv = d.get(f, np.nan)
            sf = feature_scales.get(f, np.nan)
            if pd.notna(rv) and pd.notna(dv) and pd.notna(sf) and sf > 0:
                vals.append(((float(rv) - float(dv)) / float(sf)) ** 2)
        if len(vals) >= min_common_features:
            dist = float(np.sqrt(np.mean(vals)))
            dists.append((int(didx), dist, len(vals)))

    if not dists:
        return None, 0, 0.0

    dists = sorted(dists, key=lambda x: x[1])[:k]
    dist_arr = np.array([x[1] for x in dists], dtype=float)
    y_arr = np.array([donors.loc[x[0], "gpr_country_annual"] for x in dists], dtype=float)
    weights = 1.0 / (dist_arr + 1e-6)
    pred = float(np.sum(weights * y_arr) / np.sum(weights))
    avg_common = float(np.mean([x[2] for x in dists]))
    return pred, len(dists), avg_common


def apply_profile_knn_gpr_imputation(
    panel: pd.DataFrame,
    k: int = 5,
    min_common_features: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = panel.copy()
    df["gpr_profile_knn_annual"] = df["gpr_country_annual"]
    df["gpr_profile_knn_method"] = np.where(
        df["gpr_country_annual"].notna(), "observed_country_gpr", ""
    )
    df["gpr_profile_knn_neighbors"] = np.where(df["gpr_country_annual"].notna(), 0, np.nan)
    df["gpr_profile_knn_avg_common_features"] = np.where(
        df["gpr_country_annual"].notna(), 0.0, np.nan
    )

    for y, idx in df.groupby("year").groups.items():
        ydf = df.loc[idx].copy()
        donors = ydf[ydf["gpr_country_annual"].notna()].copy()
        recips = ydf[ydf["gpr_country_annual"].isna()].copy()
        if donors.empty or recips.empty:
            continue

        feature_scales = donors[PROFILE_FEATURES].std(skipna=True, ddof=0).replace(0, np.nan)
        year_median = float(donors["gpr_country_annual"].median())

        for ridx, r in recips.iterrows():
            pred, n_nb, avg_common = _predict_knn_for_row(
                recipient=r,
                donors=donors,
                feature_scales=feature_scales,
                k=k,
                min_common_features=min_common_features,
            )
            method = "knn_profile_year"
            if pred is None:
                pred, n_nb, avg_common = _predict_knn_for_row(
                    recipient=r,
                    donors=donors,
                    feature_scales=feature_scales,
                    k=k,
                    min_common_features=1,
                )
                method = "knn_profile_year_min1"

            if pred is None:
                pred = year_median
                n_nb = int(len(donors))
                avg_common = 0.0
                method = "year_median_fallback"

            df.at[ridx, "gpr_profile_knn_annual"] = float(pred)
            df.at[ridx, "gpr_profile_knn_method"] = method
            df.at[ridx, "gpr_profile_knn_neighbors"] = int(n_nb)
            df.at[ridx, "gpr_profile_knn_avg_common_features"] = float(avg_common)

    audit = (
        df.groupby(["year", "gpr_profile_knn_method"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
    )
    return df, audit


def add_engineered_columns(
    panel: pd.DataFrame,
    gpr_col: str,
) -> pd.DataFrame:
    df = panel.copy()
    df["gvc_linkage_stability"] = -df["delta_tiva_fexgr_dva_share"].abs()
    df["post_paper"] = df["year"].between(2019, 2022).astype(int)
    df["pe"] = df["post_paper"] * df["exposed"]

    for col, zcol in [
        ("eci", "z_eci"),
        ("coi", "z_coi"),
        (gpr_col, "z_gpr"),
        ("log_gdp_pc", "z_log_gdp_pc"),
        ("wdi_trade_openness_pct_gdp", "z_trade_open"),
        ("wgi_institutional_quality_composite", "z_wgi"),
        ("wdi_natural_resource_rents_pct_gdp", "z_rents"),
        ("us_china_trade_intensity_pre", "z_intensity"),
    ]:
        df[zcol] = zscore(df[col])
    return df


def add_size_quartile(panel: pd.DataFrame) -> pd.DataFrame:
    csize = (
        panel[["country_iso3_code", "baseline_export_2015_2017"]]
        .dropna()
        .drop_duplicates()
        .groupby("country_iso3_code", as_index=False)["baseline_export_2015_2017"]
        .mean()
    )
    csize["size_q"] = pd.qcut(
        csize["baseline_export_2015_2017"],
        q=4,
        labels=[1, 2, 3, 4],
        duplicates="drop",
    )
    return panel.merge(csize[["country_iso3_code", "size_q"]], on="country_iso3_code", how="left")


def control_terms(include_rents_main: bool) -> list[str]:
    terms = ["z_log_gdp_pc", "z_trade_open", "z_wgi", "z_gpr", "z_intensity"]
    if include_rents_main:
        terms.insert(3, "z_rents")
    return terms


def prepare_sample(
    panel: pd.DataFrame,
    outcome_col: str,
    sample_name: str,
    include_rents_main: bool,
) -> pd.DataFrame:
    df = panel.copy()
    if sample_name == "core_excl_q1":
        df = df[df["size_q"] != 1].copy()
    req = [outcome_col, "z_eci", "pe", "country_iso3_code", "year"] + control_terms(
        include_rents_main
    )
    return df.dropna(subset=req).copy()


def primary_formula(
    outcome_col: str,
    include_rents_main: bool,
    pe_col: str = "pe",
) -> str:
    cterms = " + ".join(control_terms(include_rents_main))
    return (
        f"{outcome_col} ~ z_eci + {pe_col} + z_eci:{pe_col} + "
        f"{cterms} + C(country_iso3_code) + C(year)"
    )


def moderation_formula(outcome_col: str, moderator: str, include_rents_main: bool) -> str:
    cterms = " + ".join(control_terms(include_rents_main))
    if moderator == "coi":
        return (
            f"{outcome_col} ~ z_eci*pe*z_coi + "
            f"{cterms} + C(country_iso3_code) + C(year)"
        )
    return (
        f"{outcome_col} ~ z_eci*pe*z_gpr + "
        f"{cterms} + C(country_iso3_code) + C(year)"
    )


def fit_clustered(formula: str, df: pd.DataFrame) -> Any:
    return smf.ols(formula=formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["country_iso3_code"]},
    )


def run_wild_cluster_bootstrap_term(
    df: pd.DataFrame,
    outcome_col: str,
    target_term: str,
    reps: int,
    seed: int,
    include_rents_main: bool,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    f_full = primary_formula(outcome_col, include_rents_main=include_rents_main)
    f_restricted = f_full.replace(f" + {target_term}", "")

    fit_full = fit_clustered(f_full, df)
    coef_obs = float(fit_full.params.get(target_term, np.nan))
    se_obs = float(fit_full.bse.get(target_term, np.nan))
    t_obs = coef_obs / se_obs if pd.notna(coef_obs) and pd.notna(se_obs) and se_obs != 0 else np.nan

    fit0 = smf.ols(formula=f_restricted, data=df).fit()
    yhat0 = fit0.fittedvalues.to_numpy()
    u0 = fit0.resid.to_numpy()

    clusters = df["country_iso3_code"].astype(str).to_numpy()
    unique_clusters = np.unique(clusters)
    ystar = f"{outcome_col}__star"
    f_star = f_full.replace(outcome_col, ystar, 1)
    bdf = df.copy()

    t_star: list[float] = []
    coef_star: list[float] = []
    for _ in range(reps):
        draw = rng.choice([-1.0, 1.0], size=len(unique_clusters))
        wmap = dict(zip(unique_clusters, draw))
        w = np.array([wmap[c] for c in clusters], dtype=float)
        bdf[ystar] = yhat0 + u0 * w
        try:
            fb = fit_clustered(f_star, bdf)
            c = float(fb.params.get(target_term, np.nan))
            s = float(fb.bse.get(target_term, np.nan))
            if pd.notna(c) and pd.notna(s) and s != 0:
                coef_star.append(c)
                t_star.append(c / s)
        except Exception:
            continue

    tarr = np.array(t_star, dtype=float) if t_star else np.array([np.nan])
    carr = np.array(coef_star, dtype=float) if coef_star else np.array([np.nan])
    p_boot = (
        float(np.mean(np.abs(tarr[np.isfinite(tarr)]) >= abs(t_obs)))
        if np.isfinite(t_obs) and np.isfinite(tarr).any()
        else np.nan
    )
    return {
        "coef_obs": coef_obs,
        "se_obs": se_obs,
        "p_cluster": float(fit_full.pvalues.get(target_term, np.nan)),
        "p_boot": p_boot,
        "bootstrap_reps_requested": int(reps),
        "bootstrap_reps_success": int(np.isfinite(tarr).sum()),
        "boot_ci_low_2p5": float(np.nanpercentile(carr, 2.5)) if np.isfinite(carr).any() else np.nan,
        "boot_ci_high_97p5": float(np.nanpercentile(carr, 97.5)) if np.isfinite(carr).any() else np.nan,
    }


def run_primary_tests(
    panel: pd.DataFrame,
    out_dir: Path,
    bootstrap_reps: int,
    seed: int,
    include_rents_main: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, (hid, (dv_label, outcome_col)) in enumerate(OUTCOME_MAP.items()):
        df = prepare_sample(panel, outcome_col, sample_name="full", include_rents_main=include_rents_main)
        if df.empty:
            continue
        fit = fit_clustered(primary_formula(outcome_col, include_rents_main=include_rents_main), df)
        term = "z_eci:pe"
        coef = float(fit.params.get(term, np.nan))
        p = float(fit.pvalues.get(term, np.nan))

        boot = run_wild_cluster_bootstrap_term(
            df=df,
            outcome_col=outcome_col,
            target_term=term,
            reps=bootstrap_reps,
            seed=seed + i,
            include_rents_main=include_rents_main,
        )
        rows.append(
            {
                "hypothesis_id": hid,
                "dv": dv_label,
                "outcome_col": outcome_col,
                "expected_sign": "positive",
                "coef": coef,
                "se": float(fit.bse.get(term, np.nan)),
                "p_cluster": p,
                "p_boot": boot["p_boot"],
                "boot_ci_low_2p5": boot["boot_ci_low_2p5"],
                "boot_ci_high_97p5": boot["boot_ci_high_97p5"],
                "bootstrap_reps_success": boot["bootstrap_reps_success"],
                "n_obs": int(fit.nobs),
                "n_countries": int(df["country_iso3_code"].nunique()),
                "year_min": int(df["year"].min()),
                "year_max": int(df["year"].max()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "confirmatory_primary_tests.csv", index=False)
    return out


def run_moderation_tests(
    panel: pd.DataFrame,
    out_dir: Path,
    include_rents_main: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, (dv_label, outcome_col) in OUTCOME_MAP.items():
        df = prepare_sample(panel, outcome_col, sample_name="full", include_rents_main=include_rents_main)
        if df.empty:
            continue

        fit_h4 = fit_clustered(
            moderation_formula(outcome_col, "coi", include_rents_main=include_rents_main),
            df.dropna(subset=["z_coi"]).copy(),
        )
        term_h4 = "z_eci:pe:z_coi"
        rows.append(
            {
                "hypothesis_id": "H4",
                "dv": dv_label,
                "outcome_col": outcome_col,
                "expected_sign": "positive",
                "term": term_h4,
                "coef": float(fit_h4.params.get(term_h4, np.nan)),
                "se": float(fit_h4.bse.get(term_h4, np.nan)),
                "p_cluster": float(fit_h4.pvalues.get(term_h4, np.nan)),
                "n_obs": int(fit_h4.nobs),
                "n_countries": int(df["country_iso3_code"].nunique()),
            }
        )

        fit_h5 = fit_clustered(
            moderation_formula(outcome_col, "gpr", include_rents_main=include_rents_main),
            df,
        )
        term_h5 = "z_eci:pe:z_gpr"
        rows.append(
            {
                "hypothesis_id": "H5",
                "dv": dv_label,
                "outcome_col": outcome_col,
                "expected_sign": "negative",
                "term": term_h5,
                "coef": float(fit_h5.params.get(term_h5, np.nan)),
                "se": float(fit_h5.bse.get(term_h5, np.nan)),
                "p_cluster": float(fit_h5.pvalues.get(term_h5, np.nan)),
                "n_obs": int(fit_h5.nobs),
                "n_countries": int(df["country_iso3_code"].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "confirmatory_moderation_tests.csv", index=False)
    return out


def run_pretrend_and_placebo(
    panel: pd.DataFrame,
    out_dir: Path,
    include_rents_main: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ctrl = " + ".join(control_terms(include_rents_main))
    for hid, (dv_label, outcome_col) in OUTCOME_MAP.items():
        df = prepare_sample(panel, outcome_col, sample_name="full", include_rents_main=include_rents_main)
        if df.empty:
            continue

        est = df.copy()
        est["event_time"] = (est["year"] - 2018).astype(int)
        est = est[est["event_time"].between(-6, 4)].copy()
        f_es = (
            f"{outcome_col} ~ z_eci + {ctrl} + "
            "C(country_iso3_code) + C(year) + C(event_time, Treatment(reference=-1)):exposed:z_eci"
        )
        fit_es = fit_clustered(f_es, est)
        pre_terms = [
            f"C(event_time, Treatment(reference=-1))[{k}]:exposed:z_eci"
            for k in [-6, -5, -4, -3, -2]
            if f"C(event_time, Treatment(reference=-1))[{k}]:exposed:z_eci" in fit_es.params.index
        ]
        pre_p = float(fit_es.wald_test(", ".join([f"{t}=0" for t in pre_terms])).pvalue) if pre_terms else np.nan

        plc = df.copy()
        plc["post_placebo"] = plc["year"].between(2014, 2015).astype(int)
        plc["pe_placebo"] = plc["post_placebo"] * plc["exposed"]
        f_pl = primary_formula(outcome_col, include_rents_main=include_rents_main, pe_col="pe_placebo")
        fit_pl = fit_clustered(f_pl, plc)
        plc_term = "z_eci:pe_placebo"
        plc_p = float(fit_pl.pvalues.get(plc_term, np.nan))
        plc_coef = float(fit_pl.params.get(plc_term, np.nan))

        rows.append(
            {
                "hypothesis_id": hid,
                "dv": dv_label,
                "pretrend_joint_pvalue": pre_p,
                "pretrend_pass_p_gt_0p05": int(pre_p > 0.05) if pd.notna(pre_p) else 0,
                "placebo_coef": plc_coef,
                "placebo_pvalue": plc_p,
                "placebo_pass_p_gt_0p05": int(plc_p > 0.05) if pd.notna(plc_p) else 0,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "confirmatory_pretrend_placebo.csv", index=False)
    return out


def bh_fdr_adjust(pvals: pd.Series) -> pd.Series:
    n = len(pvals)
    if n == 0:
        return pd.Series(dtype=float)
    order = np.argsort(pvals.to_numpy())
    ranked = pvals.to_numpy()[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(q, 1.0)
    return pd.Series(out, index=pvals.index)


def run_fdr_family(primary_df: pd.DataFrame, mod_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, r in primary_df.iterrows():
        rows.append(
            {
                "family_test_id": f"{r['hypothesis_id']}_{r['dv']}",
                "hypothesis_id": r["hypothesis_id"],
                "dv": r["dv"],
                "coef": r["coef"],
                "expected_sign": r["expected_sign"],
                "pvalue_for_fdr": r["p_boot"],
                "source": "primary_bootstrap",
            }
        )
    for _, r in mod_df.iterrows():
        rows.append(
            {
                "family_test_id": f"{r['hypothesis_id']}_{r['dv']}",
                "hypothesis_id": r["hypothesis_id"],
                "dv": r["dv"],
                "coef": r["coef"],
                "expected_sign": r["expected_sign"],
                "pvalue_for_fdr": r["p_cluster"],
                "source": "moderation_cluster",
            }
        )
    out = pd.DataFrame(rows)
    out["fdr_bh_qvalue"] = bh_fdr_adjust(out["pvalue_for_fdr"])
    out["sig_raw_p_lt_0p05"] = (out["pvalue_for_fdr"] < 0.05).astype(int)
    out["sig_fdr_q_lt_0p05"] = (out["fdr_bh_qvalue"] < 0.05).astype(int)
    out.to_csv(out_dir / "confirmatory_multiple_testing.csv", index=False)
    return out


def run_robustness_layer(
    panel: pd.DataFrame,
    out_dir: Path,
    include_rents_main: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_name in ["full", "core_excl_q1"]:
        for hid, (dv_label, outcome_col) in OUTCOME_MAP.items():
            df = prepare_sample(panel, outcome_col, sample_name=sample_name, include_rents_main=include_rents_main)
            if df.empty:
                continue
            f = primary_formula(outcome_col, include_rents_main=include_rents_main)
            fit_ols = fit_clustered(f, df)
            rows.append(
                {
                    "sample": sample_name,
                    "hypothesis_id": hid,
                    "dv": dv_label,
                    "spec": "OLS_FE_cluster",
                    "coef": float(fit_ols.params.get("z_eci:pe", np.nan)),
                    "pvalue": float(fit_ols.pvalues.get("z_eci:pe", np.nan)),
                    "n_obs": int(fit_ols.nobs),
                    "n_countries": int(df["country_iso3_code"].nunique()),
                }
            )

            try:
                fit_hb = smf.rlm(f, data=df, M=sm.robust.norms.HuberT()).fit()
                rows.append(
                    {
                        "sample": sample_name,
                        "hypothesis_id": hid,
                        "dv": dv_label,
                        "spec": "RLM_Huber_FE",
                        "coef": float(fit_hb.params.get("z_eci:pe", np.nan)),
                        "pvalue": float(fit_hb.pvalues.get("z_eci:pe", np.nan)),
                        "n_obs": int(len(df)),
                        "n_countries": int(df["country_iso3_code"].nunique()),
                    }
                )
            except Exception:
                pass

            lo, hi = df[outcome_col].quantile([0.05, 0.95]).tolist()
            dtrim = df[(df[outcome_col] >= lo) & (df[outcome_col] <= hi)].copy()
            if not dtrim.empty:
                fit_tr = fit_clustered(f, dtrim)
                rows.append(
                    {
                        "sample": sample_name,
                        "hypothesis_id": hid,
                        "dv": dv_label,
                        "spec": "Trimmed_5_95_FE_cluster",
                        "coef": float(fit_tr.params.get("z_eci:pe", np.nan)),
                        "pvalue": float(fit_tr.pvalues.get("z_eci:pe", np.nan)),
                        "n_obs": int(fit_tr.nobs),
                        "n_countries": int(dtrim["country_iso3_code"].nunique()),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "confirmatory_robustness_layer.csv", index=False)
    return out


def build_hypothesis_matrix(
    primary_df: pd.DataFrame,
    mod_df: pd.DataFrame,
    prepl_df: pd.DataFrame,
    fdr_df: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for hid in ["H1", "H2", "H3"]:
        p = primary_df[primary_df["hypothesis_id"] == hid].copy()
        if p.empty:
            continue
        r = p.iloc[0]
        pre = prepl_df[prepl_df["hypothesis_id"] == hid].iloc[0]
        f = fdr_df[fdr_df["hypothesis_id"] == hid].iloc[0]
        sign_correct = int(float(r["coef"]) > 0)
        pre_pass = int(pre["pretrend_pass_p_gt_0p05"])
        plc_pass = int(pre["placebo_pass_p_gt_0p05"])
        supports = int(
            (sign_correct == 1)
            and (float(r["p_boot"]) < 0.05)
            and (float(f["fdr_bh_qvalue"]) < 0.05)
            and (pre_pass == 1)
            and (plc_pass == 1)
        )
        partial = int(
            (sign_correct == 1)
            and (
                (float(r["p_boot"]) < 0.10)
                or (float(r["p_cluster"]) < 0.10)
                or (float(f["fdr_bh_qvalue"]) < 0.10)
            )
            and (pre_pass == 1)
            and (plc_pass == 1)
        )
        status = "SUPPORTED" if supports else ("PARTIALLY_SUPPORTED" if partial else "NOT_SUPPORTED")
        rows.append(
            {
                "hypothesis_id": hid,
                "hypothesis_label": r["dv"],
                "n_tests": 1,
                "sign_correct_count": sign_correct,
                "raw_sig_count": int(float(r["p_boot"]) < 0.05),
                "fdr_sig_count": int(float(f["fdr_bh_qvalue"]) < 0.05),
                "pretrend_pass_all": pre_pass,
                "placebo_pass_all": plc_pass,
                "key_pvalue": float(r["p_boot"]),
                "key_qvalue": float(f["fdr_bh_qvalue"]),
                "status": status,
            }
        )

    for hid in ["H4", "H5"]:
        sub = mod_df[mod_df["hypothesis_id"] == hid].copy()
        if sub.empty:
            continue
        merged = sub.merge(
            fdr_df[["hypothesis_id", "dv", "fdr_bh_qvalue"]],
            on=["hypothesis_id", "dv"],
            how="left",
        )
        sign_ok = merged["coef"] > 0 if hid == "H4" else merged["coef"] < 0
        raw_sig = (merged["p_cluster"] < 0.05) & sign_ok
        fdr_sig = (merged["fdr_bh_qvalue"] < 0.05) & sign_ok

        pre_subset = prepl_df[prepl_df["hypothesis_id"].isin(["H1", "H2", "H3"])]
        pre_pass_all = int((pre_subset["pretrend_pass_p_gt_0p05"] == 1).all())
        plc_pass_all = int((pre_subset["placebo_pass_p_gt_0p05"] == 1).all())

        supports = int((fdr_sig.sum() >= 2) and (pre_pass_all == 1) and (plc_pass_all == 1))
        partial = int((raw_sig.sum() >= 1) and (pre_pass_all == 1) and (plc_pass_all == 1))
        status = "SUPPORTED" if supports else ("PARTIALLY_SUPPORTED" if partial else "NOT_SUPPORTED")

        rows.append(
            {
                "hypothesis_id": hid,
                "hypothesis_label": "Moderation across resilience outcomes",
                "n_tests": int(len(merged)),
                "sign_correct_count": int(sign_ok.sum()),
                "raw_sig_count": int(raw_sig.sum()),
                "fdr_sig_count": int(fdr_sig.sum()),
                "pretrend_pass_all": pre_pass_all,
                "placebo_pass_all": plc_pass_all,
                "key_pvalue": float(merged["p_cluster"].min()),
                "key_qvalue": float(merged["fdr_bh_qvalue"].min()),
                "status": status,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "hypothesis_test_matrix.csv", index=False)
    return out


def write_summary(
    out_dir: Path,
    matrix_df: pd.DataFrame,
    include_rents_main: bool,
    gpr_col: str,
    bootstrap_reps: int,
) -> None:
    lines = [
        "# Confirmatory Hypothesis Test Pack (Profile-GPR Variant)",
        "",
        "Design:",
        "- H1-H3 tested via FE DiD with key term `z_eci:pe` and wild-cluster bootstrap p-values.",
        "- H4 tested via `z_eci:pe:z_coi` moderation.",
        "- H5 tested via `z_eci:pe:z_gpr` moderation.",
        f"- GPR column used: `{gpr_col}`.",
        f"- Main controls include rents: `{include_rents_main}`.",
        f"- Bootstrap replications: `{bootstrap_reps}`.",
        "- Multiple-testing correction uses Benjamini-Hochberg FDR across the confirmatory family.",
        "",
    ]

    if not matrix_df.empty:
        lines.append("## Hypothesis decisions")
        for _, r in matrix_df.iterrows():
            lines.append(
                f"- {r['hypothesis_id']}: {r['status']} | raw sig={int(r['raw_sig_count'])}, "
                f"FDR sig={int(r['fdr_sig_count'])}, pretrend_pass={int(r['pretrend_pass_all'])}, "
                f"placebo_pass={int(r['placebo_pass_all'])}"
            )
        lines.append("")

    lines.append("Output files:")
    lines.append("- `confirmatory_primary_tests.csv`")
    lines.append("- `confirmatory_moderation_tests.csv`")
    lines.append("- `confirmatory_pretrend_placebo.csv`")
    lines.append("- `confirmatory_multiple_testing.csv`")
    lines.append("- `confirmatory_robustness_layer.csv`")
    lines.append("- `hypothesis_test_matrix.csv`")
    (out_dir / "confirmatory_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run confirmatory hypothesis pack (H1-H5) with profile-based GPR imputation."
    )
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project base directory.",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=999)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--include-rents-main",
        type=int,
        default=0,
        choices=[0, 1],
        help="Include natural-resource-rents in main controls (1=yes, 0=no).",
    )
    parser.add_argument(
        "--out-subdir",
        default="confirmatory_hypotheses_profile_knn_999",
        help="Output subdirectory under reports.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=5,
        help="Number of nearest neighbors for profile GPR imputation.",
    )
    parser.add_argument(
        "--knn-min-common",
        type=int,
        default=2,
        help="Minimum common profile features for first-pass KNN match.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_dir = base_dir / "reports" / args.out_subdir
    ensure_dir(out_dir)

    panel = pd.read_csv(base_dir / "data" / "processed" / "regression_panel_2012_2022.csv")
    panel = panel[panel["year"].between(2012, 2022)].copy()

    panel, gpr_audit = apply_profile_knn_gpr_imputation(
        panel,
        k=int(args.knn_k),
        min_common_features=int(args.knn_min_common),
    )
    panel = add_engineered_columns(panel, gpr_col="gpr_profile_knn_annual")
    panel = add_size_quartile(panel)

    include_rents_main = bool(args.include_rents_main)

    primary_df = run_primary_tests(
        panel,
        out_dir=out_dir,
        bootstrap_reps=int(args.bootstrap_reps),
        seed=int(args.seed),
        include_rents_main=include_rents_main,
    )
    mod_df = run_moderation_tests(
        panel,
        out_dir=out_dir,
        include_rents_main=include_rents_main,
    )
    prepl_df = run_pretrend_and_placebo(
        panel,
        out_dir=out_dir,
        include_rents_main=include_rents_main,
    )
    fdr_df = run_fdr_family(primary_df, mod_df, out_dir=out_dir)
    robustness_df = run_robustness_layer(
        panel,
        out_dir=out_dir,
        include_rents_main=include_rents_main,
    )
    matrix_df = build_hypothesis_matrix(primary_df, mod_df, prepl_df, fdr_df, out_dir=out_dir)

    gpr_audit.to_csv(out_dir / "gpr_profile_knn_audit_by_year_method.csv", index=False)
    panel[
        [
            "country_iso3_code",
            "year",
            "gpr_country_annual",
            "gpr_profile_knn_annual",
            "gpr_profile_knn_method",
            "gpr_profile_knn_neighbors",
            "gpr_profile_knn_avg_common_features",
        ]
    ].to_csv(out_dir / "gpr_profile_knn_panel_extract.csv", index=False)

    write_summary(
        out_dir=out_dir,
        matrix_df=matrix_df,
        include_rents_main=include_rents_main,
        gpr_col="gpr_profile_knn_annual",
        bootstrap_reps=int(args.bootstrap_reps),
    )

    meta = {
        "panel_rows": int(len(panel)),
        "panel_countries": int(panel["country_iso3_code"].nunique()),
        "bootstrap_reps": int(args.bootstrap_reps),
        "include_rents_main": include_rents_main,
        "gpr_column": "gpr_profile_knn_annual",
        "gpr_country_missing_before": int(panel["gpr_country_annual"].isna().sum()),
        "gpr_profile_missing_after": int(panel["gpr_profile_knn_annual"].isna().sum()),
        "gpr_profile_methods": panel["gpr_profile_knn_method"].value_counts(dropna=False).to_dict(),
        "knn_k": int(args.knn_k),
        "knn_min_common_first_pass": int(args.knn_min_common),
        "primary_tests": int(len(primary_df)),
        "moderation_tests": int(len(mod_df)),
        "fdr_tests": int(len(fdr_df)),
        "robustness_rows": int(len(robustness_df)),
        "hypotheses_in_matrix": int(len(matrix_df)),
    }
    (out_dir / "confirmatory_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Confirmatory profile-GPR hypothesis pack completed.")
    print(f"Output directory: {out_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
