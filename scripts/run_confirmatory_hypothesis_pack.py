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

CONTROL_COLS = [
    "log_gdp_pc",
    "wdi_trade_openness_pct_gdp",
    "wgi_institutional_quality_composite",
    "wdi_natural_resource_rents_pct_gdp",
    "gpr_for_model_annual",
    "us_china_trade_intensity_pre",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def zscore(series: pd.Series) -> pd.Series:
    mu = series.mean(skipna=True)
    sigma = series.std(skipna=True, ddof=0)
    if pd.isna(sigma) or sigma == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - mu) / sigma


def add_engineered_columns(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    # H1 needs a stability measure where higher means more stable.
    # We transform change to inverse absolute disruption.
    df["gvc_linkage_stability"] = -df["delta_tiva_fexgr_dva_share"].abs()
    df["post_paper"] = df["year"].between(2019, 2022).astype(int)
    df["pe"] = df["post_paper"] * df["exposed"]

    for col, zcol in [
        ("eci", "z_eci"),
        ("coi", "z_coi"),
        ("gpr_for_model_annual", "z_gpr"),
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
    csize["size_q"] = pd.qcut(csize["baseline_export_2015_2017"], q=4, labels=[1, 2, 3, 4], duplicates="drop")
    return panel.merge(csize[["country_iso3_code", "size_q"]], on="country_iso3_code", how="left")


def prepare_sample(panel: pd.DataFrame, outcome_col: str, sample_name: str) -> pd.DataFrame:
    df = panel.copy()
    if sample_name == "core_excl_q1":
        df = df[df["size_q"] != 1].copy()
    req = [outcome_col, "z_eci", "pe", "country_iso3_code", "year"] + [
        "z_log_gdp_pc",
        "z_trade_open",
        "z_wgi",
        "z_rents",
        "z_gpr",
        "z_intensity",
    ]
    return df.dropna(subset=req).copy()


def primary_formula(outcome_col: str, pe_col: str = "pe") -> str:
    return (
        f"{outcome_col} ~ z_eci + {pe_col} + z_eci:{pe_col} + "
        "z_log_gdp_pc + z_trade_open + z_wgi + z_rents + z_gpr + z_intensity + "
        "C(country_iso3_code) + C(year)"
    )


def moderation_formula(outcome_col: str, moderator: str) -> str:
    if moderator == "coi":
        return (
            f"{outcome_col} ~ z_eci*pe*z_coi + "
            "z_log_gdp_pc + z_trade_open + z_wgi + z_rents + z_gpr + z_intensity + "
            "C(country_iso3_code) + C(year)"
        )
    # moderator == gpr
    return (
        f"{outcome_col} ~ z_eci*pe*z_gpr + "
        "z_log_gdp_pc + z_trade_open + z_wgi + z_rents + z_intensity + "
        "C(country_iso3_code) + C(year)"
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
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    f_full = primary_formula(outcome_col)
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


def run_primary_tests(panel: pd.DataFrame, out_dir: Path, bootstrap_reps: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, (hid, (dv_label, outcome_col)) in enumerate(OUTCOME_MAP.items()):
        df = prepare_sample(panel, outcome_col, sample_name="full")
        if df.empty:
            continue
        fit = fit_clustered(primary_formula(outcome_col), df)
        term = "z_eci:pe"
        coef = float(fit.params.get(term, np.nan))
        p = float(fit.pvalues.get(term, np.nan))

        boot = run_wild_cluster_bootstrap_term(
            df=df,
            outcome_col=outcome_col,
            target_term=term,
            reps=bootstrap_reps,
            seed=seed + i,
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


def run_moderation_tests(panel: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for hid, (dv_label, outcome_col) in OUTCOME_MAP.items():
        df = prepare_sample(panel, outcome_col, sample_name="full")
        if df.empty:
            continue

        # H4
        fit_h4 = fit_clustered(moderation_formula(outcome_col, "coi"), df.dropna(subset=["z_coi"]).copy())
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

        # H5
        fit_h5 = fit_clustered(moderation_formula(outcome_col, "gpr"), df)
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


def run_pretrend_and_placebo(panel: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for hid, (dv_label, outcome_col) in OUTCOME_MAP.items():
        df = prepare_sample(panel, outcome_col, sample_name="full")
        if df.empty:
            continue

        # Event-study pretrend test.
        est = df.copy()
        est["event_time"] = (est["year"] - 2018).astype(int)
        est = est[est["event_time"].between(-6, 4)].copy()
        f_es = (
            f"{outcome_col} ~ z_eci + z_log_gdp_pc + z_trade_open + z_wgi + z_rents + z_gpr + z_intensity + "
            "C(country_iso3_code) + C(year) + C(event_time, Treatment(reference=-1)):exposed:z_eci"
        )
        fit_es = fit_clustered(f_es, est)
        pre_terms = [
            f"C(event_time, Treatment(reference=-1))[{k}]:exposed:z_eci"
            for k in [-6, -5, -4, -3, -2]
            if f"C(event_time, Treatment(reference=-1))[{k}]:exposed:z_eci" in fit_es.params.index
        ]
        if pre_terms:
            w = fit_es.wald_test(", ".join([f"{t}=0" for t in pre_terms]))
            pre_p = float(w.pvalue)
        else:
            pre_p = np.nan

        # Placebo test (fake post in 2014-2015).
        plc = df.copy()
        plc["post_placebo"] = plc["year"].between(2014, 2015).astype(int)
        plc["pe_placebo"] = plc["post_placebo"] * plc["exposed"]
        f_pl = primary_formula(outcome_col, pe_col="pe_placebo")
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
                "pvalue_for_fdr": r["p_boot"],  # bootstrap as primary inference
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


def run_robustness_layer(panel: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_name in ["full", "core_excl_q1"]:
        for hid, (dv_label, outcome_col) in OUTCOME_MAP.items():
            df = prepare_sample(panel, outcome_col, sample_name=sample_name)
            if df.empty:
                continue
            # OLS clustered
            fit_ols = fit_clustered(primary_formula(outcome_col), df)
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

            # Huber
            try:
                fit_hb = smf.rlm(primary_formula(outcome_col), data=df, M=sm.robust.norms.HuberT()).fit()
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

            # Trimmed
            lo, hi = df[outcome_col].quantile([0.05, 0.95]).tolist()
            dtrim = df[(df[outcome_col] >= lo) & (df[outcome_col] <= hi)].copy()
            if not dtrim.empty:
                fit_tr = fit_clustered(primary_formula(outcome_col), dtrim)
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

    # H1-H3
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

    # H4 and H5 aggregate across outcomes.
    for hid in ["H4", "H5"]:
        sub = mod_df[mod_df["hypothesis_id"] == hid].copy()
        if sub.empty:
            continue
        merged = sub.merge(
            fdr_df[["hypothesis_id", "dv", "fdr_bh_qvalue"]],
            on=["hypothesis_id", "dv"],
            how="left",
        )
        if hid == "H4":
            sign_ok = merged["coef"] > 0
        else:
            sign_ok = merged["coef"] < 0
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
    primary_df: pd.DataFrame,
    mod_df: pd.DataFrame,
    prepl_df: pd.DataFrame,
    fdr_df: pd.DataFrame,
) -> None:
    lines = [
        "# Confirmatory Hypothesis Test Pack",
        "",
        "Design:",
        "- H1-H3 tested via FE DiD with key term `z_eci:pe` and wild-cluster bootstrap p-values.",
        "- H4 tested via `z_eci:pe:z_coi` moderation.",
        "- H5 tested via `z_eci:pe:z_gpr` moderation.",
        "- Pretrend and placebo checks included for outcome-level identification diagnostics.",
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
    parser = argparse.ArgumentParser(description="Run confirmatory hypothesis pack (H1-H5).")
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project base directory.",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=299)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_dir = base_dir / "reports" / "confirmatory_hypotheses"
    ensure_dir(out_dir)

    panel = pd.read_csv(base_dir / "data" / "processed" / "regression_panel_2012_2022.csv")
    panel = panel[panel["year"].between(2012, 2022)].copy()
    panel = add_engineered_columns(panel)
    panel = add_size_quartile(panel)

    primary_df = run_primary_tests(panel, out_dir=out_dir, bootstrap_reps=args.bootstrap_reps, seed=args.seed)
    mod_df = run_moderation_tests(panel, out_dir=out_dir)
    prepl_df = run_pretrend_and_placebo(panel, out_dir=out_dir)
    fdr_df = run_fdr_family(primary_df, mod_df, out_dir=out_dir)
    robustness_df = run_robustness_layer(panel, out_dir=out_dir)
    matrix_df = build_hypothesis_matrix(primary_df, mod_df, prepl_df, fdr_df, out_dir=out_dir)

    write_summary(out_dir, matrix_df, primary_df, mod_df, prepl_df, fdr_df)

    meta = {
        "panel_rows": int(len(panel)),
        "panel_countries": int(panel["country_iso3_code"].nunique()),
        "bootstrap_reps": int(args.bootstrap_reps),
        "primary_tests": int(len(primary_df)),
        "moderation_tests": int(len(mod_df)),
        "fdr_tests": int(len(fdr_df)),
        "robustness_rows": int(len(robustness_df)),
        "hypotheses_in_matrix": int(len(matrix_df)),
    }
    (out_dir / "confirmatory_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Confirmatory hypothesis pack completed.")
    print(f"Output directory: {out_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
