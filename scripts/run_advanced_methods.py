from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, export_text


DV_MAP = {
    "DV1_GVC_Linkage_Change": "delta_tiva_fexgr_dva_share",
    "DV2_Export_Recovery": "export_recovery_index",
    "DV3_Partner_Diversification": "partner_diversification_1_minus_hhi",
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


def prepare_panel(panel: pd.DataFrame, dv_col: str) -> pd.DataFrame:
    req = [dv_col, "eci", "post_paper", "exposed", "country_iso3_code", "year"] + CONTROL_COLS
    df = panel.dropna(subset=req).copy()
    df["pe"] = df["post_paper"] * df["exposed"]
    df["z_eci"] = zscore(df["eci"])
    df["z_log_gdp_pc"] = zscore(df["log_gdp_pc"])
    df["z_trade_open"] = zscore(df["wdi_trade_openness_pct_gdp"])
    df["z_wgi"] = zscore(df["wgi_institutional_quality_composite"])
    df["z_rents"] = zscore(df["wdi_natural_resource_rents_pct_gdp"])
    df["z_gpr"] = zscore(df["gpr_for_model_annual"])
    df["z_intensity"] = zscore(df["us_china_trade_intensity_pre"])
    return df


def full_formula(dv_col: str) -> str:
    return (
        f"{dv_col} ~ z_eci + pe + z_eci:pe + z_log_gdp_pc + z_trade_open + z_wgi + "
        "z_rents + z_gpr + z_intensity + C(country_iso3_code) + C(year)"
    )


def restricted_formula(dv_col: str) -> str:
    return (
        f"{dv_col} ~ z_eci + pe + z_log_gdp_pc + z_trade_open + z_wgi + "
        "z_rents + z_gpr + z_intensity + C(country_iso3_code) + C(year)"
    )


def fit_main(df: pd.DataFrame, dv_col: str) -> Any:
    return smf.ols(formula=full_formula(dv_col), data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["country_iso3_code"]},
    )


def run_wild_cluster_bootstrap(
    df: pd.DataFrame,
    dv_col: str,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    term = "z_eci:pe"
    rng = np.random.default_rng(seed)

    fit_full = fit_main(df, dv_col)
    coef_obs = float(fit_full.params.get(term, np.nan))
    se_obs = float(fit_full.bse.get(term, np.nan))
    t_obs = coef_obs / se_obs if pd.notna(coef_obs) and pd.notna(se_obs) and se_obs != 0 else np.nan

    fit_restricted = smf.ols(formula=restricted_formula(dv_col), data=df).fit()
    yhat0 = fit_restricted.fittedvalues.to_numpy()
    u0 = fit_restricted.resid.to_numpy()

    clusters = df["country_iso3_code"].astype(str).to_numpy()
    unique_clusters = np.unique(clusters)

    coef_star: list[float] = []
    t_star: list[float] = []
    dv_star = f"{dv_col}__star"
    formula_star = full_formula(dv_col).replace(dv_col, dv_star, 1)
    boot_df = df.copy()

    for _ in range(reps):
        draws = rng.choice([-1.0, 1.0], size=len(unique_clusters))
        w_map = dict(zip(unique_clusters, draws))
        weights = np.array([w_map[c] for c in clusters], dtype=float)
        boot_df[dv_star] = yhat0 + u0 * weights
        try:
            fit_b = smf.ols(formula=formula_star, data=boot_df).fit(
                cov_type="cluster",
                cov_kwds={"groups": boot_df["country_iso3_code"]},
            )
            c = float(fit_b.params.get(term, np.nan))
            s = float(fit_b.bse.get(term, np.nan))
            if pd.notna(c) and pd.notna(s) and s != 0:
                coef_star.append(c)
                t_star.append(c / s)
        except Exception:
            continue

    coef_arr = np.array(coef_star, dtype=float) if coef_star else np.array([np.nan])
    t_arr = np.array(t_star, dtype=float) if t_star else np.array([np.nan])
    if np.isfinite(t_obs) and np.isfinite(t_arr).any():
        p_boot = float(np.mean(np.abs(t_arr[np.isfinite(t_arr)]) >= abs(t_obs)))
    else:
        p_boot = np.nan

    ci_low = float(np.nanpercentile(coef_arr, 2.5)) if np.isfinite(coef_arr).any() else np.nan
    ci_high = float(np.nanpercentile(coef_arr, 97.5)) if np.isfinite(coef_arr).any() else np.nan

    return {
        "coef_obs": coef_obs,
        "se_obs": se_obs,
        "p_obs_cluster": float(fit_full.pvalues.get(term, np.nan)),
        "t_obs": t_obs,
        "bootstrap_reps_requested": int(reps),
        "bootstrap_reps_success": int(np.isfinite(t_arr).sum()),
        "p_boot_wild_cluster": p_boot,
        "coef_boot_ci_low_2p5": ci_low,
        "coef_boot_ci_high_97p5": ci_high,
    }


def run_bootstrap_suite(panel: pd.DataFrame, out_dir: Path, reps: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, (dv_name, dv_col) in enumerate(DV_MAP.items()):
        df = prepare_panel(panel, dv_col)
        if df.empty:
            continue
        result = run_wild_cluster_bootstrap(df, dv_col, reps=reps, seed=seed + i)
        result.update(
            {
                "dv": dv_name,
                "n_obs": int(len(df)),
                "n_countries": int(df["country_iso3_code"].nunique()),
                "year_min": int(df["year"].min()),
                "year_max": int(df["year"].max()),
            }
        )
        rows.append(result)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "wild_cluster_bootstrap_results.csv", index=False)
    return out


def run_robust_fe_suite(panel: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    term = "z_eci:pe"

    for dv_name, dv_col in DV_MAP.items():
        df = prepare_panel(panel, dv_col)
        if df.empty:
            continue

        # Baseline FE OLS with clustered SEs
        fit_ols = fit_main(df, dv_col)
        rows.append(
            {
                "dv": dv_name,
                "spec": "OLS_FE_cluster",
                "coef_z_eci_pe": float(fit_ols.params.get(term, np.nan)),
                "se": float(fit_ols.bse.get(term, np.nan)),
                "pvalue": float(fit_ols.pvalues.get(term, np.nan)),
                "n_obs": int(fit_ols.nobs),
                "n_countries": int(df["country_iso3_code"].nunique()),
            }
        )

        # Robust M-estimator with Huber loss and FE dummies
        try:
            fit_rlm = smf.rlm(
                formula=full_formula(dv_col),
                data=df,
                M=sm.robust.norms.HuberT(),
            ).fit()
            rows.append(
                {
                    "dv": dv_name,
                    "spec": "RLM_Huber_FE",
                    "coef_z_eci_pe": float(fit_rlm.params.get(term, np.nan)),
                    "se": float(fit_rlm.bse.get(term, np.nan)),
                    "pvalue": float(fit_rlm.pvalues.get(term, np.nan)),
                    "n_obs": int(len(df)),
                    "n_countries": int(df["country_iso3_code"].nunique()),
                }
            )
        except Exception:
            rows.append(
                {
                    "dv": dv_name,
                    "spec": "RLM_Huber_FE",
                    "coef_z_eci_pe": np.nan,
                    "se": np.nan,
                    "pvalue": np.nan,
                    "n_obs": int(len(df)),
                    "n_countries": int(df["country_iso3_code"].nunique()),
                }
            )

        # Trimmed-mean style FE: drop 5% tails of DV and refit.
        lo, hi = df[dv_col].quantile([0.05, 0.95]).tolist()
        dtrim = df[(df[dv_col] >= lo) & (df[dv_col] <= hi)].copy()
        if not dtrim.empty:
            fit_trim = fit_main(dtrim, dv_col)
            rows.append(
                {
                    "dv": dv_name,
                    "spec": "Trimmed_5_95_FE_cluster",
                    "coef_z_eci_pe": float(fit_trim.params.get(term, np.nan)),
                    "se": float(fit_trim.bse.get(term, np.nan)),
                    "pvalue": float(fit_trim.pvalues.get(term, np.nan)),
                    "n_obs": int(fit_trim.nobs),
                    "n_countries": int(dtrim["country_iso3_code"].nunique()),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "robust_fe_results.csv", index=False)
    return out


def run_causal_heterogeneity(panel: pd.DataFrame, out_dir: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    by_group_rows: list[dict[str, Any]] = []
    country_rows: list[dict[str, Any]] = []

    for i, (dv_name, dv_col) in enumerate(DV_MAP.items()):
        df = prepare_panel(panel, dv_col)
        if df.empty:
            continue

        # T-learner with random forest:
        # treated group is post*exposed (pe), controls are covariates + year.
        feats = [
            "eci",
            "log_gdp_pc",
            "wdi_trade_openness_pct_gdp",
            "wgi_institutional_quality_composite",
            "wdi_natural_resource_rents_pct_gdp",
            "gpr_for_model_annual",
            "us_china_trade_intensity_pre",
            "year",
        ]
        tcol = "pe"
        req = feats + [tcol, dv_col, "country_iso3_code", "baseline_export_2015_2017"]
        d = df.dropna(subset=req).copy()
        if d.empty:
            continue

        treated = d[d[tcol] == 1].copy()
        control = d[d[tcol] == 0].copy()
        if treated.empty or control.empty:
            continue

        x_t = treated[feats].to_numpy()
        y_t = treated[dv_col].to_numpy()
        x_c = control[feats].to_numpy()
        y_c = control[dv_col].to_numpy()

        rf_t = RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=20,
            random_state=seed + i,
            n_jobs=1,
        )
        rf_c = RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=20,
            random_state=seed + 100 + i,
            n_jobs=1,
        )
        rf_t.fit(x_t, y_t)
        rf_c.fit(x_c, y_c)

        x_all = d[feats].to_numpy()
        mu1 = rf_t.predict(x_all)
        mu0 = rf_c.predict(x_all)
        d["cate_tlearner"] = mu1 - mu0

        summary_rows.append(
            {
                "dv": dv_name,
                "n_obs": int(len(d)),
                "n_countries": int(d["country_iso3_code"].nunique()),
                "cate_mean": float(d["cate_tlearner"].mean()),
                "cate_std": float(d["cate_tlearner"].std(ddof=0)),
                "cate_p25": float(d["cate_tlearner"].quantile(0.25)),
                "cate_p50": float(d["cate_tlearner"].quantile(0.50)),
                "cate_p75": float(d["cate_tlearner"].quantile(0.75)),
                "share_cate_positive": float((d["cate_tlearner"] > 0).mean()),
            }
        )

        # Group summaries by ECI tercile.
        d["eci_tercile"] = pd.qcut(d["eci"], q=3, labels=["low", "mid", "high"], duplicates="drop")
        g1 = (
            d.groupby("eci_tercile", as_index=False)["cate_tlearner"]
            .mean()
            .rename(columns={"cate_tlearner": "cate_mean"})
        )
        for _, row in g1.iterrows():
            by_group_rows.append(
                {
                    "dv": dv_name,
                    "group_type": "eci_tercile",
                    "group_name": str(row["eci_tercile"]),
                    "cate_mean": float(row["cate_mean"]),
                }
            )

        # Group summaries by export-size quartile.
        d["size_quartile"] = pd.qcut(
            d["baseline_export_2015_2017"],
            q=4,
            labels=["Q1_smallest", "Q2", "Q3", "Q4_largest"],
            duplicates="drop",
        )
        g2 = (
            d.groupby("size_quartile", as_index=False)["cate_tlearner"]
            .mean()
            .rename(columns={"cate_tlearner": "cate_mean"})
        )
        for _, row in g2.iterrows():
            by_group_rows.append(
                {
                    "dv": dv_name,
                    "group_type": "size_quartile",
                    "group_name": str(row["size_quartile"]),
                    "cate_mean": float(row["cate_mean"]),
                }
            )

        # Country-level ranking.
        cdf = (
            d.groupby("country_iso3_code", as_index=False)["cate_tlearner"]
            .mean()
            .rename(columns={"cate_tlearner": "cate_mean_country"})
            .sort_values("cate_mean_country")
        )
        cdf["dv"] = dv_name
        country_rows.append(cdf)

        # Policy-tree style segmentation on estimated CATE.
        try:
            tree = DecisionTreeRegressor(
                max_depth=3,
                min_samples_leaf=max(40, int(0.05 * len(d))),
                random_state=seed + 200 + i,
            )
            tree.fit(d[feats].to_numpy(), d["cate_tlearner"].to_numpy())
            rules = export_text(tree, feature_names=feats)
            (out_dir / f"policy_tree_rules_{dv_name}.txt").write_text(rules, encoding="utf-8")
        except Exception:
            pass

    summary_df = pd.DataFrame(summary_rows)
    by_group_df = pd.DataFrame(by_group_rows)
    country_df = pd.concat(country_rows, ignore_index=True) if country_rows else pd.DataFrame()

    summary_df.to_csv(out_dir / "causal_heterogeneity_summary.csv", index=False)
    by_group_df.to_csv(out_dir / "causal_heterogeneity_group_means.csv", index=False)
    country_df.to_csv(out_dir / "causal_heterogeneity_country_rank.csv", index=False)
    return summary_df, by_group_df, country_df


def write_summary(
    out_dir: Path,
    bootstrap_df: pd.DataFrame,
    robust_df: pd.DataFrame,
    cate_summary_df: pd.DataFrame,
) -> None:
    lines = [
        "# Advanced Methods Results",
        "",
        "Implemented methods:",
        "- Wild cluster bootstrap (Rademacher) for FE coefficient `z_eci:pe`.",
        "- Robust FE checks: Huber RLM and trimmed (5-95) FE.",
        "- Causal heterogeneity exploration via T-learner random forests and policy-tree segmentation.",
        "",
    ]

    if not bootstrap_df.empty:
        lines.append("## Wild Cluster Bootstrap")
        for _, r in bootstrap_df.iterrows():
            lines.append(
                f"- {r['dv']}: coef={r['coef_obs']:.4f}, cluster-p={r['p_obs_cluster']:.4g}, "
                f"wild-bootstrap p={r['p_boot_wild_cluster']:.4g}, reps={int(r['bootstrap_reps_success'])}"
            )
        lines.append("")

    if not robust_df.empty:
        lines.append("## Robust FE")
        for dv in robust_df["dv"].dropna().unique():
            sub = robust_df[robust_df["dv"] == dv]
            parts = []
            for _, r in sub.iterrows():
                parts.append(f"{r['spec']}={r['coef_z_eci_pe']:.4f}(p={r['pvalue']:.4g})")
            lines.append(f"- {dv}: " + "; ".join(parts))
        lines.append("")

    if not cate_summary_df.empty:
        lines.append("## Causal Heterogeneity (T-learner)")
        for _, r in cate_summary_df.iterrows():
            lines.append(
                f"- {r['dv']}: mean CATE={r['cate_mean']:.4f}, "
                f"p25/p50/p75=({r['cate_p25']:.4f}, {r['cate_p50']:.4f}, {r['cate_p75']:.4f}), "
                f"share positive={r['share_cate_positive']:.2%}"
            )
        lines.append("")

    lines.extend(
        [
            "Caution:",
            "- T-learner CATE outputs are exploratory and rely on unconfoundedness assumptions.",
            "- Use these to guide segmentation and mechanism discussion, not as definitive causal proof alone.",
        ]
    )
    (out_dir / "advanced_methods_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run advanced robustness and heterogeneity methods.")
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project base directory.",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=59,
        help="Wild cluster bootstrap replications per DV.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_dir = base_dir / "reports" / "advanced_methods"
    ensure_dir(out_dir)

    panel = pd.read_csv(base_dir / "data" / "processed" / "regression_panel_2012_2022.csv")
    panel = panel[panel["year"].between(2012, 2022)].copy()
    panel["post_paper"] = panel["year"].between(2019, 2022).astype(int)

    bootstrap_df = run_bootstrap_suite(
        panel=panel,
        out_dir=out_dir,
        reps=args.bootstrap_reps,
        seed=args.seed,
    )
    robust_df = run_robust_fe_suite(panel=panel, out_dir=out_dir)
    cate_summary_df, cate_group_df, cate_country_df = run_causal_heterogeneity(
        panel=panel,
        out_dir=out_dir,
        seed=args.seed,
    )

    meta = {
        "panel_rows": int(len(panel)),
        "panel_countries": int(panel["country_iso3_code"].nunique()),
        "bootstrap_rows": int(len(bootstrap_df)),
        "robust_rows": int(len(robust_df)),
        "cate_summary_rows": int(len(cate_summary_df)),
        "cate_group_rows": int(len(cate_group_df)),
        "cate_country_rows": int(len(cate_country_df)),
        "bootstrap_reps": int(args.bootstrap_reps),
    }
    (out_dir / "advanced_methods_metadata.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    write_summary(out_dir, bootstrap_df, robust_df, cate_summary_df)

    print("Advanced methods completed.")
    print(f"Output directory: {out_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
