from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


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


def run_dv2_high_precision_bootstrap(
    panel: pd.DataFrame,
    out_dir: Path,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    dv_col = "export_recovery_index"
    term = "z_eci:pe"
    df = prepare_panel(panel, dv_col)
    if df.empty:
        out = pd.DataFrame()
        out.to_csv(out_dir / "dv2_high_precision_bootstrap.csv", index=False)
        return out

    rng = np.random.default_rng(seed)

    fit_full = smf.ols(formula=full_formula(dv_col), data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["country_iso3_code"]},
    )
    coef_obs = float(fit_full.params.get(term, np.nan))
    se_obs = float(fit_full.bse.get(term, np.nan))
    t_obs = coef_obs / se_obs if pd.notna(coef_obs) and pd.notna(se_obs) and se_obs != 0 else np.nan

    fit_restricted = smf.ols(formula=restricted_formula(dv_col), data=df).fit()
    yhat0 = fit_restricted.fittedvalues.to_numpy()
    u0 = fit_restricted.resid.to_numpy()

    clusters = df["country_iso3_code"].astype(str).to_numpy()
    unique_clusters = np.unique(clusters)
    dv_star = f"{dv_col}__star"
    formula_star = full_formula(dv_col).replace(dv_col, dv_star, 1)
    boot_df = df.copy()

    coef_star: list[float] = []
    t_star: list[float] = []
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
    p_boot = (
        float(np.mean(np.abs(t_arr[np.isfinite(t_arr)]) >= abs(t_obs)))
        if np.isfinite(t_obs) and np.isfinite(t_arr).any()
        else np.nan
    )

    result = pd.DataFrame(
        [
            {
                "dv": "DV2_Export_Recovery",
                "coef_obs": coef_obs,
                "se_obs": se_obs,
                "p_obs_cluster": float(fit_full.pvalues.get(term, np.nan)),
                "t_obs": t_obs,
                "bootstrap_reps_requested": int(reps),
                "bootstrap_reps_success": int(np.isfinite(t_arr).sum()),
                "p_boot_wild_cluster": p_boot,
                "coef_boot_ci_low_2p5": float(np.nanpercentile(coef_arr, 2.5))
                if np.isfinite(coef_arr).any()
                else np.nan,
                "coef_boot_ci_high_97p5": float(np.nanpercentile(coef_arr, 97.5))
                if np.isfinite(coef_arr).any()
                else np.nan,
                "n_obs": int(len(df)),
                "n_countries": int(df["country_iso3_code"].nunique()),
                "year_min": int(df["year"].min()),
                "year_max": int(df["year"].max()),
            }
        ]
    )
    result.to_csv(out_dir / "dv2_high_precision_bootstrap.csv", index=False)
    return result


def run_mediation(
    panel: pd.DataFrame,
    out_dir: Path,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    # Mediation chain:
    # X = z_eci:pe
    # M = export_recovery_index
    # Y = partner_diversification_1_minus_hhi
    # indirect = a * b
    req = [
        "eci",
        "post_paper",
        "exposed",
        "country_iso3_code",
        "year",
        "export_recovery_index",
        "partner_diversification_1_minus_hhi",
    ] + CONTROL_COLS
    df = panel.dropna(subset=req).copy()
    if df.empty:
        out = pd.DataFrame()
        out.to_csv(out_dir / "mediation_results.csv", index=False)
        return out

    df["pe"] = df["post_paper"] * df["exposed"]
    df["z_eci"] = zscore(df["eci"])
    df["z_log_gdp_pc"] = zscore(df["log_gdp_pc"])
    df["z_trade_open"] = zscore(df["wdi_trade_openness_pct_gdp"])
    df["z_wgi"] = zscore(df["wgi_institutional_quality_composite"])
    df["z_rents"] = zscore(df["wdi_natural_resource_rents_pct_gdp"])
    df["z_gpr"] = zscore(df["gpr_for_model_annual"])
    df["z_intensity"] = zscore(df["us_china_trade_intensity_pre"])
    df["z_mediator"] = zscore(df["export_recovery_index"])

    f_a = (
        "export_recovery_index ~ z_eci + pe + z_eci:pe + z_log_gdp_pc + z_trade_open + z_wgi + "
        "z_rents + z_gpr + z_intensity + C(country_iso3_code) + C(year)"
    )
    f_b = (
        "partner_diversification_1_minus_hhi ~ z_mediator + z_eci + pe + z_eci:pe + "
        "z_log_gdp_pc + z_trade_open + z_wgi + z_rents + z_gpr + z_intensity + "
        "C(country_iso3_code) + C(year)"
    )
    f_t = (
        "partner_diversification_1_minus_hhi ~ z_eci + pe + z_eci:pe + z_log_gdp_pc + z_trade_open + "
        "z_wgi + z_rents + z_gpr + z_intensity + C(country_iso3_code) + C(year)"
    )

    fit_a = smf.ols(f_a, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["country_iso3_code"]})
    fit_b = smf.ols(f_b, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["country_iso3_code"]})
    fit_t = smf.ols(f_t, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["country_iso3_code"]})

    a = float(fit_a.params.get("z_eci:pe", np.nan))
    b = float(fit_b.params.get("z_mediator", np.nan))
    direct = float(fit_b.params.get("z_eci:pe", np.nan))
    total = float(fit_t.params.get("z_eci:pe", np.nan))
    indirect = a * b if pd.notna(a) and pd.notna(b) else np.nan

    # Cluster bootstrap for indirect effect.
    rng = np.random.default_rng(seed)
    clusters = df["country_iso3_code"].astype(str).unique().tolist()
    indirect_star: list[float] = []
    direct_star: list[float] = []
    total_star: list[float] = []

    for _ in range(reps):
        draw = rng.choice(clusters, size=len(clusters), replace=True)
        chunk = []
        for c in draw:
            dsub = df[df["country_iso3_code"] == c].copy()
            dsub["country_boot"] = f"{c}_{len(chunk)}"
            chunk.append(dsub)
        bs = pd.concat(chunk, ignore_index=True)
        # Preserve fixed-effects structure in bootstrap sample via boot cluster id.
        f_a_bs = f_a.replace("C(country_iso3_code)", "C(country_boot)")
        f_b_bs = f_b.replace("C(country_iso3_code)", "C(country_boot)")
        f_t_bs = f_t.replace("C(country_iso3_code)", "C(country_boot)")
        try:
            a_bs = float(smf.ols(f_a_bs, data=bs).fit().params.get("z_eci:pe", np.nan))
            b_bs = float(smf.ols(f_b_bs, data=bs).fit().params.get("z_mediator", np.nan))
            d_bs = float(smf.ols(f_b_bs, data=bs).fit().params.get("z_eci:pe", np.nan))
            t_bs = float(smf.ols(f_t_bs, data=bs).fit().params.get("z_eci:pe", np.nan))
            if pd.notna(a_bs) and pd.notna(b_bs):
                indirect_star.append(a_bs * b_bs)
            if pd.notna(d_bs):
                direct_star.append(d_bs)
            if pd.notna(t_bs):
                total_star.append(t_bs)
        except Exception:
            continue

    ind_arr = np.array(indirect_star, dtype=float) if indirect_star else np.array([np.nan])
    p_ind_boot = (
        float(2 * min(np.mean(ind_arr <= 0), np.mean(ind_arr >= 0)))
        if np.isfinite(ind_arr).any()
        else np.nan
    )

    out = pd.DataFrame(
        [
            {
                "path_a_coef_x_to_m": a,
                "path_a_pvalue": float(fit_a.pvalues.get("z_eci:pe", np.nan)),
                "path_b_coef_m_to_y": b,
                "path_b_pvalue": float(fit_b.pvalues.get("z_mediator", np.nan)),
                "direct_coef_x_to_y_cond_m": direct,
                "direct_pvalue": float(fit_b.pvalues.get("z_eci:pe", np.nan)),
                "total_coef_x_to_y": total,
                "total_pvalue": float(fit_t.pvalues.get("z_eci:pe", np.nan)),
                "indirect_coef_a_times_b": indirect,
                "indirect_boot_pvalue": p_ind_boot,
                "indirect_boot_ci_low_2p5": float(np.nanpercentile(ind_arr, 2.5))
                if np.isfinite(ind_arr).any()
                else np.nan,
                "indirect_boot_ci_high_97p5": float(np.nanpercentile(ind_arr, 97.5))
                if np.isfinite(ind_arr).any()
                else np.nan,
                "boot_reps_requested": int(reps),
                "boot_reps_success": int(np.isfinite(ind_arr).sum()),
                "n_obs": int(len(df)),
                "n_countries": int(df["country_iso3_code"].nunique()),
            }
        ]
    )
    out.to_csv(out_dir / "mediation_results.csv", index=False)
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


def collect_pvalues(base_dir: Path, high_boot_df: pd.DataFrame, mediation_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_from_df(
        df: pd.DataFrame,
        source: str,
        dv_col: str,
        test_col: str,
        p_col: str,
        coef_col: str | None = None,
    ) -> None:
        if df.empty:
            return
        for _, r in df.iterrows():
            p = r.get(p_col, np.nan)
            if pd.isna(p):
                continue
            rows.append(
                {
                    "source": source,
                    "dv": r.get(dv_col, ""),
                    "test_id": str(r.get(test_col, "")),
                    "coef": r.get(coef_col, np.nan) if coef_col else np.nan,
                    "pvalue_raw": float(p),
                }
            )

    # Main models
    add_from_df(
        pd.read_csv(base_dir / "reports" / "comprehensive_table2_4_main_models.csv"),
        source="main_models",
        dv_col="dv",
        test_col="model",
        p_col="p_key",
        coef_col="coef_key",
    )
    # Outlier robust models
    add_from_df(
        pd.read_csv(base_dir / "reports" / "outlier_diagnostics" / "outlier_robustness_reestimation.csv"),
        source="outlier_robust",
        dv_col="dv",
        test_col="spec",
        p_col="pvalue",
        coef_col="coef_z_eci_pe",
    )
    # Additional quantiles
    add_from_df(
        pd.read_csv(base_dir / "reports" / "additional_ideas" / "quantile_results.csv"),
        source="quantile",
        dv_col="dv",
        test_col="quantile",
        p_col="pvalue",
        coef_col="coef_z_eci_pe",
    )
    # Additional regime
    add_from_df(
        pd.read_csv(base_dir / "reports" / "additional_ideas" / "regime_eci_tercile_results.csv"),
        source="regime_eci",
        dv_col="dv",
        test_col="eci_regime",
        p_col="pvalue",
        coef_col="coef_z_eci_pe",
    )
    # Advanced robust FE
    add_from_df(
        pd.read_csv(base_dir / "reports" / "advanced_methods" / "robust_fe_results.csv"),
        source="advanced_robust_fe",
        dv_col="dv",
        test_col="spec",
        p_col="pvalue",
        coef_col="coef_z_eci_pe",
    )
    # Event-study joint tests
    add_from_df(
        pd.read_csv(base_dir / "reports" / "additional_ideas" / "event_study_joint_tests.csv"),
        source="event_study_joint",
        dv_col="dv",
        test_col="test",
        p_col="pvalue",
        coef_col=None,
    )
    # High precision bootstrap DV2
    add_from_df(
        high_boot_df.rename(columns={"p_boot_wild_cluster": "pvalue_boot", "coef_obs": "coef_main"}),
        source="high_precision_bootstrap",
        dv_col="dv",
        test_col="dv",
        p_col="pvalue_boot",
        coef_col="coef_main",
    )

    # Mediation p-values
    if not mediation_df.empty:
        m = mediation_df.iloc[0]
        rows.append(
            {
                "source": "mediation",
                "dv": "DV3_Partner_Diversification",
                "test_id": "indirect_effect_boot",
                "coef": float(m.get("indirect_coef_a_times_b", np.nan)),
                "pvalue_raw": float(m.get("indirect_boot_pvalue", np.nan)),
            }
        )
        rows.append(
            {
                "source": "mediation",
                "dv": "DV2_Export_Recovery",
                "test_id": "path_a_x_to_m",
                "coef": float(m.get("path_a_coef_x_to_m", np.nan)),
                "pvalue_raw": float(m.get("path_a_pvalue", np.nan)),
            }
        )
        rows.append(
            {
                "source": "mediation",
                "dv": "DV3_Partner_Diversification",
                "test_id": "path_b_m_to_y",
                "coef": float(m.get("path_b_coef_m_to_y", np.nan)),
                "pvalue_raw": float(m.get("path_b_pvalue", np.nan)),
            }
        )

    out = pd.DataFrame(rows)
    out = out[out["pvalue_raw"].notna()].copy()
    out["fdr_bh_qvalue"] = bh_fdr_adjust(out["pvalue_raw"])
    out["sig_0p05_raw"] = (out["pvalue_raw"] < 0.05).astype(int)
    out["sig_0p05_fdr"] = (out["fdr_bh_qvalue"] < 0.05).astype(int)
    return out


def build_decision_table(
    high_boot_df: pd.DataFrame,
    mediation_df: pd.DataFrame,
    fdr_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    # Claim 1: DV2 channel effect.
    if not high_boot_df.empty:
        r = high_boot_df.iloc[0]
        rows.append(
            {
                "claim_id": "C1",
                "claim": "DV2 channel effect exists and is robust",
                "evidence_metric": "high_precision_wild_bootstrap_p",
                "value": float(r.get("p_boot_wild_cluster", np.nan)),
                "threshold": 0.05,
                "supports_claim": int(float(r.get("p_boot_wild_cluster", np.nan)) < 0.05)
                if pd.notna(r.get("p_boot_wild_cluster", np.nan))
                else 0,
                "note": "Two-sided wild cluster bootstrap on z_eci:pe (DV2)",
            }
        )

    # Claim 2: mediation through DV2.
    if not mediation_df.empty:
        m = mediation_df.iloc[0]
        p = float(m.get("indirect_boot_pvalue", np.nan))
        rows.append(
            {
                "claim_id": "C2",
                "claim": "Effect on DV3 is mediated by DV2",
                "evidence_metric": "mediation_indirect_boot_p",
                "value": p,
                "threshold": 0.05,
                "supports_claim": int(p < 0.05) if pd.notna(p) else 0,
                "note": "Cluster bootstrap indirect effect a*b",
            }
        )

    # Claim 3: global evidence survives multiplicity.
    if not fdr_df.empty:
        dv2_sig = int(
            (
                (fdr_df["dv"] == "DV2_Export_Recovery")
                & (fdr_df["sig_0p05_fdr"] == 1)
            ).sum()
        )
        total_sig = int((fdr_df["sig_0p05_fdr"] == 1).sum())
        rows.append(
            {
                "claim_id": "C3",
                "claim": "Meaningful evidence remains after FDR correction",
                "evidence_metric": "count_fdr_significant_tests",
                "value": total_sig,
                "threshold": 1,
                "supports_claim": int(total_sig >= 1),
                "note": f"DV2 FDR-significant tests: {dv2_sig}",
            }
        )

    return pd.DataFrame(rows)


def write_summary(
    out_dir: Path,
    high_boot_df: pd.DataFrame,
    mediation_df: pd.DataFrame,
    fdr_df: pd.DataFrame,
    decision_df: pd.DataFrame,
) -> None:
    lines = [
        "# Final Inference Round",
        "",
        "This round includes:",
        "- High-precision wild cluster bootstrap for DV2.",
        "- Mediation test (X=z_eci:pe, M=DV2, Y=DV3) with cluster bootstrap.",
        "- Global Benjamini-Hochberg FDR correction across main and robustness tests.",
        "",
    ]
    if not high_boot_df.empty:
        r = high_boot_df.iloc[0]
        lines.append("## DV2 high-precision bootstrap")
        lines.append(
            f"- coef={r['coef_obs']:.4f}, cluster-p={r['p_obs_cluster']:.4g}, "
            f"wild-bootstrap p={r['p_boot_wild_cluster']:.4g}, "
            f"CI=[{r['coef_boot_ci_low_2p5']:.4f}, {r['coef_boot_ci_high_97p5']:.4f}], "
            f"reps={int(r['bootstrap_reps_success'])}"
        )
        lines.append("")

    if not mediation_df.empty:
        m = mediation_df.iloc[0]
        lines.append("## Mediation (DV2 -> DV3)")
        lines.append(
            f"- Path a (X->M): coef={m['path_a_coef_x_to_m']:.4f}, p={m['path_a_pvalue']:.4g}"
        )
        lines.append(
            f"- Path b (M->Y|X): coef={m['path_b_coef_m_to_y']:.4f}, p={m['path_b_pvalue']:.4g}"
        )
        lines.append(
            f"- Indirect a*b={m['indirect_coef_a_times_b']:.4f}, "
            f"boot-p={m['indirect_boot_pvalue']:.4g}, "
            f"CI=[{m['indirect_boot_ci_low_2p5']:.4f}, {m['indirect_boot_ci_high_97p5']:.4f}]"
        )
        lines.append("")

    if not fdr_df.empty:
        lines.append("## FDR correction")
        lines.append(f"- Total hypotheses: {len(fdr_df)}")
        lines.append(f"- Raw p<0.05: {int((fdr_df['sig_0p05_raw'] == 1).sum())}")
        lines.append(f"- FDR q<0.05: {int((fdr_df['sig_0p05_fdr'] == 1).sum())}")
        lines.append("")

    if not decision_df.empty:
        lines.append("## Decision table")
        for _, r in decision_df.iterrows():
            status = "SUPPORTED" if int(r["supports_claim"]) == 1 else "NOT SUPPORTED"
            lines.append(f"- {r['claim_id']} {status}: {r['claim']} ({r['evidence_metric']}={r['value']})")

    (out_dir / "final_inference_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final inference round analyses.")
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project base directory.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--bootstrap-reps-dv2", type=int, default=999)
    parser.add_argument("--bootstrap-reps-mediation", type=int, default=399)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_dir = base_dir / "reports" / "final_inference"
    ensure_dir(out_dir)

    panel = pd.read_csv(base_dir / "data" / "processed" / "regression_panel_2012_2022.csv")
    panel = panel[panel["year"].between(2012, 2022)].copy()
    panel["post_paper"] = panel["year"].between(2019, 2022).astype(int)

    high_boot_df = run_dv2_high_precision_bootstrap(
        panel=panel,
        out_dir=out_dir,
        reps=args.bootstrap_reps_dv2,
        seed=args.seed,
    )
    mediation_df = run_mediation(
        panel=panel,
        out_dir=out_dir,
        reps=args.bootstrap_reps_mediation,
        seed=args.seed + 1000,
    )
    fdr_df = collect_pvalues(base_dir=base_dir, high_boot_df=high_boot_df, mediation_df=mediation_df)
    fdr_df.to_csv(out_dir / "fdr_all_tests.csv", index=False)

    decision_df = build_decision_table(high_boot_df=high_boot_df, mediation_df=mediation_df, fdr_df=fdr_df)
    decision_df.to_csv(out_dir / "final_decision_table.csv", index=False)

    write_summary(
        out_dir=out_dir,
        high_boot_df=high_boot_df,
        mediation_df=mediation_df,
        fdr_df=fdr_df,
        decision_df=decision_df,
    )

    meta = {
        "panel_rows": int(len(panel)),
        "panel_countries": int(panel["country_iso3_code"].nunique()),
        "dv2_bootstrap_reps_requested": int(args.bootstrap_reps_dv2),
        "mediation_bootstrap_reps_requested": int(args.bootstrap_reps_mediation),
        "fdr_hypotheses_count": int(len(fdr_df)),
        "fdr_significant_count": int((fdr_df["sig_0p05_fdr"] == 1).sum()) if not fdr_df.empty else 0,
    }
    (out_dir / "final_inference_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Final inference round completed.")
    print(f"Output directory: {out_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
