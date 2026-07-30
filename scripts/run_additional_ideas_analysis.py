from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


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


def prepare_main_panel(panel: pd.DataFrame, dv_col: str) -> pd.DataFrame:
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


def fit_main_fe(df: pd.DataFrame, dv_col: str) -> Any:
    formula = (
        f"{dv_col} ~ z_eci + pe + z_eci:pe + z_log_gdp_pc + z_trade_open + z_wgi + "
        "z_rents + z_gpr + z_intensity + C(country_iso3_code) + C(year)"
    )
    return smf.ols(formula=formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["country_iso3_code"]},
    )


def compute_event_study(
    panel: pd.DataFrame, out_dir: Path, min_event: int = -6, max_event: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coef_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []

    for dv_name, dv_col in DV_MAP.items():
        df = prepare_main_panel(panel, dv_col)
        if df.empty:
            continue

        df["event_time"] = (df["year"] - 2018).astype(int)
        df = df[df["event_time"].between(min_event, max_event)].copy()
        df["event_time"] = df["event_time"].astype(int)

        formula = (
            f"{dv_col} ~ z_eci + z_log_gdp_pc + z_trade_open + z_wgi + z_rents + z_gpr + z_intensity "
            " + C(country_iso3_code) + C(year)"
            " + C(event_time, Treatment(reference=-1)):exposed:z_eci"
        )
        fit = smf.ols(formula=formula, data=df).fit(
            cov_type="cluster",
            cov_kwds={"groups": df["country_iso3_code"]},
        )

        # Parse event-time-specific coefficients.
        for k in range(min_event, max_event + 1):
            if k == -1:
                continue
            term = f"C(event_time, Treatment(reference=-1))[{k}]:exposed:z_eci"
            coef = float(fit.params.get(term, np.nan))
            se = float(fit.bse.get(term, np.nan))
            pval = float(fit.pvalues.get(term, np.nan))
            coef_rows.append(
                {
                    "dv": dv_name,
                    "event_time": int(k),
                    "coef": coef,
                    "se": se,
                    "pvalue": pval,
                    "ci_low": coef - 1.96 * se if pd.notna(coef) and pd.notna(se) else np.nan,
                    "ci_high": coef + 1.96 * se if pd.notna(coef) and pd.notna(se) else np.nan,
                    "n_obs": int(fit.nobs),
                    "n_countries": int(df["country_iso3_code"].nunique()),
                }
            )

        # Joint pre-trend and post-period tests.
        pre_terms = [
            f"C(event_time, Treatment(reference=-1))[{k}]:exposed:z_eci"
            for k in range(min_event, -1)
            if k <= -2
        ]
        post_terms = [
            f"C(event_time, Treatment(reference=-1))[{k}]:exposed:z_eci"
            for k in range(1, max_event + 1)
        ]
        available_pre = [t for t in pre_terms if t in fit.params.index]
        available_post = [t for t in post_terms if t in fit.params.index]

        if available_pre:
            hypothesis = ", ".join([f"{t} = 0" for t in available_pre])
            w = fit.wald_test(hypothesis)
            joint_rows.append(
                {
                    "dv": dv_name,
                    "test": "pretrend_joint_zero",
                    "n_terms": len(available_pre),
                    "stat": float(np.asarray(w.statistic).reshape(-1)[0]),
                    "pvalue": float(w.pvalue),
                }
            )

        if available_post:
            hypothesis = ", ".join([f"{t} = 0" for t in available_post])
            w = fit.wald_test(hypothesis)
            joint_rows.append(
                {
                    "dv": dv_name,
                    "test": "post_joint_zero",
                    "n_terms": len(available_post),
                    "stat": float(np.asarray(w.statistic).reshape(-1)[0]),
                    "pvalue": float(w.pvalue),
                }
            )

    coef_df = pd.DataFrame(coef_rows)
    joint_df = pd.DataFrame(joint_rows)
    coef_df.to_csv(out_dir / "event_study_coefficients.csv", index=False)
    joint_df.to_csv(out_dir / "event_study_joint_tests.csv", index=False)
    return coef_df, joint_df


def plot_event_study(coef_df: pd.DataFrame, out_dir: Path) -> None:
    if coef_df.empty:
        return
    for dv_name in coef_df["dv"].dropna().unique():
        sub = coef_df[coef_df["dv"] == dv_name].sort_values("event_time")
        plt.figure(figsize=(9, 4.5))
        plt.errorbar(
            sub["event_time"],
            sub["coef"],
            yerr=1.96 * sub["se"],
            fmt="o-",
            capsize=4,
            color="#1f77b4",
        )
        plt.axhline(0, color="black", linestyle="--", linewidth=1)
        plt.axvline(-1, color="gray", linestyle=":", linewidth=1)
        plt.xlabel("Event time (year - 2018)")
        plt.ylabel("Coef on event_time x exposed x z_eci")
        plt.title(f"Event-study: {dv_name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"event_study_{dv_name}.png", dpi=180)
        plt.close()


def run_sample_split(panel: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    # Define core economies by excluding Q1 (smallest baseline-export quartile).
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
    panel2 = panel.merge(csize[["country_iso3_code", "size_q"]], on="country_iso3_code", how="left")

    rows: list[dict[str, Any]] = []
    for sample_name, sample_df in [
        ("full_sample", panel2),
        ("core_economies_excl_q1", panel2[panel2["size_q"] != 1].copy()),
    ]:
        for dv_name, dv_col in DV_MAP.items():
            df = prepare_main_panel(sample_df, dv_col)
            if df.empty or df["country_iso3_code"].nunique() < 20:
                continue
            fit = fit_main_fe(df, dv_col)
            rows.append(
                {
                    "sample": sample_name,
                    "dv": dv_name,
                    "coef_z_eci_pe": float(fit.params.get("z_eci:pe", np.nan)),
                    "se": float(fit.bse.get("z_eci:pe", np.nan)),
                    "pvalue": float(fit.pvalues.get("z_eci:pe", np.nan)),
                    "n_obs": int(fit.nobs),
                    "n_countries": int(df["country_iso3_code"].nunique()),
                    "year_min": int(df["year"].min()),
                    "year_max": int(df["year"].max()),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "sample_split_results.csv", index=False)
    return out


def run_quantile_and_regime(panel: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    quant_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []

    # Quantile FE-like (year FE only to keep estimator stable with quantreg).
    quantiles = [0.25, 0.5, 0.75]
    for dv_name, dv_col in DV_MAP.items():
        # DV1 is sparse and noisier; still attempt.
        df = prepare_main_panel(panel, dv_col)
        if df.empty:
            continue
        q_formula = (
            f"{dv_col} ~ z_eci + pe + z_eci:pe + z_log_gdp_pc + z_trade_open + z_wgi + "
            "z_rents + z_gpr + z_intensity + C(year)"
        )
        for q in quantiles:
            try:
                fit_q = smf.quantreg(q_formula, data=df).fit(q=q, max_iter=4000)
                quant_rows.append(
                    {
                        "dv": dv_name,
                        "quantile": q,
                        "coef_z_eci_pe": float(fit_q.params.get("z_eci:pe", np.nan)),
                        "se": float(fit_q.bse.get("z_eci:pe", np.nan)),
                        "pvalue": float(fit_q.pvalues.get("z_eci:pe", np.nan)),
                        "n_obs": int(len(df)),
                        "n_countries": int(df["country_iso3_code"].nunique()),
                    }
                )
            except Exception:
                continue

        # Regime by ECI terciles in model sample.
        temp = df.copy()
        temp["eci_regime"] = pd.qcut(temp["eci"], q=3, labels=["low_eci", "mid_eci", "high_eci"], duplicates="drop")
        for regime in temp["eci_regime"].dropna().unique():
            dsub = temp[temp["eci_regime"] == regime].copy()
            if dsub["country_iso3_code"].nunique() < 20:
                continue
            try:
                fit_r = fit_main_fe(dsub, dv_col)
            except Exception:
                continue
            regime_rows.append(
                {
                    "dv": dv_name,
                    "eci_regime": str(regime),
                    "coef_z_eci_pe": float(fit_r.params.get("z_eci:pe", np.nan)),
                    "se": float(fit_r.bse.get("z_eci:pe", np.nan)),
                    "pvalue": float(fit_r.pvalues.get("z_eci:pe", np.nan)),
                    "n_obs": int(fit_r.nobs),
                    "n_countries": int(dsub["country_iso3_code"].nunique()),
                }
            )

    quant_df = pd.DataFrame(quant_rows)
    regime_df = pd.DataFrame(regime_rows)
    quant_df.to_csv(out_dir / "quantile_results.csv", index=False)
    regime_df.to_csv(out_dir / "regime_eci_tercile_results.csv", index=False)
    return quant_df, regime_df


def build_us_china_trade_intensity(bilateral_df: pd.DataFrame) -> pd.DataFrame:
    pre = bilateral_df[bilateral_df["year"].between(2015, 2017)].copy()
    pre["trade_pair"] = pre["export_value"].fillna(0) + pre["import_value"].fillna(0)

    total = (
        pre.groupby(["country_iso3_code", "year"], as_index=False)["trade_pair"]
        .sum()
        .rename(columns={"trade_pair": "total_trade"})
    )
    uschn = (
        pre[pre["partner_iso3_code"].isin(["USA", "CHN"])]
        .groupby(["country_iso3_code", "year"], as_index=False)["trade_pair"]
        .sum()
        .rename(columns={"trade_pair": "us_china_trade"})
    )
    merged = total.merge(uschn, on=["country_iso3_code", "year"], how="left")
    merged["us_china_trade"] = merged["us_china_trade"].fillna(0)
    merged = merged[merged["total_trade"] > 0].copy()
    merged["intensity_y"] = merged["us_china_trade"] / merged["total_trade"]
    intensity = (
        merged.groupby("country_iso3_code", as_index=False)["intensity_y"]
        .mean()
        .rename(columns={"intensity_y": "us_china_trade_intensity_pre"})
    )
    q2 = intensity["us_china_trade_intensity_pre"].quantile(2 / 3)
    intensity["exposed"] = (intensity["us_china_trade_intensity_pre"] >= q2).astype(int)
    return intensity


def run_post2022_external_validation(base_dir: Path, out_dir: Path) -> pd.DataFrame:
    db_path = base_dir / "data" / "processed" / "databank_country_year_2012_2024.csv"
    bilateral_path = base_dir / "data" / "processed" / "source_atlas_country_country_year_2012_2024.csv"
    panel = pd.read_csv(db_path)
    bilateral = pd.read_csv(bilateral_path)
    intensity = build_us_china_trade_intensity(bilateral)
    panel = panel.merge(intensity, on="country_iso3_code", how="left")

    panel["post_paper"] = panel["year"].between(2019, 2024).astype(int)
    panel["post_after_2022"] = (panel["year"] >= 2023).astype(int)
    panel["pe"] = panel["post_paper"] * panel["exposed"]
    panel["z_eci"] = zscore(panel["eci"])

    rows: list[dict[str, Any]] = []
    for dv_name, dv_col in [
        ("DV2_Export_Recovery", "export_recovery_index"),
        ("DV3_Partner_Diversification", "partner_diversification_1_minus_hhi"),
    ]:
        req = [dv_col, "z_eci", "pe", "country_iso3_code", "year", "post_after_2022"]
        df = panel.dropna(subset=req).copy()
        if df.empty:
            continue

        # Base extended model (no external controls; validation purpose).
        formula_base = (
            f"{dv_col} ~ z_eci + pe + z_eci:pe + C(country_iso3_code) + C(year)"
        )
        fit_base = smf.ols(formula=formula_base, data=df).fit(
            cov_type="cluster",
            cov_kwds={"groups": df["country_iso3_code"]},
        )

        # Stability test: does slope change after 2022?
        formula_stab = (
            f"{dv_col} ~ z_eci + pe + z_eci:pe + post_after_2022 + "
            "z_eci:pe:post_after_2022 + C(country_iso3_code) + C(year)"
        )
        fit_stab = smf.ols(formula=formula_stab, data=df).fit(
            cov_type="cluster",
            cov_kwds={"groups": df["country_iso3_code"]},
        )

        rows.append(
            {
                "dv": dv_name,
                "model": "extended_2012_2024_base",
                "coef_z_eci_pe": float(fit_base.params.get("z_eci:pe", np.nan)),
                "se": float(fit_base.bse.get("z_eci:pe", np.nan)),
                "pvalue": float(fit_base.pvalues.get("z_eci:pe", np.nan)),
                "n_obs": int(fit_base.nobs),
                "n_countries": int(df["country_iso3_code"].nunique()),
                "years_min": int(df["year"].min()),
                "years_max": int(df["year"].max()),
            }
        )
        rows.append(
            {
                "dv": dv_name,
                "model": "post2022_slope_shift_test",
                "coef_z_eci_pe": float(fit_stab.params.get("z_eci:pe", np.nan)),
                "se": float(fit_stab.bse.get("z_eci:pe", np.nan)),
                "pvalue": float(fit_stab.pvalues.get("z_eci:pe", np.nan)),
                "coef_shift_post2022": float(
                    fit_stab.params.get("z_eci:pe:post_after_2022", np.nan)
                ),
                "pvalue_shift_post2022": float(
                    fit_stab.pvalues.get("z_eci:pe:post_after_2022", np.nan)
                ),
                "n_obs": int(fit_stab.nobs),
                "n_countries": int(df["country_iso3_code"].nunique()),
                "years_min": int(df["year"].min()),
                "years_max": int(df["year"].max()),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "post2022_external_validation.csv", index=False)
    return out


def write_summary(
    out_dir: Path,
    event_joint: pd.DataFrame,
    split_df: pd.DataFrame,
    quant_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    ext_df: pd.DataFrame,
) -> None:
    lines = [
        "# Additional Ideas: Deep Analysis Results",
        "",
        "This report implements:",
        "- Event-study pre-trend and dynamic effects.",
        "- Dual sample reporting (full sample vs. core economies excluding smallest export quartile).",
        "- Quantile/regime heterogeneity analysis.",
        "- Post-2022 external validation using non-TiVA outcomes (DV2, DV3) through 2024.",
        "",
    ]

    if not event_joint.empty:
        lines.append("## Event-study joint tests")
        for _, r in event_joint.iterrows():
            lines.append(
                f"- {r['dv']} | {r['test']}: p={r['pvalue']:.4g} (terms={int(r['n_terms'])})"
            )
        lines.append("")

    if not split_df.empty:
        lines.append("## Full vs core sample")
        for dv in split_df["dv"].dropna().unique():
            sub = split_df[split_df["dv"] == dv].copy()
            for _, r in sub.iterrows():
                lines.append(
                    f"- {dv} | {r['sample']}: coef={r['coef_z_eci_pe']:.4f}, p={r['pvalue']:.4g}, N={int(r['n_obs'])}"
                )
        lines.append("")

    if not quant_df.empty:
        lines.append("## Quantile results (z_eci:pe)")
        for dv in quant_df["dv"].dropna().unique():
            sub = quant_df[quant_df["dv"] == dv].sort_values("quantile")
            vals = ", ".join(
                [f"q{int(100*q)}={coef:.4f}(p={p:.3g})" for q, coef, p in zip(sub["quantile"], sub["coef_z_eci_pe"], sub["pvalue"])]
            )
            lines.append(f"- {dv}: {vals}")
        lines.append("")

    if not regime_df.empty:
        lines.append("## ECI regime results (z_eci:pe)")
        for dv in regime_df["dv"].dropna().unique():
            sub = regime_df[regime_df["dv"] == dv]
            for _, r in sub.iterrows():
                lines.append(
                    f"- {dv} | {r['eci_regime']}: coef={r['coef_z_eci_pe']:.4f}, p={r['pvalue']:.4g}"
                )
        lines.append("")

    if not ext_df.empty:
        lines.append("## Post-2022 external validation (2012-2024)")
        for _, r in ext_df.iterrows():
            if r["model"] == "post2022_slope_shift_test":
                lines.append(
                    f"- {r['dv']} | shift test: baseline coef={r['coef_z_eci_pe']:.4f} (p={r['pvalue']:.4g}); "
                    f"post-2022 shift={r.get('coef_shift_post2022', np.nan):.4f} "
                    f"(p={r.get('pvalue_shift_post2022', np.nan):.4g})"
                )
            else:
                lines.append(
                    f"- {r['dv']} | extended base: coef={r['coef_z_eci_pe']:.4f}, p={r['pvalue']:.4g}, years={int(r['years_min'])}-{int(r['years_max'])}"
                )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "- Strongest and most stable signal remains DV2 (Export Recovery), including outlier-robust settings.",
            "- DV1 and DV3 show stronger heterogeneity and sensitivity; these are better framed as conditional effects rather than universal effects.",
            "- Post-2022 validation helps assess whether effects persist in the extended period where TiVA is unavailable.",
        ]
    )

    (out_dir / "additional_ideas_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run additional deep-analysis ideas and export results.")
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project base directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    reports_dir = base_dir / "reports"
    out_dir = reports_dir / "additional_ideas"
    ensure_dir(out_dir)

    panel = pd.read_csv(base_dir / "data" / "processed" / "regression_panel_2012_2022.csv")
    panel = panel[panel["year"].between(2012, 2022)].copy()
    panel["post_paper"] = panel["year"].between(2019, 2022).astype(int)

    event_coef, event_joint = compute_event_study(panel, out_dir=out_dir)
    plot_event_study(event_coef, out_dir=out_dir)
    split_df = run_sample_split(panel, out_dir=out_dir)
    quant_df, regime_df = run_quantile_and_regime(panel, out_dir=out_dir)
    ext_df = run_post2022_external_validation(base_dir=base_dir, out_dir=out_dir)

    meta = {
        "panel_rows_2012_2022": int(len(panel)),
        "panel_countries_2012_2022": int(panel["country_iso3_code"].nunique()),
        "event_coeff_rows": int(len(event_coef)),
        "event_joint_rows": int(len(event_joint)),
        "sample_split_rows": int(len(split_df)),
        "quantile_rows": int(len(quant_df)),
        "regime_rows": int(len(regime_df)),
        "external_validation_rows": int(len(ext_df)),
    }
    (out_dir / "additional_ideas_metadata.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    write_summary(
        out_dir=out_dir,
        event_joint=event_joint,
        split_df=split_df,
        quant_df=quant_df,
        regime_df=regime_df,
        ext_df=ext_df,
    )

    print("Additional deep analysis completed.")
    print(f"Output directory: {out_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
