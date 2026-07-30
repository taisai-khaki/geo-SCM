from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence


DV_MAP = {
    "DV1_GVC_Linkage_Change": "delta_tiva_fexgr_dva_share",
    "DV2_Export_Recovery": "export_recovery_index",
    "DV3_Partner_Diversification": "partner_diversification_1_minus_hhi",
}

BASE_CONTROLS = [
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


def prep_model_df(panel: pd.DataFrame, dv_col: str) -> pd.DataFrame:
    req = [dv_col, "eci", "pe", "country_iso3_code", "year"] + BASE_CONTROLS
    df = panel.dropna(subset=req).copy()
    df["z_eci"] = zscore(df["eci"])
    df["z_log_gdp_pc"] = zscore(df["log_gdp_pc"])
    df["z_trade_open"] = zscore(df["wdi_trade_openness_pct_gdp"])
    df["z_wgi"] = zscore(df["wgi_institutional_quality_composite"])
    df["z_rents"] = zscore(df["wdi_natural_resource_rents_pct_gdp"])
    df["z_gpr"] = zscore(df["gpr_for_model_annual"])
    df["z_intensity"] = zscore(df["us_china_trade_intensity_pre"])
    return df


def model_formula(dv_col: str) -> str:
    return (
        f"{dv_col} ~ z_eci + pe + z_eci:pe + z_log_gdp_pc + z_trade_open + z_wgi"
        " + z_rents + z_gpr + z_intensity + C(country_iso3_code) + C(year)"
    )


def fit_clustered(df: pd.DataFrame, dv_col: str) -> Any:
    return smf.ols(formula=model_formula(dv_col), data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["country_iso3_code"]},
    )


def add_influence_flags(df: pd.DataFrame, dv_col: str) -> pd.DataFrame:
    plain_fit = smf.ols(formula=model_formula(dv_col), data=df).fit()
    infl = OLSInfluence(plain_fit)
    dffits = np.abs(infl.dffits[0])
    cooks = infl.cooks_distance[0]
    leverage = infl.hat_matrix_diag

    n = len(df)
    p = int(plain_fit.df_model) + 1
    thr_dffits = 2 * np.sqrt(p / n)
    thr_cook = 4 / n
    thr_lev = 2 * p / n

    out = df.copy()
    out["abs_dffits"] = dffits
    out["cooks_d"] = cooks
    out["leverage"] = leverage
    out["is_high_dffits"] = out["abs_dffits"] > thr_dffits
    out["is_high_cooks"] = out["cooks_d"] > thr_cook
    out["is_high_leverage"] = out["leverage"] > thr_lev
    out["is_any_flag"] = out[["is_high_dffits", "is_high_cooks", "is_high_leverage"]].any(
        axis=1
    )
    return out


def run_loo(df: pd.DataFrame, dv_col: str, dv_name: str, out_dir: Path) -> pd.DataFrame:
    base_fit = fit_clustered(df, dv_col)
    full_coef = float(base_fit.params.get("z_eci:pe", np.nan))
    rows = []
    for iso3 in sorted(df["country_iso3_code"].unique()):
        dsub = df[df["country_iso3_code"] != iso3].copy()
        if dsub["country_iso3_code"].nunique() < 20:
            continue
        try:
            fit = fit_clustered(dsub, dv_col)
        except Exception:
            continue
        coef = float(fit.params.get("z_eci:pe", np.nan))
        pval = float(fit.pvalues.get("z_eci:pe", np.nan))
        rows.append(
            {
                "dv": dv_name,
                "dropped_country": iso3,
                "coef_z_eci_pe": coef,
                "pvalue_z_eci_pe": pval,
                "delta_vs_full": coef - full_coef,
            }
        )
    loo_df = pd.DataFrame(rows)
    loo_df.to_csv(out_dir / f"{dv_name}_leave_one_country_out.csv", index=False)
    return loo_df


def run_robust_specs(
    df: pd.DataFrame, dv_col: str, dv_name: str, flagged: pd.DataFrame
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fit_main = fit_clustered(df, dv_col)
    rows.append(
        {
            "dv": dv_name,
            "spec": "Main_FE_cluster",
            "coef_z_eci_pe": float(fit_main.params.get("z_eci:pe", np.nan)),
            "se": float(fit_main.bse.get("z_eci:pe", np.nan)),
            "pvalue": float(fit_main.pvalues.get("z_eci:pe", np.nan)),
            "n_obs": int(fit_main.nobs),
            "n_countries": int(df["country_iso3_code"].nunique()),
            "dropped_countries": "",
        }
    )

    trimmed = flagged[~flagged["is_any_flag"]].copy()
    if not trimmed.empty:
        fit_trim = fit_clustered(trimmed, dv_col)
        rows.append(
            {
                "dv": dv_name,
                "spec": "Trim_flagged_obs",
                "coef_z_eci_pe": float(fit_trim.params.get("z_eci:pe", np.nan)),
                "se": float(fit_trim.bse.get("z_eci:pe", np.nan)),
                "pvalue": float(fit_trim.pvalues.get("z_eci:pe", np.nan)),
                "n_obs": int(fit_trim.nobs),
                "n_countries": int(trimmed["country_iso3_code"].nunique()),
                "dropped_countries": "",
            }
        )

    wins = df.copy()
    lo_dv, hi_dv = wins[dv_col].quantile([0.01, 0.99]).tolist()
    lo_eci, hi_eci = wins["eci"].quantile([0.01, 0.99]).tolist()
    wins[dv_col] = wins[dv_col].clip(lo_dv, hi_dv)
    wins["eci"] = wins["eci"].clip(lo_eci, hi_eci)
    wins["z_eci"] = zscore(wins["eci"])
    fit_wins = fit_clustered(wins, dv_col)
    rows.append(
        {
            "dv": dv_name,
            "spec": "Winsorize_1_99_dv_eci",
            "coef_z_eci_pe": float(fit_wins.params.get("z_eci:pe", np.nan)),
            "se": float(fit_wins.bse.get("z_eci:pe", np.nan)),
            "pvalue": float(fit_wins.pvalues.get("z_eci:pe", np.nan)),
            "n_obs": int(fit_wins.nobs),
            "n_countries": int(wins["country_iso3_code"].nunique()),
            "dropped_countries": "",
        }
    )

    by_country = (
        flagged.groupby("country_iso3_code", as_index=False)["is_any_flag"]
        .sum()
        .rename(columns={"is_any_flag": "flagged_obs"})
        .sort_values("flagged_obs", ascending=False)
    )
    top5 = by_country.head(5)["country_iso3_code"].astype(str).tolist()
    drop5 = df[~df["country_iso3_code"].isin(top5)].copy()
    if not drop5.empty:
        fit_drop5 = fit_clustered(drop5, dv_col)
        rows.append(
            {
                "dv": dv_name,
                "spec": "Drop_top5_influential_countries",
                "coef_z_eci_pe": float(fit_drop5.params.get("z_eci:pe", np.nan)),
                "se": float(fit_drop5.bse.get("z_eci:pe", np.nan)),
                "pvalue": float(fit_drop5.pvalues.get("z_eci:pe", np.nan)),
                "n_obs": int(fit_drop5.nobs),
                "n_countries": int(drop5["country_iso3_code"].nunique()),
                "dropped_countries": "|".join(top5),
            }
        )

    return rows


def plot_stability(robust_df: pd.DataFrame, out_dir: Path) -> None:
    order = [
        "Main_FE_cluster",
        "Winsorize_1_99_dv_eci",
        "Trim_flagged_obs",
        "Drop_top5_influential_countries",
    ]
    labels = {
        "Main_FE_cluster": "Main",
        "Winsorize_1_99_dv_eci": "Winsorize 1-99",
        "Trim_flagged_obs": "Trim flagged obs",
        "Drop_top5_influential_countries": "Drop top-5 countries",
    }

    plot_df = robust_df[robust_df["spec"].isin(order)].copy()
    plot_df["spec_label"] = pd.Categorical(
        plot_df["spec"].map(labels),
        categories=[labels[o] for o in order],
        ordered=True,
    )
    dvs = plot_df["dv"].dropna().unique().tolist()
    fig, axes = plt.subplots(len(dvs), 1, figsize=(10, 3.5 * len(dvs)), sharex=True)
    if len(dvs) == 1:
        axes = [axes]
    for ax, dv in zip(axes, dvs):
        sub = plot_df[plot_df["dv"] == dv].sort_values("spec_label")
        ax.errorbar(
            x=sub["spec_label"].astype(str),
            y=sub["coef_z_eci_pe"],
            yerr=1.96 * sub["se"],
            fmt="o-",
            capsize=4,
            color="#1f77b4",
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(dv)
        ax.set_ylabel("Coef z_eci:pe")
        for _, row in sub.iterrows():
            ax.annotate(
                f"p={row['pvalue']:.3f}",
                (row["spec_label"], row["coef_z_eci_pe"]),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    axes[-1].set_xlabel("Specification")
    plt.tight_layout()
    plt.savefig(out_dir / "coef_stability_outlier_specs.png", dpi=180)
    plt.close()


def plot_flag_share_by_year(influence_frames: list[pd.DataFrame], out_dir: Path) -> None:
    rows = []
    for frame in influence_frames:
        dv_name = str(frame["dv"].iloc[0])
        temp = frame.copy()
        temp["flag"] = temp["is_any_flag"].astype(int)
        agg = (
            temp.groupby("year", as_index=False)["flag"]
            .mean()
            .rename(columns={"flag": "flag_share"})
        )
        agg["dv"] = dv_name
        rows.append(agg)
    if not rows:
        return
    plot_df = pd.concat(rows, ignore_index=True)
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=plot_df, x="year", y="flag_share", hue="dv", marker="o")
    plt.xlabel("Year")
    plt.ylabel("Share of flagged observations")
    plt.title("Outlier Flag Concentration Over Time")
    plt.tight_layout()
    plt.savefig(out_dir / "flagged_obs_share_by_year.png", dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deep outlier diagnostics and robust re-estimation.")
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project base directory",
    )
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument(
        "--run-loo",
        action="store_true",
        help="Run leave-one-country-out checks (slower).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    panel_path = base_dir / "data" / "processed" / "regression_panel_2012_2022.csv"
    out_dir = base_dir / "reports" / "outlier_diagnostics"
    ensure_dir(out_dir)

    panel = pd.read_csv(panel_path)
    panel = panel[panel["year"].between(args.start_year, args.end_year)].copy()
    panel["post_paper"] = panel["year"].between(2019, 2022).astype(int)
    panel["pe"] = panel["post_paper"] * panel["exposed"]

    summary_rows: list[dict[str, Any]] = []
    robust_rows: list[dict[str, Any]] = []
    top_country_rows: list[pd.DataFrame] = []
    influence_frames: list[pd.DataFrame] = []

    for dv_name, dv_col in DV_MAP.items():
        df = prep_model_df(panel, dv_col)
        fit = fit_clustered(df, dv_col)
        coef = float(fit.params.get("z_eci:pe", np.nan))
        pval = float(fit.pvalues.get("z_eci:pe", np.nan))

        flagged = add_influence_flags(df, dv_col)
        flagged["dv"] = dv_name
        influence_frames.append(flagged[["dv", "country_iso3_code", "year", "abs_dffits", "cooks_d", "leverage", "is_high_dffits", "is_high_cooks", "is_high_leverage", "is_any_flag"]])
        influence_frames[-1].to_csv(out_dir / f"{dv_name}_observation_influence.csv", index=False)

        country_rank = flagged.groupby("country_iso3_code", as_index=False).agg(
            n_obs=("country_iso3_code", "size"),
            flagged_obs=("is_any_flag", "sum"),
            max_cooks=("cooks_d", "max"),
            max_abs_dffits=("abs_dffits", "max"),
            max_leverage=("leverage", "max"),
        )
        country_rank["flag_rate"] = country_rank["flagged_obs"] / country_rank["n_obs"]
        country_rank = country_rank.sort_values(["flagged_obs", "max_cooks"], ascending=[False, False])
        country_rank.to_csv(out_dir / f"{dv_name}_country_influence_rank.csv", index=False)
        temp = country_rank.head(20).copy()
        temp["dv"] = dv_name
        top_country_rows.append(temp)

        loo_df = pd.DataFrame()
        if args.run_loo:
            loo_df = run_loo(df, dv_col, dv_name, out_dir)

        if not loo_df.empty:
            loo_min = float(loo_df["coef_z_eci_pe"].min())
            loo_max = float(loo_df["coef_z_eci_pe"].max())
            if coef == 0:
                flips = int((loo_df["coef_z_eci_pe"] != 0).sum())
            else:
                flips = int((np.sign(loo_df["coef_z_eci_pe"]) != np.sign(coef)).sum())
            loo_sig = int((loo_df["pvalue_z_eci_pe"] < 0.05).sum())
        else:
            loo_min = np.nan
            loo_max = np.nan
            flips = np.nan
            loo_sig = np.nan

        n_flag = int(flagged["is_any_flag"].sum())
        summary_rows.append(
            {
                "dv": dv_name,
                "n_obs_model": int(len(df)),
                "n_countries_model": int(df["country_iso3_code"].nunique()),
                "coef_full_z_eci_pe": coef,
                "p_full_z_eci_pe": pval,
                "flagged_obs_count": n_flag,
                "flagged_obs_share": float(n_flag / len(df)),
                "loo_coef_min": loo_min,
                "loo_coef_max": loo_max,
                "loo_sign_flip_count": flips,
                "loo_sig_count_p_lt_0_05": loo_sig,
            }
        )

        robust_rows.extend(run_robust_specs(df, dv_col, dv_name, flagged))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "outlier_stability_summary.csv", index=False)

    robust_df = pd.DataFrame(robust_rows)
    robust_df.to_csv(out_dir / "outlier_robustness_reestimation.csv", index=False)

    delta_rows = []
    for dv in robust_df["dv"].dropna().unique():
        sub = robust_df[robust_df["dv"] == dv].copy()
        main_coef = float(sub.loc[sub["spec"] == "Main_FE_cluster", "coef_z_eci_pe"].iloc[0])
        for _, row in sub.iterrows():
            delta_rows.append(
                {
                    "dv": dv,
                    "spec": row["spec"],
                    "coef": row["coef_z_eci_pe"],
                    "pvalue": row["pvalue"],
                    "delta_vs_main": row["coef_z_eci_pe"] - main_coef,
                    "n_obs": row["n_obs"],
                    "n_countries": row["n_countries"],
                }
            )
    pd.DataFrame(delta_rows).to_csv(out_dir / "outlier_robustness_delta_vs_main.csv", index=False)

    if top_country_rows:
        pd.concat(top_country_rows, ignore_index=True).to_csv(
            out_dir / "top_influential_countries_all_dv.csv", index=False
        )

    # Recurrent influential countries: at least 3 flagged obs in a DV.
    top_path = out_dir / "top_influential_countries_all_dv.csv"
    if top_path.exists():
        top = pd.read_csv(top_path)
        recurrent = (
            top[top["flagged_obs"] >= 3]
            .groupby("country_iso3_code", as_index=False)
            .agg(
                dvs=("dv", "nunique"),
                total_flagged=("flagged_obs", "sum"),
                avg_flag_rate=("flag_rate", "mean"),
            )
            .sort_values(["dvs", "total_flagged"], ascending=[False, False])
        )
        recurrent.to_csv(out_dir / "recurrent_influential_countries.csv", index=False)

    # Flag share by size quartile.
    size_source = (
        panel[["country_iso3_code", "baseline_export_2015_2017"]]
        .dropna()
        .drop_duplicates()
        .groupby("country_iso3_code", as_index=False)["baseline_export_2015_2017"]
        .mean()
    )
    if not size_source.empty:
        size_source["size_bin"] = pd.qcut(
            size_source["baseline_export_2015_2017"],
            q=4,
            labels=["Q1_smallest", "Q2", "Q3", "Q4_largest"],
            duplicates="drop",
        )
        size_rows = []
        for frame in influence_frames:
            temp = frame.merge(
                size_source[["country_iso3_code", "size_bin"]],
                on="country_iso3_code",
                how="left",
            )
            temp["flag"] = temp["is_any_flag"].astype(int)
            agg = temp.groupby("size_bin", as_index=False, observed=False).agg(
                flag_share=("flag", "mean"),
                n=("flag", "size"),
                flag_n=("flag", "sum"),
            )
            agg["dv"] = temp["dv"].iloc[0]
            size_rows.append(agg)
        pd.concat(size_rows, ignore_index=True).to_csv(
            out_dir / "flag_share_by_country_size_quartile.csv",
            index=False,
        )

    # Weighted vs unweighted FE
    weighted_rows = []
    for dv_name, dv_col in DV_MAP.items():
        req = [dv_col, "baseline_export_2015_2017"]
        df = prep_model_df(panel, dv_col).dropna(subset=req).copy()
        if df.empty:
            continue
        fit_unw = fit_clustered(df, dv_col)
        weights = np.log1p(df["baseline_export_2015_2017"].clip(lower=0))
        weights = weights / weights.mean()
        fit_w = smf.wls(formula=model_formula(dv_col), data=df, weights=weights).fit(
            cov_type="cluster",
            cov_kwds={"groups": df["country_iso3_code"]},
        )
        for label, fit in [
            ("Unweighted_FE", fit_unw),
            ("Weighted_logBaselineExport_FE", fit_w),
        ]:
            weighted_rows.append(
                {
                    "dv": dv_name,
                    "spec": label,
                    "coef_z_eci_pe": float(fit.params.get("z_eci:pe", np.nan)),
                    "se": float(fit.bse.get("z_eci:pe", np.nan)),
                    "pvalue": float(fit.pvalues.get("z_eci:pe", np.nan)),
                    "n_obs": int(fit.nobs),
                    "n_countries": int(df["country_iso3_code"].nunique()),
                }
            )
    pd.DataFrame(weighted_rows).to_csv(out_dir / "weighted_vs_unweighted_results.csv", index=False)

    plot_stability(robust_df, out_dir)
    plot_flag_share_by_year(influence_frames, out_dir)

    lines = [
        "# Outlier-Focused Recommendation Pack",
        "",
        "Generated assets:",
        "- `outlier_stability_summary.csv`",
        "- `outlier_robustness_reestimation.csv`",
        "- `outlier_robustness_delta_vs_main.csv`",
        "- `top_influential_countries_all_dv.csv`",
        "- `recurrent_influential_countries.csv`",
        "- `flag_share_by_country_size_quartile.csv`",
        "- `weighted_vs_unweighted_results.csv`",
        "- `coef_stability_outlier_specs.png`",
        "- `flagged_obs_share_by_year.png`",
    ]
    if args.run_loo:
        lines.append("- `*_leave_one_country_out.csv`")
    lines.append("")
    lines.append("Model term tracked: `z_eci:pe` from FE specification with clustered SEs by country.")
    (out_dir / "outlier_recommendation_pack.md").write_text("\n".join(lines), encoding="utf-8")

    print("Outlier diagnostics completed.")
    print(f"Output directory: {out_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
