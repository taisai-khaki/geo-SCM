from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import norm


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def zscore(s: pd.Series) -> pd.Series:
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


def fit_cluster_heterogeneity(df: pd.DataFrame, dv: str) -> tuple[Any, pd.DataFrame]:
    work = df.copy()
    req = [
        dv,
        "eci",
        "post_paper",
        "exposed",
        "cluster_label",
        "log_gdp_pc",
        "wdi_trade_openness_pct_gdp",
        "wgi_institutional_quality_composite",
        "wdi_natural_resource_rents_pct_gdp",
        "gpr_for_model_annual",
        "us_china_trade_intensity_pre",
        "country_iso3_code",
        "year",
    ]
    work = work.dropna(subset=req).copy()
    if work.empty:
        raise ValueError(f"No data after filtering for {dv}")

    work["z_eci"] = zscore(work["eci"])
    work["z_log_gdp_pc"] = zscore(work["log_gdp_pc"])
    work["z_trade_open"] = zscore(work["wdi_trade_openness_pct_gdp"])
    work["z_wgi"] = zscore(work["wgi_institutional_quality_composite"])
    work["z_rents"] = zscore(work["wdi_natural_resource_rents_pct_gdp"])
    work["z_gpr"] = zscore(work["gpr_for_model_annual"])
    work["z_intensity"] = zscore(work["us_china_trade_intensity_pre"])
    work["pe"] = work["post_paper"] * work["exposed"]

    formula = (
        f"{dv} ~ z_eci + pe + z_eci:pe + z_eci:pe:C(cluster_label)"
        " + z_log_gdp_pc + z_trade_open + z_wgi + z_rents + z_gpr + z_intensity"
        " + C(country_iso3_code) + C(year)"
    )
    fit = smf.ols(formula=formula, data=work).fit(
        cov_type="cluster", cov_kwds={"groups": work["country_iso3_code"]}
    )
    return fit, work


def cluster_effects_from_fit(fit: Any, df: pd.DataFrame, dv_name: str, dv_col: str) -> pd.DataFrame:
    params = fit.params
    cov = fit.cov_params()
    param_names = list(params.index)

    main_term = "z_eci:pe"
    if main_term not in param_names:
        raise ValueError(f"Missing key term {main_term} for {dv_name}")

    clusters = sorted(df["cluster_label"].dropna().astype(str).unique().tolist())
    ref_cluster = clusters[0]
    rows: list[dict[str, Any]] = []

    def lincomb(effect_terms: dict[str, float]) -> tuple[float, float, float]:
        l = np.zeros(len(param_names))
        for t, w in effect_terms.items():
            if t in param_names:
                l[param_names.index(t)] = w
        beta = float(np.dot(l, params.values))
        var = float(np.dot(l, np.dot(cov.values, l)))
        se = float(np.sqrt(var)) if var >= 0 else np.nan
        if pd.isna(se) or se == 0:
            p = np.nan
        else:
            z = beta / se
            p = float(2 * norm.sf(abs(z)))
        return beta, se, p

    # Baseline cluster
    b0, se0, p0 = lincomb({main_term: 1.0})
    rows.append(
        {
            "dv": dv_name,
            "dv_column": dv_col,
            "cluster_label": ref_cluster,
            "effect_post_exposed_eci": b0,
            "se": se0,
            "pvalue": p0,
            "ci_low": b0 - 1.96 * se0 if pd.notna(se0) else np.nan,
            "ci_high": b0 + 1.96 * se0 if pd.notna(se0) else np.nan,
            "is_reference_cluster": True,
            "n_obs": int(fit.nobs),
            "n_countries": int(df["country_iso3_code"].nunique()),
            "years_min": int(df["year"].min()),
            "years_max": int(df["year"].max()),
        }
    )

    for c in clusters[1:]:
        diff_term = f"z_eci:pe:C(cluster_label)[T.{c}]"
        b, se, p = lincomb({main_term: 1.0, diff_term: 1.0})
        b_diff = float(params.get(diff_term, np.nan))
        se_diff = float(fit.bse.get(diff_term, np.nan))
        p_diff = float(fit.pvalues.get(diff_term, np.nan))
        rows.append(
            {
                "dv": dv_name,
                "dv_column": dv_col,
                "cluster_label": c,
                "effect_post_exposed_eci": b,
                "se": se,
                "pvalue": p,
                "ci_low": b - 1.96 * se if pd.notna(se) else np.nan,
                "ci_high": b + 1.96 * se if pd.notna(se) else np.nan,
                "is_reference_cluster": False,
                "delta_vs_reference_coef": b_diff,
                "delta_vs_reference_se": se_diff,
                "delta_vs_reference_pvalue": p_diff,
                "n_obs": int(fit.nobs),
                "n_countries": int(df["country_iso3_code"].nunique()),
                "years_min": int(df["year"].min()),
                "years_max": int(df["year"].max()),
            }
        )

    return pd.DataFrame(rows)


def plot_cluster_effects(effects: pd.DataFrame, out_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    plot_df = effects.copy()
    dorder = ["DV1_GVC_Linkage_Change", "DV2_Export_Recovery", "DV3_Partner_Diversification"]
    plot_df["dv"] = pd.Categorical(plot_df["dv"], categories=dorder, ordered=True)
    plot_df = plot_df.sort_values(["dv", "cluster_label"])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    for i, dv in enumerate(dorder):
        ax = axes[i]
        sub = plot_df[plot_df["dv"] == dv].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        x = np.arange(len(sub))
        y = sub["effect_post_exposed_eci"].values
        yerr = 1.96 * sub["se"].values
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            capsize=4,
            color="#1f77b4",
            ecolor="#1f77b4",
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["cluster_label"].tolist())
        ax.set_title(dv.replace("_", " "))
        ax.set_xlabel("Cluster")
        if i == 0:
            ax.set_ylabel("Effect of ECI in Post×Exposed\n(95% CI)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cluster-regime heterogeneity models with cluster-specific marginal effects."
    )
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project base directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    panel_path = base_dir / "data" / "processed" / "regression_panel_2012_2022.csv"
    cluster_path = base_dir / "reports" / "exploratory_patterns" / "country_clusters_with_outcomes.csv"
    out_dir = base_dir / "reports" / "cluster_regime_models"
    ensure_dirs(out_dir)

    panel = pd.read_csv(panel_path)
    clusters = pd.read_csv(cluster_path)[["country_iso3_code", "cluster_label"]].drop_duplicates()
    df = panel.merge(clusters, on="country_iso3_code", how="inner")
    df["post_paper"] = df["year"].between(2019, 2022).astype(int)

    dvars = {
        "DV1_GVC_Linkage_Change": "delta_tiva_fexgr_dva_share",
        "DV2_Export_Recovery": "export_recovery_index",
        "DV3_Partner_Diversification": "partner_diversification_1_minus_hhi",
    }

    all_effects: list[pd.DataFrame] = []
    model_meta: list[dict[str, Any]] = []

    for dv_name, dv_col in dvars.items():
        fit, model_df = fit_cluster_heterogeneity(df, dv=dv_col)
        effects = cluster_effects_from_fit(fit, model_df, dv_name=dv_name, dv_col=dv_col)
        all_effects.append(effects)

        term_rows = []
        for t in ["z_eci:pe"] + [x for x in fit.params.index if x.startswith("z_eci:pe:C(cluster_label)")]:
            term_rows.append(
                {
                    "dv": dv_name,
                    "term": t,
                    "coef": float(fit.params.get(t, np.nan)),
                    "se": float(fit.bse.get(t, np.nan)),
                    "pvalue": float(fit.pvalues.get(t, np.nan)),
                }
            )
        pd.DataFrame(term_rows).to_csv(out_dir / f"{dv_name}_interaction_terms.csv", index=False)
        with open(out_dir / f"{dv_name}_model_summary.txt", "w", encoding="utf-8") as fh:
            fh.write(fit.summary().as_text())

        model_meta.append(
            {
                "dv": dv_name,
                "n_obs": int(fit.nobs),
                "n_countries": int(model_df["country_iso3_code"].nunique()),
                "year_min": int(model_df["year"].min()),
                "year_max": int(model_df["year"].max()),
                "r2": float(getattr(fit, "rsquared", np.nan)),
            }
        )

    effects_df = pd.concat(all_effects, ignore_index=True)
    effects_df.to_csv(out_dir / "cluster_specific_marginal_effects.csv", index=False)
    plot_cluster_effects(effects_df, out_dir / "cluster_specific_effects_plot.png")

    md_lines = [
        "# Cluster-Regime Heterogeneity Results",
        "",
        "Model core:",
        "- Outcome ~ z(ECI) + (Post×Exposed) + z(ECI)×(Post×Exposed)",
        "- Plus cluster interaction: z(ECI)×(Post×Exposed)×Cluster",
        "- Country FE + Year FE + clustered SE by country",
        "- Post period: 2019–2022",
        "",
        "Primary output:",
        "- `cluster_specific_marginal_effects.csv`",
        "- `cluster_specific_effects_plot.png`",
    ]
    (out_dir / "cluster_regime_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    (out_dir / "cluster_regime_metadata.json").write_text(
        json.dumps(model_meta, indent=2), encoding="utf-8"
    )

    print("Cluster-regime models completed.")
    print(f"Rows in marginal-effects table: {len(effects_df)}")


if __name__ == "__main__":
    main()
