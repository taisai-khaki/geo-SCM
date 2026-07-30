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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def zscore(s: pd.Series) -> pd.Series:
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


def build_country_profile(panel: pd.DataFrame) -> pd.DataFrame:
    pre = panel[panel["year"].between(2012, 2017)].copy()
    profile = (
        pre.groupby("country_iso3_code", as_index=False)[
            [
                "eci",
                "coi",
                "log_gdp_pc",
                "wdi_trade_openness_pct_gdp",
                "wgi_institutional_quality_composite",
                "wdi_natural_resource_rents_pct_gdp",
                "wdi_tariff_applied_weighted_mean_all_products_pct",
                "us_china_trade_intensity_pre",
            ]
        ]
        .mean()
        .rename(
            columns={
                "eci": "pre_eci",
                "coi": "pre_coi",
                "log_gdp_pc": "pre_log_gdp_pc",
                "wdi_trade_openness_pct_gdp": "pre_trade_open",
                "wgi_institutional_quality_composite": "pre_wgi",
                "wdi_natural_resource_rents_pct_gdp": "pre_rents",
                "wdi_tariff_applied_weighted_mean_all_products_pct": "pre_tariff",
                "us_china_trade_intensity_pre": "pre_us_china_intensity",
            }
        )
    )
    return profile


def choose_k_and_cluster(profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    feature_cols = [
        "pre_eci",
        "pre_coi",
        "pre_log_gdp_pc",
        "pre_trade_open",
        "pre_wgi",
        "pre_rents",
        "pre_tariff",
        "pre_us_china_intensity",
    ]
    x_raw = profile[feature_cols].copy()
    imputer = SimpleImputer(strategy="median")
    x_imp = imputer.fit_transform(x_raw)
    scaler = StandardScaler()
    x = scaler.fit_transform(x_imp)

    sil_rows: list[dict[str, Any]] = []
    best_k = 2
    best_s = -1.0
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=25)
        labels = km.fit_predict(x)
        s = silhouette_score(x, labels)
        sil_rows.append({"k": k, "silhouette": float(s)})
        if s > best_s:
            best_s = float(s)
            best_k = k

    km = KMeans(n_clusters=best_k, random_state=42, n_init=25)
    labels = km.fit_predict(x)
    profile = profile.copy()
    profile["cluster"] = labels.astype(int)

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(x)
    profile["pc1"] = pcs[:, 0]
    profile["pc2"] = pcs[:, 1]
    profile["cluster_label"] = profile["cluster"].map(lambda v: f"C{int(v)}")

    sil = pd.DataFrame(sil_rows)
    return profile, sil, best_k


def build_outcome_country_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel[["country_iso3_code"]].drop_duplicates().copy()
    post = panel[panel["year"].between(2019, 2022)].copy()
    pre = panel[panel["year"].between(2012, 2017)].copy()

    def gmean(df: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
        return df.groupby("country_iso3_code", as_index=False)[col].mean().rename(
            columns={col: name}
        )

    out = out.merge(
        gmean(post, "delta_tiva_fexgr_dva_share", "dv1_post_mean"),
        on="country_iso3_code",
        how="left",
    )
    out = out.merge(
        gmean(pre[pre["year"] >= 2013], "delta_tiva_fexgr_dva_share", "dv1_pre_mean"),
        on="country_iso3_code",
        how="left",
    )
    out["dv1_change_post_minus_pre"] = out["dv1_post_mean"] - out["dv1_pre_mean"]

    out = out.merge(
        gmean(post, "export_recovery_index", "dv2_post_mean"),
        on="country_iso3_code",
        how="left",
    )
    out = out.merge(
        gmean(pre, "export_recovery_index", "dv2_pre_mean"),
        on="country_iso3_code",
        how="left",
    )
    out["dv2_change_post_minus_pre"] = out["dv2_post_mean"] - out["dv2_pre_mean"]

    out = out.merge(
        gmean(
            post, "partner_diversification_1_minus_hhi", "dv3_post_mean"
        ),
        on="country_iso3_code",
        how="left",
    )
    out = out.merge(
        gmean(pre, "partner_diversification_1_minus_hhi", "dv3_pre_mean"),
        on="country_iso3_code",
        how="left",
    )
    out["dv3_change_post_minus_pre"] = out["dv3_post_mean"] - out["dv3_pre_mean"]
    return out


def save_plots(df: pd.DataFrame, sil: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=sil, x="k", y="silhouette", marker="o")
    plt.title("Silhouette Score by Number of Clusters")
    plt.tight_layout()
    plt.savefig(out_dir / "silhouette_by_k.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="pc1", y="pc2", hue="cluster_label", palette="tab10", s=50)
    plt.title("Country Clusters in PCA Space (Pre-shock Profile)")
    plt.tight_layout()
    plt.savefig(out_dir / "clusters_pca_scatter.png", dpi=180)
    plt.close()

    for col, title, fn in [
        ("dv2_post_mean", "DV2 (Export Recovery) by Cluster", "dv2_by_cluster.png"),
        (
            "dv3_post_mean",
            "DV3 (Partner Diversification) by Cluster",
            "dv3_by_cluster.png",
        ),
        ("dv1_post_mean", "DV1 (GVC Linkage Change) by Cluster", "dv1_by_cluster.png"),
    ]:
        plt.figure(figsize=(7, 4.5))
        sns.boxplot(data=df, x="cluster_label", y=col)
        sns.stripplot(data=df, x="cluster_label", y=col, color="black", alpha=0.25, size=2)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dir / fn, dpi=180)
        plt.close()

    feat_cols = [
        "pre_eci",
        "pre_coi",
        "pre_log_gdp_pc",
        "pre_trade_open",
        "pre_wgi",
        "pre_rents",
        "pre_tariff",
        "pre_us_china_intensity",
    ]
    cluster_means = df.groupby("cluster_label", as_index=True)[feat_cols].mean()
    cluster_means_z = (cluster_means - cluster_means.mean()) / cluster_means.std(ddof=0)
    plt.figure(figsize=(10, 4.8))
    sns.heatmap(cluster_means_z, cmap="coolwarm", center=0, annot=False)
    plt.title("Cluster Structural Profiles (z-scored Means)")
    plt.tight_layout()
    plt.savefig(out_dir / "cluster_profile_heatmap.png", dpi=180)
    plt.close()


def interaction_screen(panel: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    dvars = {
        "DV2_Export_Recovery": "export_recovery_index",
        "DV3_Partner_Diversification": "partner_diversification_1_minus_hhi",
    }
    moderators = [
        "coi",
        "log_gdp_pc",
        "wdi_trade_openness_pct_gdp",
        "wgi_institutional_quality_composite",
        "wdi_natural_resource_rents_pct_gdp",
        "wdi_tariff_applied_weighted_mean_all_products_pct",
        "gpr_for_model_annual",
    ]

    rows: list[dict[str, Any]] = []
    for dv_name, dv_col in dvars.items():
        for mod in moderators:
            req = [
                dv_col,
                "eci",
                "post",
                "exposed",
                mod,
                "us_china_trade_intensity_pre",
                "country_iso3_code",
                "year",
            ]
            df = panel.dropna(subset=req).copy()
            if len(df) < 400:
                continue
            df["z_eci"] = zscore(df["eci"])
            df["z_mod"] = zscore(df[mod])
            df["z_intensity"] = zscore(df["us_china_trade_intensity_pre"])
            df["pe"] = df["post"] * df["exposed"]

            # Keep parsimonious controls for stable screening.
            formula = (
                f"{dv_col} ~ z_eci + pe + z_eci:pe + z_mod + z_eci:pe:z_mod + z_intensity"
                " + C(country_iso3_code) + C(year)"
            )
            fit = smf.ols(formula=formula, data=df).fit(
                cov_type="cluster", cov_kwds={"groups": df["country_iso3_code"]}
            )
            term = "z_eci:pe:z_mod"
            coef = float(fit.params.get(term, np.nan))
            pval = float(fit.pvalues.get(term, np.nan))
            rows.append(
                {
                    "dv": dv_name,
                    "moderator": mod,
                    "coef_triple": coef,
                    "pvalue_triple": pval,
                    "abs_coef": abs(coef),
                    "n_obs": int(fit.nobs),
                    "n_countries": int(df["country_iso3_code"].nunique()),
                }
            )
    out = pd.DataFrame(rows).sort_values(["dv", "pvalue_triple", "abs_coef"])
    out.to_csv(out_dir / "moderator_interaction_screen.csv", index=False)
    return out


def write_summary(
    merged: pd.DataFrame, sil: pd.DataFrame, best_k: int, screen: pd.DataFrame, out_dir: Path
) -> None:
    cluster_summary = (
        merged.groupby("cluster_label", as_index=False)
        .agg(
            n_countries=("country_iso3_code", "nunique"),
            dv1_post_mean=("dv1_post_mean", "mean"),
            dv2_post_mean=("dv2_post_mean", "mean"),
            dv3_post_mean=("dv3_post_mean", "mean"),
            pre_eci=("pre_eci", "mean"),
            pre_coi=("pre_coi", "mean"),
            pre_wgi=("pre_wgi", "mean"),
        )
        .sort_values("cluster_label")
    )
    cluster_summary.to_csv(out_dir / "cluster_outcome_summary.csv", index=False)

    top = (
        screen.groupby("dv", as_index=False)
        .apply(lambda g: g.nsmallest(3, "pvalue_triple"))
        .reset_index(drop=True)
    )
    top.to_csv(out_dir / "top_moderator_candidates.csv", index=False)

    md = [
        "# Pattern Discovery Summary",
        "",
        f"- Selected clusters (silhouette): `{best_k}`",
        f"- Countries clustered: `{merged['country_iso3_code'].nunique()}`",
        "",
        "Files generated:",
        "- `silhouette_by_k.png`",
        "- `clusters_pca_scatter.png`",
        "- `cluster_profile_heatmap.png`",
        "- `dv1_by_cluster.png`",
        "- `dv2_by_cluster.png`",
        "- `dv3_by_cluster.png`",
        "- `cluster_outcome_summary.csv`",
        "- `moderator_interaction_screen.csv`",
        "- `top_moderator_candidates.csv`",
    ]
    (out_dir / "pattern_discovery_summary.md").write_text("\n".join(md), encoding="utf-8")

    meta = {
        "best_k": int(best_k),
        "silhouette": sil.to_dict(orient="records"),
        "n_countries": int(merged["country_iso3_code"].nunique()),
    }
    (out_dir / "pattern_discovery_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clustering and visualization pattern discovery.")
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
    out_dir = base_dir / "reports" / "exploratory_patterns"
    ensure_dirs(out_dir)

    panel = pd.read_csv(panel_path)
    profile = build_country_profile(panel)
    prof_clustered, sil, best_k = choose_k_and_cluster(profile)
    outcomes = build_outcome_country_metrics(panel)
    merged = prof_clustered.merge(outcomes, on="country_iso3_code", how="left")

    merged.to_csv(out_dir / "country_clusters_with_outcomes.csv", index=False)
    sil.to_csv(out_dir / "silhouette_scores.csv", index=False)

    save_plots(merged, sil, out_dir)
    screen = interaction_screen(panel, out_dir)
    write_summary(merged, sil, best_k, screen, out_dir)

    print("Pattern discovery completed.")
    print(f"Best k: {best_k}")
    print(f"Countries: {merged['country_iso3_code'].nunique()}")
    print(f"Screen rows: {len(screen)}")


if __name__ == "__main__":
    main()
