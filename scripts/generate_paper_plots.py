from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTCOME_LABELS = {
    "DV1_GVC_Linkage_Stability": "GVC Linkage Stability",
    "DV2_Export_Recovery": "Export Recovery",
    "DV3_Partner_Diversification": "Partner Diversification",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_confirmatory_plot(
    primary_csv: Path,
    moderation_csv: Path,
    out_png: Path,
    out_pdf: Path,
) -> None:
    p = pd.read_csv(primary_csv)
    m = pd.read_csv(moderation_csv)

    main = p.rename(columns={"hypothesis_id": "hypothesis"}).copy()
    main["term_label"] = "Main Effect (H1/H2/H3): zECI x PE"
    main["ci_low"] = main["boot_ci_low_2p5"]
    main["ci_high"] = main["boot_ci_high_97p5"]
    main["pvalue"] = main["p_boot"]

    mod = m.rename(columns={"hypothesis_id": "hypothesis"}).copy()
    mod["term_label"] = np.where(
        mod["hypothesis"] == "H4",
        "H4 Moderation: zECI x PE x zCOI",
        "H5 Moderation: zECI x PE x zGPR",
    )
    mod["ci_low"] = mod["coef"] - 1.96 * mod["se"]
    mod["ci_high"] = mod["coef"] + 1.96 * mod["se"]
    mod["pvalue"] = mod["p_cluster"]

    use_cols = ["dv", "hypothesis", "term_label", "coef", "ci_low", "ci_high", "pvalue"]
    all_df = pd.concat([main[use_cols], mod[use_cols]], ignore_index=True)
    all_df["dv_label"] = all_df["dv"].map(OUTCOME_LABELS)

    order_terms = [
        "Main Effect (H1/H2/H3): zECI x PE",
        "H4 Moderation: zECI x PE x zCOI",
        "H5 Moderation: zECI x PE x zGPR",
    ]
    colors = {
        order_terms[0]: "#1f77b4",
        order_terms[1]: "#2ca02c",
        order_terms[2]: "#ff7f0e",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, dv in zip(axes, OUTCOME_LABELS.keys()):
        sub = all_df[all_df["dv"] == dv].copy()
        y_map = {t: i for i, t in enumerate(order_terms[::-1])}
        for _, r in sub.iterrows():
            y = y_map[r["term_label"]]
            c = colors[r["term_label"]]
            xerr = np.array([[r["coef"] - r["ci_low"]], [r["ci_high"] - r["coef"]]])
            ax.errorbar(
                x=r["coef"],
                y=y,
                xerr=xerr,
                fmt="o",
                color=c,
                ecolor=c,
                capsize=3,
                markersize=6,
                lw=1.8,
            )
            if pd.notna(r["pvalue"]) and float(r["pvalue"]) < 0.05:
                ax.scatter(r["coef"], y, s=70, facecolors="none", edgecolors="black", linewidths=1.2)

        ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
        ax.set_yticks([0, 1, 2], labels=order_terms[::-1])
        ax.set_title(OUTCOME_LABELS[dv], fontsize=11)
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle(
        "Figure X. Confirmatory Estimates for H1-H5\nPoint Estimates with 95% Confidence Intervals",
        fontsize=12,
    )
    axes[0].set_xlabel("Coefficient")
    axes[1].set_xlabel("Coefficient")
    axes[2].set_xlabel("Coefficient")

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def build_heterogeneity_plot(
    income_h1_h3_csv: Path,
    income_h4_h5_csv: Path,
    eci_regime_csv: Path,
    out_png: Path,
    out_pdf: Path,
) -> None:
    h1h3 = pd.read_csv(income_h1_h3_csv)
    h4h5 = pd.read_csv(income_h4_h5_csv)
    reg = pd.read_csv(eci_regime_csv)

    e1 = h1h3[(h1h3["sample"] == "UMC") & (h1h3["hypothesis"] == "H1")].iloc[0]
    e2 = reg[(reg["dv"] == "DV2_Export_Recovery") & (reg["eci_regime"] == "low_eci")].iloc[0]
    e3 = h4h5[
        (h4h5["sample"] == "HIC")
        & (h4h5["hypothesis"] == "H5")
        & (h4h5["outcome"] == "partner_diversification_1_minus_hhi")
    ].iloc[0]

    rows = [
        {
            "label": "E1: H1 in UMC (GVC Stability)",
            "coef": float(e1["coef_z_eci_pe"]),
            "se": float(e1["se"]),
            "pvalue": float(e1["pvalue"]),
        },
        {
            "label": "E2: H2 in Low-ECI Regime (Export Recovery)",
            "coef": float(e2["coef_z_eci_pe"]),
            "se": float(e2["se"]),
            "pvalue": float(e2["pvalue"]),
        },
        {
            "label": "E3: H5 in HIC (Partner Diversification)",
            "coef": float(e3["coef"]),
            "se": float(e3["se"]),
            "pvalue": float(e3["pvalue"]),
        },
    ]
    df = pd.DataFrame(rows)
    df["ci_low"] = df["coef"] - 1.96 * df["se"]
    df["ci_high"] = df["coef"] + 1.96 * df["se"]
    df = df.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10.5, 4.6), constrained_layout=True)
    y = np.arange(len(df))
    xerr = np.vstack([df["coef"] - df["ci_low"], df["ci_high"] - df["coef"]])
    ax.errorbar(
        x=df["coef"],
        y=y,
        xerr=xerr,
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        capsize=3,
        lw=2,
        markersize=7,
    )
    for i, r in df.iterrows():
        if r["pvalue"] < 0.05:
            ax.scatter(r["coef"], i, s=80, facecolors="none", edgecolors="black", linewidths=1.2)

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y, labels=df["label"])
    ax.set_xlabel("Coefficient")
    ax.set_title(
        "Figure Y. Exploratory Heterogeneity in the ECI-Resilience Relationship\n"
        "Selected E1-E3 subgroup estimates with 95% confidence intervals",
        fontsize=12,
    )
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def build_event_study_plot(
    event_study_csv: Path,
    joint_tests_csv: Path,
    out_png: Path,
    out_pdf: Path,
) -> None:
    es = pd.read_csv(event_study_csv)
    jt = pd.read_csv(joint_tests_csv)

    if "dv" in es.columns:
        sub = es[(es["dv"] == "DV2_Export_Recovery") & (es["coef"].notna())].copy()
    else:
        sub = es[es["coef"].notna()].copy()
    sub = sub.sort_values("event_time")

    if "dv" in jt.columns:
        p_pre = jt[(jt["dv"] == "DV2_Export_Recovery") & (jt["test"] == "pretrend_joint_zero")][
            "pvalue"
        ].iloc[0]
        p_post = jt[(jt["dv"] == "DV2_Export_Recovery") & (jt["test"] == "post_joint_zero")][
            "pvalue"
        ].iloc[0]
    else:
        p_pre = jt[jt["test"] == "pretrend_joint_zero"]["pvalue"].iloc[0]
        p_post = jt[jt["test"] == "post_joint_zero"]["pvalue"].iloc[0]

    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    ax.plot(sub["event_time"], sub["coef"], color="#1f77b4", marker="o", lw=2)
    ax.fill_between(
        sub["event_time"].to_numpy(dtype=float),
        sub["ci_low"].to_numpy(dtype=float),
        sub["ci_high"].to_numpy(dtype=float),
        color="#1f77b4",
        alpha=0.2,
    )
    ax.axhline(0.0, color="black", lw=1, linestyle="--")
    ax.axvline(0.0, color="firebrick", lw=1, linestyle="--")
    ax.set_xlabel("Event Time (Years Relative to Shock)")
    ax.set_ylabel("Coefficient on event_time x exposed x zECI")
    ax.set_title(
        "Appendix Figure A1. Event-Study Diagnostic: Export Recovery\n"
        f"Pretrend joint p={p_pre:.3f}; Post-period joint p={p_post:.3f}",
        fontsize=12,
    )
    ax.grid(alpha=0.25)

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def write_figure_notes(out_md: Path) -> None:
    lines = [
        "# Figure Notes",
        "",
        "## Figure X (Confirmatory Coefficient Plot)",
        "- Point estimates and 95% confidence intervals for main and moderation terms across the three outcomes.",
        "- Main H1-H3 intervals use bootstrap percentile 95% CI from the 999-rep confirmatory run.",
        "- H4-H5 intervals use normal-approximation CI from clustered standard errors.",
        "- Hollow markers indicate p<0.05.",
        "",
        "## Figure Y (Exploratory Heterogeneity E1-E3)",
        "- Shows three focal subgroup findings used in exploratory interpretation:",
        "  1) H1 positive in upper-middle-income countries,",
        "  2) H2 negative in low-ECI regime,",
        "  3) H5 negative moderation in high-income countries for diversification.",
        "",
        "## Appendix Figure A1 (Event Study, Export Recovery)",
        "- Diagnostic plot of event-time coefficients around the tariff shock for export recovery.",
        "- Included as transparency/identification support in appendix.",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready figures.")
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project base directory.",
    )
    parser.add_argument(
        "--out-subdir",
        default="figures_paper",
        help="Output subdirectory under reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_dir = base_dir / "reports" / args.out_subdir
    ensure_dir(out_dir)

    profile_dir = base_dir / "reports" / "confirmatory_hypotheses_profile_knn_no_rents_999"
    class_dir = base_dir / "reports" / "class_based_tests"
    add_dir = base_dir / "reports" / "additional_ideas"

    build_confirmatory_plot(
        primary_csv=profile_dir / "confirmatory_primary_tests.csv",
        moderation_csv=profile_dir / "confirmatory_moderation_tests.csv",
        out_png=out_dir / "figureX_confirmatory_coefficients.png",
        out_pdf=out_dir / "figureX_confirmatory_coefficients.pdf",
    )
    build_heterogeneity_plot(
        income_h1_h3_csv=class_dir / "h1_h3_by_official_income_group.csv",
        income_h4_h5_csv=class_dir / "h4_h5_by_official_income_group.csv",
        eci_regime_csv=add_dir / "regime_eci_tercile_results.csv",
        out_png=out_dir / "figureY_exploratory_heterogeneity_E1_E3.png",
        out_pdf=out_dir / "figureY_exploratory_heterogeneity_E1_E3.pdf",
    )
    profile_es_coef = out_dir / "appendix_event_study_export_recovery_profile_knn_coefficients.csv"
    profile_es_joint = out_dir / "appendix_event_study_export_recovery_profile_knn_joint_tests.csv"
    if profile_es_coef.exists() and profile_es_joint.exists():
        event_csv = profile_es_coef
        joint_csv = profile_es_joint
        out_png = out_dir / "appendix_figureA1_event_study_export_recovery_profile_knn.png"
        out_pdf = out_dir / "appendix_figureA1_event_study_export_recovery_profile_knn.pdf"
    else:
        event_csv = add_dir / "event_study_coefficients.csv"
        joint_csv = add_dir / "event_study_joint_tests.csv"
        out_png = out_dir / "appendix_figureA1_event_study_export_recovery.png"
        out_pdf = out_dir / "appendix_figureA1_event_study_export_recovery.pdf"

    build_event_study_plot(
        event_study_csv=event_csv,
        joint_tests_csv=joint_csv,
        out_png=out_png,
        out_pdf=out_pdf,
    )
    write_figure_notes(out_dir / "figure_notes.md")

    print(f"Generated figures in: {out_dir}")
    for p in sorted(out_dir.glob("*")):
        print(p.name)


if __name__ == "__main__":
    main()
