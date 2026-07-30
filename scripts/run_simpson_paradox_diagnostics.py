from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


DV_MAP = {
    "H1_DV1_Stability": "gvc_linkage_stability",
    "H2_DV2_ExportRecovery": "export_recovery_index",
    "H3_DV3_Diversification": "partner_diversification_1_minus_hhi",
}

CONTROL_Z = [
    "z_log_gdp_pc",
    "z_trade_open",
    "z_wgi",
    "z_rents",
    "z_gpr",
    "z_intensity",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def zscore(series: pd.Series) -> pd.Series:
    mu = series.mean(skipna=True)
    sigma = series.std(skipna=True, ddof=0)
    if pd.isna(sigma) or sigma == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - mu) / sigma


def prepare_base_panel(base_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(base_dir / "data" / "processed" / "regression_panel_2012_2022.csv")
    df = df[df["year"].between(2012, 2022)].copy()
    df["post_paper"] = df["year"].between(2019, 2022).astype(int)
    df["pe"] = df["post_paper"] * df["exposed"]
    df["gvc_linkage_stability"] = -df["delta_tiva_fexgr_dva_share"].abs()

    # z scores
    for col, zcol in [
        ("eci", "z_eci"),
        ("log_gdp_pc", "z_log_gdp_pc"),
        ("wdi_trade_openness_pct_gdp", "z_trade_open"),
        ("wgi_institutional_quality_composite", "z_wgi"),
        ("wdi_natural_resource_rents_pct_gdp", "z_rents"),
        ("gpr_for_model_annual", "z_gpr"),
        ("us_china_trade_intensity_pre", "z_intensity"),
    ]:
        df[zcol] = zscore(df[col])

    # strata
    csize = (
        df[["country_iso3_code", "baseline_export_2015_2017"]]
        .dropna()
        .drop_duplicates()
        .groupby("country_iso3_code", as_index=False)["baseline_export_2015_2017"]
        .mean()
    )
    if len(csize) > 10:
        csize["size_q"] = pd.qcut(
            csize["baseline_export_2015_2017"],
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop",
        )
        df = df.merge(csize[["country_iso3_code", "size_q"]], on="country_iso3_code", how="left")

    if df["log_gdp_pc"].notna().sum() > 20:
        df["income_q"] = pd.qcut(
            df["log_gdp_pc"],
            q=4,
            labels=["I1", "I2", "I3", "I4"],
            duplicates="drop",
        )
    if df["eci"].notna().sum() > 20:
        df["eci_t"] = pd.qcut(df["eci"], q=3, labels=["E1", "E2", "E3"], duplicates="drop")

    return df


def fit_interaction(
    df: pd.DataFrame,
    outcome: str,
    with_controls_year_fe: bool,
) -> tuple[float, float, int]:
    if with_controls_year_fe:
        f = (
            f"{outcome} ~ z_eci + pe + z_eci:pe + "
            + " + ".join(CONTROL_Z)
            + " + C(year)"
        )
    else:
        f = f"{outcome} ~ z_eci + pe + z_eci:pe"
    fit = smf.ols(f, data=df).fit()
    return (
        float(fit.params.get("z_eci:pe", np.nan)),
        float(fit.pvalues.get("z_eci:pe", np.nan)),
        int(fit.nobs),
    )


def run_sign_reversal_screen(
    df: pd.DataFrame,
    out_dir: Path,
    with_controls_year_fe: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    strata = ["income_q", "size_q", "eci_t", "exposed"]

    for dv_label, outcome in DV_MAP.items():
        req = [outcome, "z_eci", "pe"]
        if with_controls_year_fe:
            req += CONTROL_Z
        d = df.dropna(subset=req).copy()
        if len(d) < 100:
            continue
        pooled_coef, pooled_p, pooled_n = fit_interaction(d, outcome, with_controls_year_fe)
        rows.append(
            {
                "dv": dv_label,
                "stratifier": "_pooled",
                "group": "ALL",
                "coef": pooled_coef,
                "pvalue": pooled_p,
                "n": pooled_n,
                "pooled_coef": pooled_coef,
                "reversal_vs_pooled": 0,
            }
        )
        for s in strata:
            if s not in d.columns:
                continue
            for lv in d[s].dropna().unique():
                sub = d[d[s] == lv].copy()
                if len(sub) < 80 or sub["pe"].nunique() < 2:
                    continue
                try:
                    coef, pv, n = fit_interaction(sub, outcome, with_controls_year_fe)
                except Exception:
                    continue
                rev = int(
                    np.sign(coef) != 0
                    and np.sign(pooled_coef) != 0
                    and np.sign(coef) != np.sign(pooled_coef)
                )
                rows.append(
                    {
                        "dv": dv_label,
                        "stratifier": s,
                        "group": str(lv),
                        "coef": coef,
                        "pvalue": pv,
                        "n": n,
                        "pooled_coef": pooled_coef,
                        "reversal_vs_pooled": rev,
                    }
                )

    raw = pd.DataFrame(rows)
    summary_rows = []
    for (dv, s), g in raw[raw["stratifier"] != "_pooled"].groupby(["dv", "stratifier"]):
        if len(g) == 0:
            continue
        wmean = np.average(g["coef"], weights=np.maximum(g["n"], 1))
        pooled = float(g["pooled_coef"].iloc[0])
        summary_rows.append(
            {
                "dv": dv,
                "stratifier": s,
                "n_groups": int(len(g)),
                "pooled_coef": pooled,
                "weighted_mean_group_coef": float(wmean),
                "sign_pooled": int(np.sign(pooled)),
                "sign_weighted_group_mean": int(np.sign(wmean)),
                "reversal_groups_count": int(g["reversal_vs_pooled"].sum()),
                "reversal_groups_share": float(g["reversal_vs_pooled"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)

    tag = "controls_yearfe" if with_controls_year_fe else "raw"
    raw.to_csv(out_dir / f"simpson_screen_{tag}_raw.csv", index=False)
    summary.to_csv(out_dir / f"simpson_screen_{tag}_summary.csv", index=False)
    return raw, summary


def run_within_between_decomposition(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    d = df.copy()
    d["eci_mean_country"] = d.groupby("country_iso3_code")["eci"].transform("mean")
    d["eci_within"] = d["eci"] - d["eci_mean_country"]
    d["z_eci_within"] = zscore(d["eci_within"])
    d["z_eci_between"] = zscore(d["eci_mean_country"])

    rows: list[dict[str, Any]] = []
    for dv_label, outcome in DV_MAP.items():
        req = [
            outcome,
            "pe",
            "z_eci_within",
            "z_eci_between",
            "country_iso3_code",
            "year",
        ] + CONTROL_Z
        sub = d.dropna(subset=req).copy()
        if len(sub) < 200:
            continue
        f = (
            f"{outcome} ~ pe + z_eci_within + z_eci_between + pe:z_eci_within + pe:z_eci_between + "
            + " + ".join(CONTROL_Z)
            + " + C(country_iso3_code) + C(year)"
        )
        fit = smf.ols(f, data=sub).fit(
            cov_type="cluster",
            cov_kwds={"groups": sub["country_iso3_code"]},
        )
        cw = float(fit.params.get("pe:z_eci_within", np.nan))
        cb = float(fit.params.get("pe:z_eci_between", np.nan))
        rows.append(
            {
                "dv": dv_label,
                "coef_within": cw,
                "p_within": float(fit.pvalues.get("pe:z_eci_within", np.nan)),
                "coef_between": cb,
                "p_between": float(fit.pvalues.get("pe:z_eci_between", np.nan)),
                "sign_opposite": int(
                    np.sign(cw) != 0 and np.sign(cb) != 0 and np.sign(cw) != np.sign(cb)
                ),
                "n_obs": int(fit.nobs),
                "n_countries": int(sub["country_iso3_code"].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "simpson_within_between.csv", index=False)
    return out


def write_summary(
    out_dir: Path,
    raw_summary: pd.DataFrame,
    ctrl_summary: pd.DataFrame,
    wb_df: pd.DataFrame,
) -> None:
    lines = [
        "# Simpson-Paradox Diagnostics",
        "",
        "Interpretation guide:",
        "- `reversal_groups_share` near 1.0 means many subgroup coefficients have the opposite sign of pooled.",
        "- `sign_opposite=1` in within-between decomposition means within-country and between-country effects conflict.",
        "",
    ]
    if not raw_summary.empty:
        lines.append("## Raw interaction screen (no controls)")
        for _, r in raw_summary.iterrows():
            lines.append(
                f"- {r['dv']} | {r['stratifier']}: reversals={int(r['reversal_groups_count'])}/{int(r['n_groups'])}, "
                f"pooled={r['pooled_coef']:.4f}, weighted-group={r['weighted_mean_group_coef']:.4f}"
            )
        lines.append("")
    if not ctrl_summary.empty:
        lines.append("## Controlled + year-FE interaction screen")
        for _, r in ctrl_summary.iterrows():
            lines.append(
                f"- {r['dv']} | {r['stratifier']}: reversals={int(r['reversal_groups_count'])}/{int(r['n_groups'])}, "
                f"pooled={r['pooled_coef']:.4f}, weighted-group={r['weighted_mean_group_coef']:.4f}"
            )
        lines.append("")
    if not wb_df.empty:
        lines.append("## Within vs between decomposition")
        for _, r in wb_df.iterrows():
            lines.append(
                f"- {r['dv']}: within={r['coef_within']:.4f} (p={r['p_within']:.4g}), "
                f"between={r['coef_between']:.4f} (p={r['p_between']:.4g}), opposite_sign={int(r['sign_opposite'])}"
            )
        lines.append("")
    lines.append("Conclusion: Simpson-like aggregation risk exists when pooled and subgroup/within-between signs conflict.")
    (out_dir / "simpson_diagnostics_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Simpson paradox diagnostics for resilience hypotheses.")
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project base directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_dir = base_dir / "reports" / "simpson_diagnostics"
    ensure_dir(out_dir)

    panel = prepare_base_panel(base_dir)
    raw_raw, raw_summary = run_sign_reversal_screen(panel, out_dir, with_controls_year_fe=False)
    ctrl_raw, ctrl_summary = run_sign_reversal_screen(panel, out_dir, with_controls_year_fe=True)
    wb = run_within_between_decomposition(panel, out_dir)

    write_summary(out_dir, raw_summary, ctrl_summary, wb)

    meta = {
        "rows_panel": int(len(panel)),
        "countries_panel": int(panel["country_iso3_code"].nunique()),
        "raw_rows": int(len(raw_raw)),
        "ctrl_rows": int(len(ctrl_raw)),
        "within_between_rows": int(len(wb)),
    }
    (out_dir / "simpson_diagnostics_metadata.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print("Simpson diagnostics completed.")
    print(f"Output directory: {out_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
