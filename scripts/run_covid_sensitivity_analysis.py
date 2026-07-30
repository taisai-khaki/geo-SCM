from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


OUTCOME_MAP = {
    "H1": ("DV1_GVC_Linkage_Stability", "gvc_linkage_stability"),
    "H2": ("DV2_Export_Recovery", "export_recovery_index"),
    "H3": ("DV3_Partner_Diversification", "partner_diversification_1_minus_hhi"),
}


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
    df["gvc_linkage_stability"] = -df["delta_tiva_fexgr_dva_share"].abs()
    df["post_paper"] = df["year"].between(2019, 2022).astype(int)
    df["pe"] = df["post_paper"] * df["exposed"]
    df["covid_period"] = df["year"].between(2020, 2021).astype(int)

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


def controls(include_rents_main: bool) -> list[str]:
    out = ["z_log_gdp_pc", "z_trade_open", "z_wgi", "z_gpr", "z_intensity"]
    if include_rents_main:
        out.insert(3, "z_rents")
    return out


def prepare_sample(
    panel: pd.DataFrame,
    outcome_col: str,
    include_rents_main: bool,
    drop_years: set[int] | None = None,
) -> pd.DataFrame:
    df = panel.copy()
    if drop_years:
        df = df[~df["year"].isin(drop_years)].copy()
    req = [outcome_col, "z_eci", "pe", "country_iso3_code", "year"] + controls(include_rents_main)
    return df.dropna(subset=req).copy()


def primary_formula(
    outcome_col: str,
    include_rents_main: bool,
    covid_interaction: bool = False,
) -> str:
    rhs = ["z_eci", "pe", "z_eci:pe"] + controls(include_rents_main)
    if covid_interaction:
        # Year FE absorb the covid_period main effect.
        # Keep interaction terms to test differential COVID contamination.
        rhs.extend(["exposed:covid_period", "z_eci:pe:covid_period"])
    rhs.extend(["C(country_iso3_code)", "C(year)"])
    return f"{outcome_col} ~ " + " + ".join(rhs)


def moderation_formula(
    outcome_col: str,
    moderator: str,
    include_rents_main: bool,
    covid_interaction: bool = False,
) -> str:
    if moderator == "coi":
        main = "z_eci*pe*z_coi"
    else:
        main = "z_eci*pe*z_gpr"
    rhs = [main] + controls(include_rents_main)
    if covid_interaction:
        rhs.extend(["exposed:covid_period", "z_eci:pe:covid_period"])
    rhs.extend(["C(country_iso3_code)", "C(year)"])
    return f"{outcome_col} ~ " + " + ".join(rhs)


def fit_clustered(formula: str, df: pd.DataFrame) -> Any:
    return smf.ols(formula=formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["country_iso3_code"]},
    )


def run_primary_block(
    panel: pd.DataFrame,
    include_rents_main: bool,
    spec_name: str,
    drop_years: set[int] | None,
    covid_interaction: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hid, (dv_label, outcome_col) in OUTCOME_MAP.items():
        df = prepare_sample(
            panel=panel,
            outcome_col=outcome_col,
            include_rents_main=include_rents_main,
            drop_years=drop_years,
        )
        if df.empty:
            continue
        f = primary_formula(
            outcome_col=outcome_col,
            include_rents_main=include_rents_main,
            covid_interaction=covid_interaction,
        )
        fit = fit_clustered(f, df)
        rows.append(
            {
                "spec": spec_name,
                "hypothesis_id": hid,
                "dv": dv_label,
                "outcome_col": outcome_col,
                "term": "z_eci:pe",
                "coef": float(fit.params.get("z_eci:pe", np.nan)),
                "se": float(fit.bse.get("z_eci:pe", np.nan)),
                "p_cluster": float(fit.pvalues.get("z_eci:pe", np.nan)),
                "n_obs": int(fit.nobs),
                "n_countries": int(df["country_iso3_code"].nunique()),
                "year_min": int(df["year"].min()),
                "year_max": int(df["year"].max()),
                "covid_contamination_coef": float(
                    fit.params.get("z_eci:pe:covid_period", np.nan)
                ),
                "covid_contamination_p": float(
                    fit.pvalues.get("z_eci:pe:covid_period", np.nan)
                ),
            }
        )
    return rows


def run_moderation_block(
    panel: pd.DataFrame,
    include_rents_main: bool,
    spec_name: str,
    drop_years: set[int] | None,
    covid_interaction: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, (dv_label, outcome_col) in OUTCOME_MAP.items():
        df = prepare_sample(
            panel=panel,
            outcome_col=outcome_col,
            include_rents_main=include_rents_main,
            drop_years=drop_years,
        )
        if df.empty:
            continue

        # H4
        df_h4 = df.dropna(subset=["z_coi"]).copy()
        if not df_h4.empty:
            f_h4 = moderation_formula(
                outcome_col=outcome_col,
                moderator="coi",
                include_rents_main=include_rents_main,
                covid_interaction=covid_interaction,
            )
            fit_h4 = fit_clustered(f_h4, df_h4)
            rows.append(
                {
                    "spec": spec_name,
                    "hypothesis_id": "H4",
                    "dv": dv_label,
                    "outcome_col": outcome_col,
                    "term": "z_eci:pe:z_coi",
                    "coef": float(fit_h4.params.get("z_eci:pe:z_coi", np.nan)),
                    "se": float(fit_h4.bse.get("z_eci:pe:z_coi", np.nan)),
                    "p_cluster": float(fit_h4.pvalues.get("z_eci:pe:z_coi", np.nan)),
                    "n_obs": int(fit_h4.nobs),
                    "n_countries": int(df_h4["country_iso3_code"].nunique()),
                    "year_min": int(df_h4["year"].min()),
                    "year_max": int(df_h4["year"].max()),
                    "covid_contamination_coef": float(
                        fit_h4.params.get("z_eci:pe:covid_period", np.nan)
                    ),
                    "covid_contamination_p": float(
                        fit_h4.pvalues.get("z_eci:pe:covid_period", np.nan)
                    ),
                }
            )

        # H5
        f_h5 = moderation_formula(
            outcome_col=outcome_col,
            moderator="gpr",
            include_rents_main=include_rents_main,
            covid_interaction=covid_interaction,
        )
        fit_h5 = fit_clustered(f_h5, df)
        rows.append(
            {
                "spec": spec_name,
                "hypothesis_id": "H5",
                "dv": dv_label,
                "outcome_col": outcome_col,
                "term": "z_eci:pe:z_gpr",
                "coef": float(fit_h5.params.get("z_eci:pe:z_gpr", np.nan)),
                "se": float(fit_h5.bse.get("z_eci:pe:z_gpr", np.nan)),
                "p_cluster": float(fit_h5.pvalues.get("z_eci:pe:z_gpr", np.nan)),
                "n_obs": int(fit_h5.nobs),
                "n_countries": int(df["country_iso3_code"].nunique()),
                "year_min": int(df["year"].min()),
                "year_max": int(df["year"].max()),
                "covid_contamination_coef": float(
                    fit_h5.params.get("z_eci:pe:covid_period", np.nan)
                ),
                "covid_contamination_p": float(
                    fit_h5.pvalues.get("z_eci:pe:covid_period", np.nan)
                ),
            }
        )
    return rows


def write_summary(
    out_dir: Path,
    primary_df: pd.DataFrame,
    mod_df: pd.DataFrame,
    include_rents_main: bool,
) -> None:
    lines = [
        "# COVID-19 Sensitivity Analysis",
        "",
        "Primary confirmatory base aligned to legacy-corrected setup (unless stated otherwise).",
        f"- Includes natural resource rents in controls: `{include_rents_main}`",
        "",
        "## Specifications",
        "- `baseline_full`: all years in the estimation window.",
        "- `exclude_2020`: drops pandemic onset year.",
        "- `exclude_2020_2021`: drops both pandemic years.",
        "- `covid_interaction_check`: full sample with `exposed:covid_period` and `z_eci:pe:covid_period` contamination checks.",
        "",
        "## Key Terms",
        "- H1-H3 key term: `z_eci:pe`",
        "- H4 key term: `z_eci:pe:z_coi`",
        "- H5 key term: `z_eci:pe:z_gpr`",
        "",
    ]

    if not primary_df.empty:
        lines.append("## H1-H3 Snapshot")
        for _, r in primary_df.iterrows():
            lines.append(
                f"- {r['spec']} | {r['hypothesis_id']} ({r['dv']}): "
                f"coef={r['coef']:.4f}, p={r['p_cluster']:.4g}, N={int(r['n_obs'])}, years={int(r['year_min'])}-{int(r['year_max'])}"
            )
        lines.append("")

    if not mod_df.empty:
        lines.append("## H4-H5 Snapshot")
        for _, r in mod_df.iterrows():
            lines.append(
                f"- {r['spec']} | {r['hypothesis_id']} ({r['dv']}): "
                f"coef={r['coef']:.4f}, p={r['p_cluster']:.4g}, N={int(r['n_obs'])}, years={int(r['year_min'])}-{int(r['year_max'])}"
            )
        lines.append("")

    (out_dir / "covid_sensitivity_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run COVID-19 sensitivity analyses for confirmatory models.")
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project base directory.",
    )
    parser.add_argument(
        "--include-rents-main",
        type=int,
        default=1,
        choices=[0, 1],
        help="Include natural resource rents in controls (legacy corrected default: 1).",
    )
    parser.add_argument(
        "--out-subdir",
        default="covid_sensitivity",
        help="Output subdirectory under reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_dir = base_dir / "reports" / args.out_subdir
    ensure_dir(out_dir)

    panel = pd.read_csv(base_dir / "data" / "processed" / "regression_panel_2012_2022.csv")
    panel = panel[panel["year"].between(2012, 2022)].copy()
    panel = add_engineered_columns(panel)
    include_rents_main = bool(args.include_rents_main)

    specs = [
        ("baseline_full", None, False),
        ("exclude_2020", {2020}, False),
        ("exclude_2020_2021", {2020, 2021}, False),
        ("covid_interaction_check", None, True),
    ]

    primary_rows: list[dict[str, Any]] = []
    mod_rows: list[dict[str, Any]] = []
    for spec_name, drop_years, covid_interaction in specs:
        primary_rows.extend(
            run_primary_block(
                panel=panel,
                include_rents_main=include_rents_main,
                spec_name=spec_name,
                drop_years=drop_years,
                covid_interaction=covid_interaction,
            )
        )
        mod_rows.extend(
            run_moderation_block(
                panel=panel,
                include_rents_main=include_rents_main,
                spec_name=spec_name,
                drop_years=drop_years,
                covid_interaction=covid_interaction,
            )
        )

    primary_df = pd.DataFrame(primary_rows)
    mod_df = pd.DataFrame(mod_rows)

    primary_df.to_csv(out_dir / "covid_sensitivity_primary_terms.csv", index=False)
    mod_df.to_csv(out_dir / "covid_sensitivity_moderation_terms.csv", index=False)
    write_summary(out_dir, primary_df, mod_df, include_rents_main=include_rents_main)

    meta = {
        "panel_rows": int(len(panel)),
        "panel_countries": int(panel["country_iso3_code"].nunique()),
        "include_rents_main": include_rents_main,
        "specifications": [s[0] for s in specs],
        "primary_rows": int(len(primary_df)),
        "moderation_rows": int(len(mod_df)),
    }
    (out_dir / "covid_sensitivity_metadata.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print("COVID sensitivity analysis completed.")
    print(f"Output directory: {out_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
