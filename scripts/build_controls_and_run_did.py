from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf


WORLD_BANK_API = "https://api.worldbank.org/v2"
GPR_DTA_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.dta"


WDI_INDICATORS = {
    "NY.GDP.PCAP.KD": "wdi_gdp_pc_const_2015_usd",
    "NE.TRD.GNFS.ZS": "wdi_trade_openness_pct_gdp",
    "NY.GDP.TOTL.RT.ZS": "wdi_natural_resource_rents_pct_gdp",
    "TM.TAX.MRCH.WM.AR.ZS": "wdi_tariff_applied_weighted_mean_all_products_pct",
}

WGI_INDICATORS = {
    "GOV_WGI_CC.EST": "wgi_cc_est",
    "GOV_WGI_GE.EST": "wgi_ge_est",
    "GOV_WGI_PV.EST": "wgi_pv_est",
    "GOV_WGI_RL.EST": "wgi_rl_est",
    "GOV_WGI_RQ.EST": "wgi_rq_est",
    "GOV_WGI_VA.EST": "wgi_va_est",
}


def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def wb_fetch_indicator(indicator_code: str) -> pd.DataFrame:
    page = 1
    rows: list[dict] = []
    while True:
        url = (
            f"{WORLD_BANK_API}/country/all/indicator/{indicator_code}"
            f"?format=json&per_page=20000&page={page}"
        )
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        obj = response.json()
        if not isinstance(obj, list) or len(obj) < 2:
            raise RuntimeError(f"Unexpected World Bank response for {indicator_code}: {obj}")
        meta, data = obj[0], obj[1]
        rows.extend(data)
        if page >= int(meta["pages"]):
            break
        page += 1

    out = []
    for row in rows:
        iso3 = row.get("countryiso3code")
        date = row.get("date")
        value = row.get("value")
        if not iso3 or len(str(iso3)) != 3 or value is None:
            continue
        try:
            year = int(date)
        except (TypeError, ValueError):
            continue
        out.append(
            {
                "country_iso3_code": iso3,
                "year": year,
                "value": float(value),
            }
        )
    return pd.DataFrame(out)


def fetch_wdi_controls(start_year: int, end_year: int) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for indicator_code, col_name in WDI_INDICATORS.items():
        df = wb_fetch_indicator(indicator_code).rename(columns={"value": col_name})
        df = df[df["year"].between(start_year, end_year)].copy()
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["country_iso3_code", "year"], how="outer")

    assert merged is not None
    return merged


def fetch_wgi_controls(start_year: int, end_year: int) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for indicator_code, col_name in WGI_INDICATORS.items():
        df = wb_fetch_indicator(indicator_code).rename(columns={"value": col_name})
        df = df[df["year"].between(start_year, end_year)].copy()
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["country_iso3_code", "year"], how="outer")
    assert merged is not None
    wgi_cols = list(WGI_INDICATORS.values())
    merged["wgi_institutional_quality_composite"] = merged[wgi_cols].mean(axis=1)
    return merged


def fetch_gpr_annual_series(
    raw_dir: Path, start_year: int, end_year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs(raw_dir)
    out_path = raw_dir / "gpr_data_gpr_export.dta"
    if not out_path.exists():
        response = requests.get(GPR_DTA_URL, timeout=300)
        response.raise_for_status()
        out_path.write_bytes(response.content)

    df = pd.read_stata(out_path)
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df[df["month"].notna()].copy()
    df["year"] = df["month"].dt.year.astype(int)
    df["GPR"] = pd.to_numeric(df["GPR"], errors="coerce")
    df = df[df["GPR"].notna()].copy()
    gpr_annual_global = (
        df.groupby("year", as_index=False)["GPR"]
        .mean()
        .rename(columns={"GPR": "gpr_global_annual"})
    )
    gpr_annual_global = gpr_annual_global[
        gpr_annual_global["year"].between(start_year, end_year)
    ].copy()

    gpr_country_cols = [c for c in df.columns if c.startswith("GPRC_")]
    long = df[["year"] + gpr_country_cols].melt(
        id_vars=["year"],
        value_vars=gpr_country_cols,
        var_name="gpr_country_col",
        value_name="gpr_country_monthly",
    )
    long["country_iso3_code"] = long["gpr_country_col"].str.replace(
        "GPRC_", "", regex=False
    )
    long["gpr_country_monthly"] = pd.to_numeric(
        long["gpr_country_monthly"], errors="coerce"
    )
    long = long[long["gpr_country_monthly"].notna()].copy()
    gpr_annual_country = (
        long.groupby(["country_iso3_code", "year"], as_index=False)["gpr_country_monthly"]
        .mean()
        .rename(columns={"gpr_country_monthly": "gpr_country_annual"})
    )
    gpr_annual_country = gpr_annual_country[
        gpr_annual_country["year"].between(start_year, end_year)
    ].copy()
    return gpr_annual_global, gpr_annual_country


def build_us_china_trade_intensity(
    bilateral_df: pd.DataFrame, pre_start: int = 2015, pre_end: int = 2017
) -> pd.DataFrame:
    pre = bilateral_df[bilateral_df["year"].between(pre_start, pre_end)].copy()
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
    merged["us_china_intensity_year"] = merged["us_china_trade"] / merged["total_trade"]

    intensity = (
        merged.groupby("country_iso3_code", as_index=False)["us_china_intensity_year"]
        .mean()
        .rename(columns={"us_china_intensity_year": "us_china_trade_intensity_pre"})
    )

    valid = intensity["us_china_trade_intensity_pre"].dropna()
    q1, q2 = valid.quantile([1 / 3, 2 / 3]).tolist()
    intensity["exposed"] = (
        intensity["us_china_trade_intensity_pre"] >= q2
    ).astype(int)
    intensity["tertile_threshold_low"] = q1
    intensity["tertile_threshold_high"] = q2
    return intensity


def zscore_within_sample(series: pd.Series) -> pd.Series:
    mu = series.mean(skipna=True)
    sigma = series.std(skipna=True, ddof=0)
    if sigma == 0 or pd.isna(sigma):
        return pd.Series(np.nan, index=series.index)
    return (series - mu) / sigma


def run_twfe_model(
    df: pd.DataFrame,
    dv: str,
    model_id: str,
) -> dict:
    model_df = df.copy()
    required = [
        dv,
        "eci",
        "post",
        "exposed",
        "log_gdp_pc",
        "wdi_trade_openness_pct_gdp",
        "wgi_institutional_quality_composite",
        "wdi_natural_resource_rents_pct_gdp",
        "wdi_tariff_applied_weighted_mean_all_products_pct",
        "gpr_for_model_annual",
        "country_iso3_code",
        "year",
    ]
    model_df = model_df.dropna(subset=required).copy()

    for col in [
        "eci",
        "log_gdp_pc",
        "wdi_trade_openness_pct_gdp",
        "wgi_institutional_quality_composite",
        "wdi_natural_resource_rents_pct_gdp",
        "wdi_tariff_applied_weighted_mean_all_products_pct",
        "gpr_for_model_annual",
    ]:
        model_df[f"z_{col}"] = zscore_within_sample(model_df[col])

    formula = (
        f"{dv} ~ post * exposed * z_eci"
        " + z_log_gdp_pc"
        " + z_wdi_trade_openness_pct_gdp"
        " + z_wgi_institutional_quality_composite"
        " + z_wdi_natural_resource_rents_pct_gdp"
        " + z_wdi_tariff_applied_weighted_mean_all_products_pct"
        " + z_gpr_for_model_annual"
        " + C(country_iso3_code) + C(year)"
    )

    fit = smf.ols(formula=formula, data=model_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": model_df["country_iso3_code"]},
    )

    term = "post:exposed:z_eci"
    coef = float(fit.params.get(term, np.nan))
    se = float(fit.bse.get(term, np.nan))
    pval = float(fit.pvalues.get(term, np.nan))

    return {
        "model_id": model_id,
        "dv": dv,
        "n_obs": int(fit.nobs),
        "n_countries": int(model_df["country_iso3_code"].nunique()),
        "years_min": int(model_df["year"].min()),
        "years_max": int(model_df["year"].max()),
        "coef_post_exposed_eci": coef,
        "se_post_exposed_eci": se,
        "pvalue_post_exposed_eci": pval,
        "r2": float(fit.rsquared),
        "adj_r2": float(fit.rsquared_adj),
        "summary_text": fit.summary().as_text(),
    }


def save_model_outputs(results: list[dict], reports_dir: Path) -> None:
    ensure_dirs(reports_dir)
    rows = []
    for r in results:
        rows.append(
            {
                "model_id": r["model_id"],
                "dv": r["dv"],
                "n_obs": r["n_obs"],
                "n_countries": r["n_countries"],
                "years_min": r["years_min"],
                "years_max": r["years_max"],
                "coef_post_exposed_eci": r["coef_post_exposed_eci"],
                "se_post_exposed_eci": r["se_post_exposed_eci"],
                "pvalue_post_exposed_eci": r["pvalue_post_exposed_eci"],
                "r2": r["r2"],
                "adj_r2": r["adj_r2"],
            }
        )
    pd.DataFrame(rows).to_csv(reports_dir / "did_model_results.csv", index=False)

    with open(reports_dir / "did_model_summaries.txt", "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(f"\n\n===== {r['model_id']} ({r['dv']}) =====\n\n")
            fh.write(r["summary_text"])
            fh.write("\n")

    md_lines = [
        "# Paper-Style DiD Model Results",
        "",
        "Coefficient reported below is for `post * exposed * z_eci`.",
        "",
    ]
    for r in rows:
        md_lines.extend(
            [
                f"## {r['model_id']} ({r['dv']})",
                f"- N: `{r['n_obs']}`",
                f"- Countries: `{r['n_countries']}`",
                f"- Years: `{r['years_min']}-{r['years_max']}`",
                f"- Coef (`post:exposed:z_eci`): `{r['coef_post_exposed_eci']:.4f}`",
                f"- SE (cluster-country): `{r['se_post_exposed_eci']:.4f}`",
                f"- p-value: `{r['pvalue_post_exposed_eci']:.4g}`",
                f"- R2: `{r['r2']:.4f}`",
                "",
            ]
        )
    (reports_dir / "did_model_results.md").write_text("\n".join(md_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add WDI/WGI/GPR/tariff controls and run paper-style DiD models."
    )
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project base directory.",
    )
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int, default=2022)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    reports_dir = base_dir / "reports"
    ensure_dirs(raw_dir, processed_dir, reports_dir)

    panel_path = processed_dir / "databank_country_year_2012_2024.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing base panel: {panel_path}")
    panel = pd.read_csv(panel_path)
    panel = panel[panel["year"].between(args.start_year, args.end_year)].copy()

    bilateral_candidates = [
        processed_dir / f"source_atlas_country_country_year_{args.start_year}_{args.end_year}.csv",
        processed_dir / "source_atlas_country_country_year_2012_2024.csv",
    ]
    bilateral_path = next((p for p in bilateral_candidates if p.exists()), None)
    if bilateral_path is None:
        raise FileNotFoundError("Could not find bilateral source file in processed directory.")
    bilateral = pd.read_csv(bilateral_path)
    bilateral = bilateral[bilateral["year"].between(args.start_year, args.end_year)].copy()

    print("Fetching WDI controls...")
    wdi = fetch_wdi_controls(args.start_year, args.end_year)
    print("Fetching WGI controls...")
    wgi = fetch_wgi_controls(args.start_year, args.end_year)
    print("Fetching GPR annual...")
    gpr_global, gpr_country = fetch_gpr_annual_series(
        raw_dir=raw_dir, start_year=args.start_year, end_year=args.end_year
    )
    print("Computing US-China trade intensity...")
    intensity = build_us_china_trade_intensity(bilateral)

    controls = wdi.merge(wgi, on=["country_iso3_code", "year"], how="outer")
    panel = panel.merge(controls, on=["country_iso3_code", "year"], how="left")
    panel = panel.merge(intensity, on="country_iso3_code", how="left")
    panel = panel.merge(gpr_global, on="year", how="left")
    panel = panel.merge(gpr_country, on=["country_iso3_code", "year"], how="left")
    panel["gpr_for_model_annual"] = panel["gpr_country_annual"].fillna(
        panel["gpr_global_annual"]
    )

    panel["log_gdp_pc"] = np.where(
        panel["wdi_gdp_pc_const_2015_usd"] > 0,
        np.log(panel["wdi_gdp_pc_const_2015_usd"]),
        np.nan,
    )
    panel["post"] = (panel["year"] >= 2018).astype(int)
    panel["exposed"] = panel["exposed"].fillna(0).astype(int)

    out_panel = processed_dir / f"regression_panel_{args.start_year}_{args.end_year}.csv"
    panel.to_csv(out_panel, index=False)

    # Feasible sample for full paper-style model with TiVA forward-linkage DV.
    core = panel.copy()
    core = core.dropna(subset=["tiva_fexgr_dva_share"]).copy()
    keep_countries = (
        core.groupby("country_iso3_code")["year"].nunique().reset_index(name="n_years")
    )
    keep_countries = keep_countries[
        keep_countries["n_years"] == (args.end_year - args.start_year + 1)
    ]["country_iso3_code"]
    core = core[core["country_iso3_code"].isin(set(keep_countries))].copy()

    model_inputs = core.copy()
    model_inputs["dv_gvc_linkage_change"] = model_inputs["delta_tiva_fexgr_dva_share"]
    model_inputs["dv_export_recovery"] = model_inputs["export_recovery_index"]
    model_inputs["dv_partner_diversification"] = model_inputs[
        "partner_diversification_1_minus_hhi"
    ]

    results = []
    results.append(
        run_twfe_model(
            model_inputs,
            dv="dv_gvc_linkage_change",
            model_id="M1_GVC_Linkage_Change",
        )
    )
    results.append(
        run_twfe_model(
            model_inputs,
            dv="dv_export_recovery",
            model_id="M2_Export_Recovery",
        )
    )
    results.append(
        run_twfe_model(
            model_inputs,
            dv="dv_partner_diversification",
            model_id="M3_Partner_Diversification",
        )
    )
    save_model_outputs(results=results, reports_dir=reports_dir)

    control_missing = {
        c: int(panel[c].isna().sum())
        for c in [
            "wdi_gdp_pc_const_2015_usd",
            "wdi_trade_openness_pct_gdp",
            "wdi_natural_resource_rents_pct_gdp",
            "wdi_tariff_applied_weighted_mean_all_products_pct",
            "wgi_institutional_quality_composite",
            "gpr_global_annual",
            "gpr_country_annual",
            "gpr_for_model_annual",
        ]
    }

    metadata = {
        "panel_file": str(out_panel),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "n_rows_panel": int(len(panel)),
        "n_rows_model_input": int(len(model_inputs)),
        "n_countries_model_input": int(model_inputs["country_iso3_code"].nunique()),
        "control_missing_counts": control_missing,
        "control_sources": {
            "wdi_indicators": WDI_INDICATORS,
            "wgi_indicators": WGI_INDICATORS,
            "gpr_source": GPR_DTA_URL,
            "tariff_indicator_source_note": (
                "WDI indicator TM.TAX.MRCH.WM.AR.ZS (WITS/TRAINS/WTO-based)."
            ),
        },
    }
    (reports_dir / "regression_build_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Completed.")
    print(f"Panel rows: {len(panel)}")
    print(f"Model input rows: {len(model_inputs)}")
    print(f"Model countries: {model_inputs['country_iso3_code'].nunique()}")
    for r in results:
        print(
            f"{r['model_id']}: coef={r['coef_post_exposed_eci']:.4f}, "
            f"p={r['pvalue_post_exposed_eci']:.4g}, N={r['n_obs']}"
        )


if __name__ == "__main__":
    main()
