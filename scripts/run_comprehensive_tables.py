from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor


WORLD_BANK_API = "https://api.worldbank.org/v2"
OECD_ISO3 = {
    "AUS",
    "AUT",
    "BEL",
    "CAN",
    "CHL",
    "COL",
    "CRI",
    "CZE",
    "DNK",
    "EST",
    "FIN",
    "FRA",
    "DEU",
    "GRC",
    "HUN",
    "ISL",
    "IRL",
    "ISR",
    "ITA",
    "JPN",
    "KOR",
    "LVA",
    "LTU",
    "LUX",
    "MEX",
    "NLD",
    "NZL",
    "NOR",
    "POL",
    "PRT",
    "SVK",
    "SVN",
    "ESP",
    "SWE",
    "CHE",
    "TUR",
    "GBR",
    "USA",
}


def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def zscore(series: pd.Series) -> pd.Series:
    mu = series.mean(skipna=True)
    sigma = series.std(skipna=True, ddof=0)
    if pd.isna(sigma) or sigma == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - mu) / sigma


def significance_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def fmt_coef_se(coef: float, se: float, pval: float) -> str:
    if pd.isna(coef):
        return ""
    return f"{coef:.3f}{significance_stars(pval)} ({se:.3f})"


def fetch_region_map() -> pd.DataFrame:
    page = 1
    rows: list[dict[str, Any]] = []
    while True:
        url = f"{WORLD_BANK_API}/country?format=json&per_page=400&page={page}"
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        obj = r.json()
        meta, data = obj[0], obj[1]
        rows.extend(data)
        if page >= int(meta["pages"]):
            break
        page += 1

    out = []
    for row in rows:
        iso3 = row.get("id")
        if not iso3 or len(str(iso3)) != 3:
            continue
        out.append(
            {
                "country_iso3_code": iso3,
                "wb_region": row.get("region", {}).get("id"),
                "wb_region_name": row.get("region", {}).get("value"),
            }
        )
    return pd.DataFrame(out).drop_duplicates()


def make_model_df(
    panel: pd.DataFrame,
    dv: str,
    include_controls: bool,
    include_intensity: bool,
    include_tariff: bool,
    include_rents: bool,
    post_col: str = "post_paper",
    exposed_col: str = "exposed",
    eci_col: str = "eci",
    include_coi: bool = False,
    include_gpr_mod: bool = False,
    lag_eci: bool = False,
) -> pd.DataFrame:
    df = panel.copy()

    if lag_eci:
        df = df.sort_values(["country_iso3_code", "year"]).copy()
        df["eci_use"] = df.groupby("country_iso3_code")[eci_col].shift(1)
    else:
        df["eci_use"] = df[eci_col]

    df["pe"] = df[post_col] * df[exposed_col]

    req = [dv, "eci_use", "pe", "country_iso3_code", "year"]
    if include_controls:
        req.extend(
            [
                "log_gdp_pc",
                "wdi_trade_openness_pct_gdp",
                "wgi_institutional_quality_composite",
                "gpr_for_model_annual",
            ]
        )
        if include_rents:
            req.append("wdi_natural_resource_rents_pct_gdp")
        if include_tariff:
            req.append("wdi_tariff_applied_weighted_mean_all_products_pct")
    if include_intensity:
        req.append("us_china_trade_intensity_pre")
    if include_coi:
        req.append("coi")
    if include_gpr_mod:
        req.append("gpr_for_model_annual")

    model_df = df.dropna(subset=req).copy()
    if model_df.empty:
        return model_df

    model_df["z_eci"] = zscore(model_df["eci_use"])
    if include_controls:
        model_df["z_log_gdp_pc"] = zscore(model_df["log_gdp_pc"])
        model_df["z_trade_open"] = zscore(model_df["wdi_trade_openness_pct_gdp"])
        model_df["z_wgi"] = zscore(model_df["wgi_institutional_quality_composite"])
        model_df["z_gpr"] = zscore(model_df["gpr_for_model_annual"])
        if include_rents:
            model_df["z_rents"] = zscore(model_df["wdi_natural_resource_rents_pct_gdp"])
        if include_tariff:
            model_df["z_tariff"] = zscore(
                model_df["wdi_tariff_applied_weighted_mean_all_products_pct"]
            )
    if include_intensity:
        model_df["z_intensity"] = zscore(model_df["us_china_trade_intensity_pre"])
    if include_coi:
        model_df["z_coi"] = zscore(model_df["coi"])
    return model_df


def fit_twfe(
    model_df: pd.DataFrame,
    dv: str,
    include_controls: bool,
    include_intensity: bool,
    include_tariff: bool,
    include_rents: bool,
    include_coi: bool = False,
    include_gpr_mod: bool = False,
    cluster_col: str = "country_iso3_code",
    ppml: bool = False,
) -> Any:
    terms = ["z_eci", "pe", "z_eci:pe"]
    if include_controls:
        terms.extend(["z_log_gdp_pc", "z_trade_open", "z_wgi", "z_gpr"])
        if include_rents:
            terms.append("z_rents")
        if include_tariff:
            terms.append("z_tariff")
    if include_intensity:
        terms.append("z_intensity")
    if include_coi:
        terms.extend(["z_coi", "z_eci:pe:z_coi"])
    if include_gpr_mod:
        terms.append("z_eci:pe:z_gpr")

    formula = f"{dv} ~ " + " + ".join(terms) + " + C(country_iso3_code) + C(year)"
    groups = model_df[cluster_col]

    if ppml:
        y = model_df[dv]
        shift = 0.0
        if y.min() <= 0:
            shift = abs(float(y.min())) + 1e-6
        model_df = model_df.copy()
        model_df["dv_ppml"] = y + shift
        formula = formula.replace(dv, "dv_ppml", 1)
        fit = smf.glm(formula=formula, data=model_df, family=sm.families.Poisson()).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
    else:
        fit = smf.ols(formula=formula, data=model_df).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
    return fit


def extract_basic_result(
    fit: Any, model_name: str, dv: str, model_df: pd.DataFrame
) -> dict[str, Any]:
    term = "z_eci:pe"
    coef = float(fit.params.get(term, np.nan))
    se = float(fit.bse.get(term, np.nan))
    pval = float(fit.pvalues.get(term, np.nan))
    return {
        "model": model_name,
        "dv": dv,
        "coef_key": coef,
        "se_key": se,
        "p_key": pval,
        "coef_key_fmt": fmt_coef_se(coef, se, pval),
        "n_obs": int(fit.nobs),
        "n_countries": int(model_df["country_iso3_code"].nunique()),
        "year_min": int(model_df["year"].min()),
        "year_max": int(model_df["year"].max()),
        "r2": float(getattr(fit, "rsquared", np.nan)),
    }


def build_psm_country_set(
    model_df: pd.DataFrame, pre_start: int = 2012, pre_end: int = 2017
) -> set[str]:
    pre = model_df[model_df["year"].between(pre_start, pre_end)].copy()
    covars = [
        "z_eci",
        "z_log_gdp_pc",
        "z_trade_open",
        "z_wgi",
    ]
    if "z_rents" in pre.columns:
        covars.append("z_rents")
    if "z_tariff" in pre.columns:
        covars.append("z_tariff")
    if "z_intensity" in pre.columns:
        covars.append("z_intensity")

    country_pre = (
        pre.groupby("country_iso3_code", as_index=False)[covars + ["exposed"]]
        .mean()
        .dropna()
    )
    if country_pre["exposed"].nunique() < 2:
        return set(country_pre["country_iso3_code"].tolist())

    x = country_pre[covars].values
    y = country_pre["exposed"].astype(int).values
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(x, y)
    country_pre["pscore"] = clf.predict_proba(x)[:, 1]

    treated = country_pre[country_pre["exposed"] == 1].copy()
    control = country_pre[country_pre["exposed"] == 0].copy()
    used_control: set[str] = set()
    matched_controls: set[str] = set()

    for _, t in treated.sort_values("pscore").iterrows():
        cpool = control[~control["country_iso3_code"].isin(used_control)].copy()
        if cpool.empty:
            break
        idx = (cpool["pscore"] - t["pscore"]).abs().idxmin()
        ciso = str(cpool.loc[idx, "country_iso3_code"])
        used_control.add(ciso)
        matched_controls.add(ciso)

    treated_set = set(treated["country_iso3_code"].astype(str).tolist())
    return treated_set.union(matched_controls)


def make_table1(panel: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    vars_map = {
        "dv1_gvc_linkage_change": "delta_tiva_fexgr_dva_share",
        "dv2_export_recovery": "export_recovery_index",
        "dv3_partner_diversification": "partner_diversification_1_minus_hhi",
        "eci": "eci",
        "coi": "coi",
        "post_exposed": "pe",
        "log_gdp_pc": "log_gdp_pc",
        "trade_openness": "wdi_trade_openness_pct_gdp",
        "institutional_quality_wgi": "wgi_institutional_quality_composite",
        "natural_resource_rents": "wdi_natural_resource_rents_pct_gdp",
        "us_china_trade_intensity": "us_china_trade_intensity_pre",
    }
    rows: list[dict[str, Any]] = []
    for label, col in vars_map.items():
        s = pd.to_numeric(panel[col], errors="coerce")
        rows.append(
            {
                "variable": label,
                "n": int(s.notna().sum()),
                "mean": float(s.mean(skipna=True)),
                "sd": float(s.std(skipna=True, ddof=0)),
                "min": float(s.min(skipna=True)),
                "max": float(s.max(skipna=True)),
            }
        )
    t1 = pd.DataFrame(rows)

    # VIF on core controls in complete-case subset.
    vif_cols = [
        "log_gdp_pc",
        "wdi_trade_openness_pct_gdp",
        "wgi_institutional_quality_composite",
        "wdi_natural_resource_rents_pct_gdp",
        "us_china_trade_intensity_pre",
    ]
    vif_df = panel.dropna(subset=vif_cols).copy()
    if len(vif_df) > 0:
        x = vif_df[vif_cols].copy()
        x = (x - x.mean()) / x.std(ddof=0)
        x = sm.add_constant(x)
        vifs = []
        for i, c in enumerate(x.columns):
            if c == "const":
                continue
            vifs.append(
                {"variable": c, "vif": float(variance_inflation_factor(x.values, i))}
            )
        pd.DataFrame(vifs).to_csv(out_dir / "comprehensive_table1_vif.csv", index=False)

    t1.to_csv(out_dir / "comprehensive_table1_descriptive.csv", index=False)
    return t1


def run_main_tables(
    panel: pd.DataFrame, out_dir: Path, include_rents_main: bool = True
) -> pd.DataFrame:
    dvars = {
        "DV1_GVC_Linkage_Change": "delta_tiva_fexgr_dva_share",
        "DV2_Export_Recovery": "export_recovery_index",
        "DV3_Partner_Diversification": "partner_diversification_1_minus_hhi",
    }
    all_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    def collect_detail(
        fit: Any,
        dv_name_in: str,
        model_name_in: str,
        model_df_in: pd.DataFrame,
        terms: list[str],
    ) -> None:
        for t in terms:
            coef = float(fit.params.get(t, np.nan))
            se = float(fit.bse.get(t, np.nan))
            p = float(fit.pvalues.get(t, np.nan))
            detail_rows.append(
                {
                    "dv": dv_name_in,
                    "model": model_name_in,
                    "term": t,
                    "coef": coef,
                    "se": se,
                    "pvalue": p,
                    "coef_fmt": fmt_coef_se(coef, se, p),
                    "n_obs": int(fit.nobs),
                    "n_countries": int(model_df_in["country_iso3_code"].nunique()),
                }
            )

    for dv_name, dv_col in dvars.items():
        # Model 1: Baseline
        m1_df = make_model_df(
            panel,
            dv=dv_col,
            include_controls=False,
            include_intensity=False,
            include_tariff=False,
            include_rents=False,
        )
        if len(m1_df) > 0:
            m1_fit = fit_twfe(
                m1_df,
                dv=dv_col,
                include_controls=False,
                include_intensity=False,
                include_tariff=False,
                include_rents=False,
            )
            all_rows.append(extract_basic_result(m1_fit, "Model1_Baseline", dv_name, m1_df))
            collect_detail(
                m1_fit,
                dv_name,
                "Model1_Baseline",
                m1_df,
                ["z_eci", "pe", "z_eci:pe"],
            )

        # Model 2: + Controls (paper controls except tariff, to maximize country coverage)
        m2_df = make_model_df(
            panel,
            dv=dv_col,
            include_controls=True,
            include_intensity=False,
            include_tariff=False,
            include_rents=include_rents_main,
        )
        if len(m2_df) > 0:
            m2_fit = fit_twfe(
                m2_df,
                dv=dv_col,
                include_controls=True,
                include_intensity=False,
                include_tariff=False,
                include_rents=include_rents_main,
            )
            all_rows.append(extract_basic_result(m2_fit, "Model2_Controls", dv_name, m2_df))
            m2_terms = [
                "z_eci",
                "pe",
                "z_eci:pe",
                "z_log_gdp_pc",
                "z_trade_open",
                "z_wgi",
            ]
            if include_rents_main:
                m2_terms.append("z_rents")
            collect_detail(
                m2_fit,
                dv_name,
                "Model2_Controls",
                m2_df,
                m2_terms,
            )

        # Model 3: Full DiD (+ intensity)
        m3_df = make_model_df(
            panel,
            dv=dv_col,
            include_controls=True,
            include_intensity=True,
            include_tariff=False,
            include_rents=include_rents_main,
        )
        if len(m3_df) > 0:
            m3_fit = fit_twfe(
                m3_df,
                dv=dv_col,
                include_controls=True,
                include_intensity=True,
                include_tariff=False,
                include_rents=include_rents_main,
            )
            all_rows.append(extract_basic_result(m3_fit, "Model3_Full_DiD", dv_name, m3_df))
            m3_terms = [
                "z_eci",
                "pe",
                "z_eci:pe",
                "z_log_gdp_pc",
                "z_trade_open",
                "z_wgi",
            ]
            if include_rents_main:
                m3_terms.append("z_rents")
            m3_terms.append("z_intensity")
            collect_detail(
                m3_fit,
                dv_name,
                "Model3_Full_DiD",
                m3_df,
                m3_terms,
            )

        # Model 4: PSM-DiD (matched countries from Model 3 sample)
        if len(m3_df) > 0:
            matched = build_psm_country_set(m3_df)
            m4_df = m3_df[m3_df["country_iso3_code"].isin(matched)].copy()
            if len(m4_df) > 0:
                m4_fit = fit_twfe(
                    m4_df,
                    dv=dv_col,
                    include_controls=True,
                    include_intensity=True,
                    include_tariff=False,
                    include_rents=include_rents_main,
                )
                all_rows.append(
                    extract_basic_result(m4_fit, "Model4_PSM_DiD", dv_name, m4_df)
                )
                m4_terms = [
                    "z_eci",
                    "pe",
                    "z_eci:pe",
                    "z_log_gdp_pc",
                    "z_trade_open",
                    "z_wgi",
                ]
                if include_rents_main:
                    m4_terms.append("z_rents")
                m4_terms.append("z_intensity")
                collect_detail(
                    m4_fit,
                    dv_name,
                    "Model4_PSM_DiD",
                    m4_df,
                    m4_terms,
                )

    out = pd.DataFrame(all_rows)
    out.to_csv(out_dir / "comprehensive_table2_4_main_models.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(
        out_dir / "comprehensive_table2_4_detailed_terms.csv", index=False
    )
    return out


def run_moderation_table(
    panel: pd.DataFrame, out_dir: Path, include_rents_main: bool = True
) -> pd.DataFrame:
    specs = [
        ("DV1_GVC_Linkage_Change", "delta_tiva_fexgr_dva_share"),
        ("DV2_Export_Recovery", "export_recovery_index"),
        ("DV3_Partner_Diversification", "partner_diversification_1_minus_hhi"),
    ]
    rows: list[dict[str, Any]] = []
    for dv_name, dv_col in specs:
        # COI moderation
        df_coi = make_model_df(
            panel,
            dv=dv_col,
            include_controls=True,
            include_intensity=True,
            include_tariff=False,
            include_rents=include_rents_main,
            include_coi=True,
        )
        if len(df_coi) > 0:
            fit_coi = fit_twfe(
                df_coi,
                dv=dv_col,
                include_controls=True,
                include_intensity=True,
                include_tariff=False,
                include_rents=include_rents_main,
                include_coi=True,
            )
            term = "z_eci:pe:z_coi"
            rows.append(
                {
                    "model": f"{dv_name}_COI_moderation",
                    "dv": dv_name,
                    "term": term,
                    "coef": float(fit_coi.params.get(term, np.nan)),
                    "se": float(fit_coi.bse.get(term, np.nan)),
                    "pvalue": float(fit_coi.pvalues.get(term, np.nan)),
                    "coef_fmt": fmt_coef_se(
                        float(fit_coi.params.get(term, np.nan)),
                        float(fit_coi.bse.get(term, np.nan)),
                        float(fit_coi.pvalues.get(term, np.nan)),
                    ),
                    "n_obs": int(fit_coi.nobs),
                    "n_countries": int(df_coi["country_iso3_code"].nunique()),
                }
            )

        # GPR moderation
        df_gpr = make_model_df(
            panel,
            dv=dv_col,
            include_controls=True,
            include_intensity=True,
            include_tariff=False,
            include_rents=include_rents_main,
            include_gpr_mod=True,
        )
        if len(df_gpr) > 0:
            fit_gpr = fit_twfe(
                df_gpr,
                dv=dv_col,
                include_controls=True,
                include_intensity=True,
                include_tariff=False,
                include_rents=include_rents_main,
                include_gpr_mod=True,
            )
            term = "z_eci:pe:z_gpr"
            rows.append(
                {
                    "model": f"{dv_name}_GPR_moderation",
                    "dv": dv_name,
                    "term": term,
                    "coef": float(fit_gpr.params.get(term, np.nan)),
                    "se": float(fit_gpr.bse.get(term, np.nan)),
                    "pvalue": float(fit_gpr.pvalues.get(term, np.nan)),
                    "coef_fmt": fmt_coef_se(
                        float(fit_gpr.params.get(term, np.nan)),
                        float(fit_gpr.bse.get(term, np.nan)),
                        float(fit_gpr.pvalues.get(term, np.nan)),
                    ),
                    "n_obs": int(fit_gpr.nobs),
                    "n_countries": int(df_gpr["country_iso3_code"].nunique()),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "comprehensive_table5_moderation.csv", index=False)
    return out


def run_robustness_table(
    panel: pd.DataFrame, out_dir: Path, include_rents_main: bool = True
) -> pd.DataFrame:
    dvars = [
        ("DV1", "delta_tiva_fexgr_dva_share"),
        ("DV2", "export_recovery_index"),
        ("DV3", "partner_diversification_1_minus_hhi"),
    ]

    region_map = fetch_region_map()
    panel_r = panel.merge(region_map, on="country_iso3_code", how="left")

    def run_spec(
        spec_name: str,
        panel_in: pd.DataFrame,
        post_col: str = "post_paper",
        exposed_override: pd.Series | None = None,
        lag_eci: bool = False,
        cluster_region: bool = False,
        ppml: bool = False,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {"specification": spec_name}
        n_any = None
        for dv_short, dv_col in dvars:
            p = panel_in.copy()
            if exposed_override is not None:
                p = p.copy()
                p["exposed_alt"] = exposed_override.values
                e_col = "exposed_alt"
            else:
                e_col = "exposed"

            mdf = make_model_df(
                p,
                dv=dv_col,
                include_controls=True,
                include_intensity=True,
                include_tariff=False,
                include_rents=include_rents_main,
                post_col=post_col,
                exposed_col=e_col,
                lag_eci=lag_eci,
            )
            if len(mdf) == 0:
                out[f"{dv_short}_coef"] = np.nan
                continue
            cluster_col = "wb_region" if cluster_region else "country_iso3_code"
            try:
                fit = fit_twfe(
                    mdf,
                    dv=dv_col,
                    include_controls=True,
                    include_intensity=True,
                    include_tariff=False,
                    include_rents=include_rents_main,
                    cluster_col=cluster_col,
                    ppml=ppml,
                )
                coef = float(fit.params.get("z_eci:pe", np.nan))
                out[f"{dv_short}_coef"] = coef
                out[f"{dv_short}_coef_fmt"] = (
                    f"{coef:.3f}{significance_stars(float(fit.pvalues.get('z_eci:pe', np.nan)))}"
                )
                if n_any is None:
                    n_any = int(fit.nobs)
            except Exception:
                out[f"{dv_short}_coef"] = np.nan
                out[f"{dv_short}_coef_fmt"] = ""
        out["N"] = n_any
        return out

    rows: list[dict[str, Any]] = []
    rows.append(run_spec("Main specification (Model3 equivalent)", panel_r))

    # Placebo post period 2014-2015
    placebo = panel_r.copy()
    placebo["post_placebo"] = placebo["year"].between(2014, 2015).astype(int)
    rows.append(run_spec("Placebo: Pre-period shock (2014-2015)", placebo, post_col="post_placebo"))

    # Alternative threshold: top quartile of intensity
    p2 = panel_r.copy()
    intensity = p2[["country_iso3_code", "us_china_trade_intensity_pre"]].drop_duplicates()
    q3 = intensity["us_china_trade_intensity_pre"].quantile(0.75)
    p2["exposed_q4"] = (p2["us_china_trade_intensity_pre"] >= q3).astype(int)
    rows.append(
        run_spec(
            "Alternative exposure threshold (top quartile)",
            p2,
            exposed_override=p2["exposed_q4"],
        )
    )

    # Exclude resource-rich
    rows.append(
        run_spec(
            "Exclude resource-rich countries (>20% rents)",
            panel_r[panel_r["wdi_natural_resource_rents_pct_gdp"] <= 20].copy(),
        )
    )

    # Exclude OECD
    rows.append(
        run_spec(
            "Exclude OECD countries",
            panel_r[~panel_r["country_iso3_code"].isin(OECD_ISO3)].copy(),
        )
    )

    # Lagged ECI
    rows.append(run_spec("Lagged ECI (t-1)", panel_r, lag_eci=True))

    # Cluster by region
    rows.append(run_spec("Cluster SE by region", panel_r, cluster_region=True))

    # PPML analogue
    rows.append(run_spec("Poisson pseudo-max. likelihood (PPML)", panel_r, ppml=True))

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "comprehensive_table6_robustness.csv", index=False)
    return out


def write_summary(
    out_dir: Path,
    table1: pd.DataFrame,
    table24: pd.DataFrame,
    table5: pd.DataFrame,
    table6: pd.DataFrame,
    panel: pd.DataFrame,
    include_rents_main: bool = True,
) -> None:
    lines = [
        "# Comprehensive Country-Level Analysis (Expanded Panel)",
        "",
        "Sample base:",
        f"- Rows: `{len(panel)}`",
        f"- Countries: `{panel['country_iso3_code'].nunique()}`",
        f"- Years: `{int(panel['year'].min())}-{int(panel['year'].max())}`",
        "",
        "Generated tables:",
        "- `comprehensive_table1_descriptive.csv`",
        "- `comprehensive_table1_vif.csv`",
        "- `comprehensive_table2_4_main_models.csv`",
        "- `comprehensive_table5_moderation.csv`",
        "- `comprehensive_table6_robustness.csv`",
        "",
        "Notes:",
        "- Post period follows the paper note in extracted table text: 2019-2022.",
        "- FE specifications use country and year fixed effects with clustered SEs.",
        f"- Main controls include natural resource rents: `{include_rents_main}`.",
        "- Full-controls models use broad controls plus US-China intensity; tariff control excluded in the main comprehensive tables to preserve wider country coverage. Tariff appears in the dedicated regression panel and can be reintroduced for stricter replication.",
    ]
    (out_dir / "comprehensive_analysis_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    meta = {
        "table2_4_rows": int(len(table24)),
        "table5_rows": int(len(table5)),
        "table6_rows": int(len(table6)),
        "panel_rows": int(len(panel)),
        "panel_countries": int(panel["country_iso3_code"].nunique()),
        "panel_year_min": int(panel["year"].min()),
        "panel_year_max": int(panel["year"].max()),
        "include_rents_main": bool(include_rents_main),
    }
    (out_dir / "comprehensive_analysis_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run comprehensive expanded-country analysis tables.")
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project base directory",
    )
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument(
        "--include-rents-main",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to include natural resource rents in main-control models (1=yes, 0=no).",
    )
    parser.add_argument(
        "--reports-subdir",
        default="",
        help="Optional subdirectory under reports for output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    processed_dir = base_dir / "data" / "processed"
    reports_root = base_dir / "reports"
    reports_dir = (
        reports_root / args.reports_subdir
        if str(args.reports_subdir).strip()
        else reports_root
    )
    ensure_dirs(reports_dir)

    panel_path = processed_dir / "regression_panel_2012_2022.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing panel file: {panel_path}")

    panel = pd.read_csv(panel_path)
    panel = panel[panel["year"].between(args.start_year, args.end_year)].copy()
    panel["post_paper"] = panel["year"].between(2019, 2022).astype(int)
    panel["pe"] = panel["post_paper"] * panel["exposed"]

    table1 = make_table1(panel, reports_dir)
    include_rents_main = bool(args.include_rents_main)

    table24 = run_main_tables(panel, reports_dir, include_rents_main=include_rents_main)
    table5 = run_moderation_table(panel, reports_dir, include_rents_main=include_rents_main)
    table6 = run_robustness_table(panel, reports_dir, include_rents_main=include_rents_main)
    write_summary(
        reports_dir,
        table1,
        table24,
        table5,
        table6,
        panel,
        include_rents_main=include_rents_main,
    )

    print("Comprehensive analysis completed.")
    print(f"Table 1 rows: {len(table1)}")
    print(f"Table 2-4 rows: {len(table24)}")
    print(f"Table 5 rows: {len(table5)}")
    print(f"Table 6 rows: {len(table6)}")
    print(f"include_rents_main={include_rents_main}")


if __name__ == "__main__":
    main()
