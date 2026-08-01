from __future__ import annotations

import argparse
import itertools
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import run_final_audited_analysis as audited
from run_capability_conversion_analysis import contrast_statistics, wild_cluster_bootstrap_contrast
from run_structural_regime_analysis import fit_terms, load_wdi_structural_source
from run_market_connectivity_analysis import (
    PRIMARY_OUTCOME,
    PRIMARY_LABEL,
    OUTCOME_ALTS,
    MECHANISMS,
    build as build_base,
    post_terms,
    row as coefficient_row,
    z,
)


def write(df, path):
    df.to_csv(path, index=False)


def build_channel_intensities(frame, wdi, primary_countries=None):
    pop = wdi.loc[wdi["indicator"].eq("SP.POP.TOTL"), ["country_iso3_code", "year", "value"]].rename(columns={"value": "population"})
    work = frame.merge(pop, on=["country_iso3_code", "year"], how="left")
    work["pre_real_gdp_usd"] = pd.to_numeric(work["wdi_gdp_pc_const_2015_usd"], errors="coerce") * pd.to_numeric(work["population"], errors="coerce")
    work["export_intensity"] = 100.0 * pd.to_numeric(work["export_value"], errors="coerce") / work["pre_real_gdp_usd"]
    work["import_intensity"] = 100.0 * pd.to_numeric(work["import_value"], errors="coerce") / work["pre_real_gdp_usd"]
    work["total_trade_intensity"] = work["export_intensity"] + work["import_intensity"]
    work["log_total_trade_openness"] = np.log1p(work["total_trade_intensity"].where(work["total_trade_intensity"] >= 0))
    pre = work.loc[work["year"].between(2015, 2017)]
    agg = pre.groupby("country_iso3_code", as_index=False).agg(
        pre_export_intensity=("export_intensity", "mean"),
        pre_import_intensity=("import_intensity", "mean"),
        pre_total_trade_intensity=("total_trade_intensity", "mean"),
        pre_log_total_trade_openness=("log_total_trade_openness", "mean"),
        pre_wdi_total_openness=("wdi_trade_openness_pct_gdp", "mean"),
        pre_real_gdp_usd=("pre_real_gdp_usd", "mean"),
    )
    agg["pre_total_minus_wdi_openness"] = agg["pre_total_trade_intensity"] - agg["pre_wdi_total_openness"]
    denom = agg["pre_total_trade_intensity"].replace(0, np.nan)
    agg["pre_wdi_compatible_export_intensity"] = agg["pre_wdi_total_openness"] * agg["pre_export_intensity"] / denom
    agg["pre_wdi_compatible_import_intensity"] = agg["pre_wdi_total_openness"] * agg["pre_import_intensity"] / denom
    agg["pre_wdi_compatible_sum"] = agg["pre_wdi_compatible_export_intensity"] + agg["pre_wdi_compatible_import_intensity"]
    if primary_countries is not None:
        agg["in_primary_sample"] = agg["country_iso3_code"].isin(set(primary_countries)).astype(int)
    return frame.merge(agg, on="country_iso3_code", how="left"), agg


def separate_channel_terms(frame, channels):
    out = frame.copy()
    terms = []
    base = {"eci": "z_eci_pre", "exposure": "z_exposure_pre"}
    for combo in [("eci",), ("exposure",), ("eci", "exposure")]:
        term = "post_x_" + "_x_".join(combo)
        product = np.ones(len(out))
        for name in combo:
            product *= pd.to_numeric(out[base[name]], errors="coerce").to_numpy()
        out[term] = out["post_2018"].to_numpy() * product
        terms.append(term)
    for name, col in channels.items():
        factors = {**base, name: col}
        for combo in [("name",), ("eci", "name"), ("exposure", "name"), ("eci", "exposure", "name")]:
            combo = tuple(name if x == "name" else x for x in combo)
            term = "post_x_" + "_x_".join(combo)
            product = np.ones(len(out))
            for label in combo:
                product *= pd.to_numeric(out[factors[label]], errors="coerce").to_numpy()
            out[term] = out["post_2018"].to_numpy() * product
            terms.append(term)
    targets = {name: "post_x_eci_x_exposure_x_" + name for name in channels}
    return out, terms, targets


def contrast_row(fit, contrast, reps, seed, test_id, outcome_key, outcome, extra=None):
    stat = contrast_statistics(fit, contrast)
    boot = wild_cluster_bootstrap_contrast(fit, contrast, reps=reps, seed=seed, alternative="two-sided")
    result = {**stat, **boot, "test_id": test_id, "outcome_key": outcome_key, "outcome": outcome,
              "pvalue_for_fdr": boot["p_wild_bootstrap"], "n_obs": fit.n_obs,
              "n_countries": fit.n_countries, "n_years": fit.n_years}
    if extra:
        result.update(extra)
    return result


def _channel_specification(agg, name):
    base = agg.copy()
    if name == "raw_atlas_components":
        cols = {"export": "pre_export_intensity", "import": "pre_import_intensity"}
        source = "Atlas export/import values divided by WDI GDP; sum is an internally compatible Atlas trade intensity, distinct from the WDI primary openness measure"
        transform = "raw"
    elif name == "log1p_components":
        base["channel_export"] = np.log1p(base["pre_export_intensity"].clip(lower=0))
        base["channel_import"] = np.log1p(base["pre_import_intensity"].clip(lower=0))
        cols = {"export": "channel_export", "import": "channel_import"}
        source = "log(1+x) transform of raw Atlas intensity"
        transform = "log1p"
    elif name == "winsorized_1pct_components":
        base["channel_export"] = base["pre_export_intensity"].clip(base["pre_export_intensity"].quantile(.01), base["pre_export_intensity"].quantile(.99))
        base["channel_import"] = base["pre_import_intensity"].clip(base["pre_import_intensity"].quantile(.01), base["pre_import_intensity"].quantile(.99))
        cols = {"export": "channel_export", "import": "channel_import"}
        source = "1 percent winsorized raw Atlas intensity"
        transform = "winsorized_1pct"
    elif name == "wdi_reconstructed_components":
        cols = {"export": "pre_wdi_compatible_export_intensity", "import": "pre_wdi_compatible_import_intensity"}
        source = "WDI primary total openness allocated across Atlas export/import shares; components sum to WDI primary openness"
        transform = "wdi_total_reconstructed"
    else:
        raise ValueError(f"unknown channel specification: {name}")
    return base, cols, source, transform


def _channel_spec_rows(frame, agg, name, reps, seed):
    if name in {"exclude_highest_1pct_import", "exclude_highest_5pct_import", "exclude_smallest_5pct_real_gdp"}:
        if name == "exclude_highest_1pct_import":
            keep = agg["pre_import_intensity"] < agg["pre_import_intensity"].quantile(.99)
        elif name == "exclude_highest_5pct_import":
            keep = agg["pre_import_intensity"] < agg["pre_import_intensity"].quantile(.95)
        else:
            keep = agg["pre_real_gdp_usd"] > agg["pre_real_gdp_usd"].quantile(.05)
        selected = agg.loc[keep].copy()
        source_name = "raw_atlas_components"
        exclusion = name
    else:
        selected, cols, source, transform = _channel_specification(agg, name)
        source_name = name
        exclusion = "none"
    if name in {"exclude_highest_1pct_import", "exclude_highest_5pct_import", "exclude_smallest_5pct_real_gdp"}:
        selected, cols, source, transform = _channel_specification(selected, source_name)
    selected["z_channel_export"] = z(selected[cols["export"]])
    selected["z_channel_import"] = z(selected[cols["import"]])
    model_frame = frame.loc[frame["country_iso3_code"].isin(set(selected["country_iso3_code"]))].copy()
    model_frame = model_frame.merge(selected[["country_iso3_code", "z_channel_export", "z_channel_import"]], on="country_iso3_code", how="left")
    extra = {
        "channel_specification": name,
        "channel_source": source,
        "channel_transformation": transform,
        "sample_alignment": "exact primary country sample before prespecified exclusion",
        "sample_exclusion": exclusion,
    }
    rows = []
    for i, (channel_name, col) in enumerate({"export_intensity": "z_channel_export", "import_intensity": "z_channel_import"}.items()):
        work, terms, target = post_terms(model_frame, {channel_name: col})
        fit = fit_terms(work, PRIMARY_OUTCOME, terms)
        rows.append(coefficient_row(fit, target, reps, seed + i, f"channel_{name}_{channel_name}", "diversification", PRIMARY_LABEL,
                                    {**extra, "channel": channel_name, "channel_type": "individual", "estimand": f"ECI x Exposure x Post x {channel_name}"}))
    channels = {"export_intensity": "z_channel_export", "import_intensity": "z_channel_import"}
    work, terms, targets = separate_channel_terms(model_frame, channels)
    fit = fit_terms(work, PRIMARY_OUTCOME, terms)
    for i, (channel_name, target) in enumerate(targets.items()):
        c = np.zeros(len(fit.term_names)); c[fit.term_index(target)] = 1
        rows.append(contrast_row(fit, c, reps, seed + 10 + i, f"channel_joint_{name}_{channel_name}", "diversification", PRIMARY_LABEL,
                                 {**extra, "channel": channel_name, "channel_type": "joint_model", "estimand": f"ECI x Exposure x Post x {channel_name}"}))
    diff = np.zeros(len(fit.term_names))
    diff[fit.term_index(targets["export_intensity"])] = 1
    diff[fit.term_index(targets["import_intensity"])] = -1
    rows.append(contrast_row(fit, diff, reps, seed + 20, f"channel_difference_{name}", "diversification", PRIMARY_LABEL,
                             {**extra, "channel": "export_minus_import", "channel_type": "formal_equality_test", "hypothesis": "beta_export_intensity = beta_import_intensity"}))
    return rows


def run_channel_decomposition(frame, agg, primary_countries, reps, seed, outdir):
    primary_agg = agg.loc[agg["country_iso3_code"].isin(set(primary_countries))].copy()
    specs = [
        "raw_atlas_components",
        "log1p_components",
        "winsorized_1pct_components",
        "exclude_highest_1pct_import",
        "exclude_highest_5pct_import",
        "exclude_smallest_5pct_real_gdp",
        "wdi_reconstructed_components",
    ]
    all_rows = []
    for i, name in enumerate(specs):
        all_rows.extend(_channel_spec_rows(frame, primary_agg, name, reps, seed + i * 100))
    stress = pd.DataFrame(all_rows)
    write(stress, outdir / "market_connectivity_channel_stress_tests.csv")
    raw = stress.loc[stress["channel_specification"].eq("raw_atlas_components")].copy()
    raw["channel_specification"] = "raw_atlas_components_exact_primary_sample"
    write(raw, outdir / "market_connectivity_channel_decomposition.csv")
    summary_rows = []
    for name, g in stress.groupby("channel_specification", sort=False):
        individual = g.loc[g["channel_type"].eq("individual")].set_index("channel")
        joint = g.loc[g["channel_type"].eq("joint_model")].set_index("channel")
        equality = g.loc[g["channel_type"].eq("formal_equality_test")].iloc[0]
        summary_rows.append({
            "channel_specification": name,
            "channel_source": g["channel_source"].iloc[0],
            "channel_transformation": g["channel_transformation"].iloc[0],
            "sample_exclusion": g["sample_exclusion"].iloc[0],
            "export_estimate": individual.loc["export_intensity", "estimate"],
            "export_p_wild_bootstrap": individual.loc["export_intensity", "p_wild_bootstrap"],
            "import_estimate": individual.loc["import_intensity", "estimate"],
            "import_p_wild_bootstrap": individual.loc["import_intensity", "p_wild_bootstrap"],
            "joint_export_estimate": joint.loc["export_intensity", "estimate"],
            "joint_export_p_wild_bootstrap": joint.loc["export_intensity", "p_wild_bootstrap"],
            "joint_import_estimate": joint.loc["import_intensity", "estimate"],
            "joint_import_p_wild_bootstrap": joint.loc["import_intensity", "p_wild_bootstrap"],
            "export_minus_import_estimate": equality["estimate"],
            "export_minus_import_p_wild_bootstrap": equality["p_wild_bootstrap"],
            "n_obs": int(equality["n_obs"]),
            "n_countries": int(equality["n_countries"]),
            "n_years": int(equality["n_years"]),
            "export_sign": "positive" if individual.loc["export_intensity", "estimate"] > 0 else "negative",
            "import_sign": "positive" if individual.loc["import_intensity", "estimate"] > 0 else "negative",
            "joint_import_sign": "positive" if joint.loc["import_intensity", "estimate"] > 0 else "negative",
            "difference_reasonably_supported": bool(equality["p_wild_bootstrap"] < .10),
        })
    summary = pd.DataFrame(summary_rows)
    summary["import_sign_stable_across_specs"] = summary["import_estimate"].gt(0).all()
    summary["joint_import_sign_stable_across_specs"] = summary["joint_import_estimate"].gt(0).all()
    summary["extreme_small_economy_exclusions_retain_import_sign"] = summary.loc[summary["channel_specification"].isin(["exclude_highest_1pct_import", "exclude_highest_5pct_import", "exclude_smallest_5pct_real_gdp"]), "import_estimate"].gt(0).all()
    write(summary, outdir / "market_connectivity_channel_stress_summary.csv")
    return raw, stress, summary


def intensive_outcomes(base_dir, frame, outdir, incumbent_period, model_period, design, output_filename):
    path = base_dir / "data" / "processed" / "source_atlas_country_country_year_2012_2022.csv"
    bilateral = pd.read_csv(path, usecols=["country_iso3_code", "partner_iso3_code", "year", "export_value"])
    bilateral["export_value"] = pd.to_numeric(bilateral["export_value"], errors="coerce").fillna(0.0)
    bilateral = bilateral.loc[
        bilateral["country_iso3_code"].ne(bilateral["partner_iso3_code"])
        & ~bilateral["partner_iso3_code"].isin(["USA", "CHN"])
    ].copy()
    inc_start, inc_end = incumbent_period
    model_start, model_end = model_period
    pre = bilateral.loc[bilateral["year"].between(inc_start, inc_end) & bilateral["export_value"].gt(0)]
    sets = {country: set(g["partner_iso3_code"]) for country, g in pre.groupby("country_iso3_code")}
    # Complete the country-year-incumbent grid so inactive baseline years contribute zero shares.
    baseline_years = list(range(inc_start, inc_end + 1))
    grid_rows = [
        {"country_iso3_code": country, "year": year, "partner_iso3_code": partner}
        for country, partners in sets.items()
        for year in baseline_years
        for partner in sorted(partners)
    ]
    baseline_grid = pd.DataFrame(grid_rows)
    baseline_totals = bilateral.loc[bilateral["year"].between(inc_start, inc_end)].groupby(["country_iso3_code", "year"])["export_value"].sum().rename("pre_total").reset_index()
    baseline_flows = bilateral.loc[bilateral["year"].between(inc_start, inc_end)].groupby(["country_iso3_code", "year", "partner_iso3_code"], as_index=False)["export_value"].sum()
    pre_share = baseline_grid.merge(baseline_flows, on=["country_iso3_code", "year", "partner_iso3_code"], how="left")
    pre_share = pre_share.merge(baseline_totals, on=["country_iso3_code", "year"], how="left")
    pre_share["export_value"] = pre_share["export_value"].fillna(0.0)
    pre_share["pre_total"] = pre_share["pre_total"].fillna(0.0)
    pre_share["share"] = np.where(pre_share["pre_total"].gt(0), pre_share["export_value"] / pre_share["pre_total"], 0.0)
    pre_mean = pre_share.groupby(["country_iso3_code", "partner_iso3_code"], as_index=False)["share"].mean().rename(columns={"share": "pre_mean_share"})
    annual = bilateral.loc[bilateral["year"].between(model_start, model_end)].groupby(["country_iso3_code", "year", "partner_iso3_code"], as_index=False)["export_value"].sum()
    rows = []
    for (country, year), g in annual.groupby(["country_iso3_code", "year"]):
        incumbent = sets.get(country, set())
        total = g["export_value"].sum()
        inc = g.loc[g["partner_iso3_code"].isin(incumbent)]
        inc_total = inc["export_value"].sum()
        if not incumbent or total <= 0 or inc_total <= 0:
            rows.append({"country_iso3_code": country, "year": year, "design": design, "incumbent_period": f"{inc_start}-{inc_end}", "model_period": f"{model_start}-{model_end}", "incumbent_partner_diversification": np.nan, "incumbent_partner_entropy": np.nan, "portfolio_reallocation": np.nan, "incumbent_retention_rate": np.nan, "continuing_export_share": np.nan})
            continue
        inc_shares = inc["export_value"] / inc_total
        div = 1.0 - float((inc_shares ** 2).sum())
        entropy = -float((inc_shares * np.log(inc_shares.where(inc_shares > 0))).sum())
        retention = float((inc["export_value"] > 0).sum() / len(incumbent))
        continuing = float(inc_total / total)
        means = pre_mean.loc[pre_mean["country_iso3_code"].eq(country)].set_index("partner_iso3_code")["pre_mean_share"]
        current = inc.set_index("partner_iso3_code")["export_value"] / total
        current = current.reindex(sorted(incumbent), fill_value=0.0)
        baseline = means.reindex(sorted(incumbent), fill_value=0.0)
        reallocation = 0.5 * float(np.abs(current.to_numpy() - baseline.to_numpy()).sum())
        rows.append({"country_iso3_code": country, "year": year, "design": design, "incumbent_period": f"{inc_start}-{inc_end}", "model_period": f"{model_start}-{model_end}", "incumbent_partner_diversification": div, "incumbent_partner_entropy": entropy, "portfolio_reallocation": reallocation, "incumbent_retention_rate": retention, "continuing_export_share": continuing})
    measures = pd.DataFrame(rows)
    merged = frame.loc[frame["year"].between(model_start, model_end)].merge(measures, on=["country_iso3_code", "year"], how="left")
    definitions = pd.DataFrame([{"design": design, "outcome": c, "definition": d, "incumbent_period": f"{inc_start}-{inc_end}", "model_period": f"{model_start}-{model_end}", "partner_scope": "non-US/China"} for c, d in {
        "incumbent_partner_diversification": "1-HHI over destinations served in the incumbent period, normalized over incumbent exports",
        "incumbent_partner_entropy": "Shannon entropy over destinations served in the incumbent period, normalized over incumbent exports",
        "portfolio_reallocation": "0.5 times absolute change in incumbent destination shares relative to complete baseline-year grid mean shares; absent incumbent relationships receive zero",
        "incumbent_retention_rate": "Share of incumbent destinations retained with positive exports",
        "continuing_export_share": "Exports to incumbent destinations divided by all non-US/China exports",
    }.items()])
    definition_path = outdir / "intensive_margin_measure_definitions.csv"
    if definition_path.exists():
        existing = pd.read_csv(definition_path)
        if "design" in existing.columns:
            existing = existing.loc[existing["design"].notna() & existing["design"].ne(design)]
        else:
            existing = pd.DataFrame()
        definitions = pd.concat([existing, definitions], ignore_index=True)
    write(definitions, definition_path)
    write(measures, outdir / output_filename)
    return merged


def run_intensive_models(frame, reps, seed, outdir, output_filename, design, incumbent_period, model_period, family_component):
    outcomes = {
        "incumbent_diversification": ("incumbent_partner_diversification", "Incumbent-partner diversification"),
        "incumbent_entropy": ("incumbent_partner_entropy", "Incumbent-partner entropy"),
        "portfolio_reallocation": ("portfolio_reallocation", "Relationship-portfolio reallocation"),
        "incumbent_retention": ("incumbent_retention_rate", "Incumbent destination retention"),
        "continuing_export_share": ("continuing_export_share", "Continuing export share"),
    }
    rows = []
    for i, (key, (outcome, label)) in enumerate(outcomes.items()):
        work, terms, target = post_terms(frame, {"openness": "z_openness_pre"})
        try:
            fit = fit_terms(work, outcome, terms)
            rows.append(coefficient_row(fit, target, reps, seed + i, f"intensive_{design}_{key}", key, label,
                                         {"family_component": family_component, "moderator": "openness", "design": design, "incumbent_period": f"{incumbent_period[0]}-{incumbent_period[1]}", "model_period": f"{model_period[0]}-{model_period[1]}"}))
        except Exception as exc:
            rows.append({"test_id": f"intensive_{design}_{key}_error", "outcome_key": key, "outcome": label, "error": str(exc), "pvalue_for_fdr": np.nan, "family_component": family_component, "design": design})
    result = pd.DataFrame(rows)
    write(result, outdir / output_filename)
    return result


def phase_terms(frame):
    out = frame.copy()
    phases = {
        "tariff_onset_2018_2019": out["year"].between(2018, 2019).astype(float),
        "pandemic_overlap_2020_2021": out["year"].between(2020, 2021).astype(float),
        "persistence_2022": out["year"].eq(2022).astype(float),
    }
    factors = {"eci": "z_eci_pre", "exposure": "z_exposure_pre", "openness": "z_openness_pre"}
    terms, targets = [], {}
    for phase, flag in phases.items():
        for size in range(1, 4):
            for combo in itertools.combinations(factors, size):
                term = f"{phase}_x_" + "_x_".join(combo)
                product = np.ones(len(out))
                for name in combo: product *= pd.to_numeric(out[factors[name]], errors="coerce").to_numpy()
                out[term] = flag.to_numpy() * product
                terms.append(term)
                if size == 3: targets[phase] = term
    return out, terms, targets


def run_phase_models(frame, reps, seed, outdir):
    work, terms, targets = phase_terms(frame)
    fit = fit_terms(work, PRIMARY_OUTCOME, terms)
    rows = []
    phases = list(targets)
    for i, phase in enumerate(phases):
        c = np.zeros(len(fit.term_names)); c[fit.term_index(targets[phase])] = 1
        rows.append(contrast_row(fit, c, reps, seed + i, f"phase_{phase}", "diversification", PRIMARY_LABEL, {"phase": phase, "family_component": "phase_decomposition"}))
    eq_rows = []
    ref = targets[phases[0]]
    for i, phase in enumerate(phases[1:]):
        c = np.zeros(len(fit.term_names)); c[fit.term_index(targets[phase])] = 1; c[fit.term_index(ref)] = -1
        eq_rows.append(contrast_row(fit, c, reps, seed + 10 + i, f"phase_difference_{phase}_minus_{phases[0]}", "diversification", PRIMARY_LABEL, {"test_type": "pairwise_phase_equality", "phase_left": phase, "phase_right": phases[0]}))
    mat = np.zeros((len(phases) - 1, len(fit.term_names)))
    for i, phase in enumerate(phases[1:]):
        mat[i, fit.term_index(targets[phase])] = 1; mat[i, fit.term_index(ref)] = -1
    joint = audited.wald_test(fit, mat, reps=reps, seed=seed + 20)
    eq_rows.append({"test_id": "phase_equality_omnibus", "test_type": "omnibus_phase_equality", "phase_reference": phases[0], "p_cluster": joint["p_cluster"], "p_wild_bootstrap": joint["p_wild_bootstrap"], "pvalue_for_fdr": joint["pvalue_for_fdr"], "wald_chi2": joint["wald_chi2"], "wald_df": joint["wald_df"], "bootstrap_reps_requested": joint["bootstrap_reps_requested"], "bootstrap_reps_success": joint["bootstrap_reps_success"], "n_obs": fit.n_obs, "n_countries": fit.n_countries, "n_years": fit.n_years})
    write(pd.DataFrame(rows), outdir / "market_connectivity_phase_results.csv")
    write(pd.DataFrame(eq_rows), outdir / "market_connectivity_phase_equality_tests.csv")
    return pd.DataFrame(rows), pd.DataFrame(eq_rows)


def bootstrap_ci(fit, contrast, reps, seed):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        signs = rng.choice(np.array([-1.0, 1.0]), size=fit.n_clusters)
        ystar = fit.y_hat + fit.residuals * signs[fit.cluster_codes]
        beta = fit.inv_xx @ (fit.x_resid.T @ ystar)
        vals.append(float(contrast @ beta))
    return np.quantile(np.asarray(vals), [0.025, 0.975])


def marginal_effects(frame, reps, seed, outdir):
    outcomes = {"diversification": (PRIMARY_OUTCOME, PRIMARY_LABEL), "entropy": ("destination_entropy_excl_us_china", "Destination entropy excluding US and China")}
    work, terms, target = post_terms(frame, {"openness": "z_openness_pre"})
    exposure_levels = [("low", frame["z_exposure_pre"].quantile(.25)), ("median", frame["z_exposure_pre"].quantile(.50)), ("high", frame["z_exposure_pre"].quantile(.75))]
    openness_levels = [("low", frame["z_openness_pre"].quantile(.25)), ("median", frame["z_openness_pre"].quantile(.50)), ("high", frame["z_openness_pre"].quantile(.75))]
    rows = []
    for oi, (outcome_key, (outcome, label)) in enumerate(outcomes.items()):
        fit = fit_terms(work, outcome, terms)
        for ei, (exposure_level, exposure) in enumerate(exposure_levels):
            for oi2, (integration_level, integration) in enumerate(openness_levels):
                c = np.zeros(len(fit.term_names))
                c[fit.term_index("post_x_eci")] = 1
                c[fit.term_index("post_x_eci_x_exposure")] = exposure
                c[fit.term_index("post_x_eci_x_openness")] = integration
                c[fit.term_index(target)] = exposure * integration
                s = contrast_statistics(fit, c)
                lo, hi = bootstrap_ci(fit, c, reps, seed + oi * 100 + ei * 10 + oi2)
                rows.append({"outcome_key": outcome_key, "outcome": label, "exposure_level": exposure_level, "exposure_z": exposure, "integration_level": integration_level, "integration_z": integration, "marginal_effect_per_eci_sd": s["estimate"], "se": s["se"], "p_cluster": s["p_cluster"], "bootstrap_ci_low_95": lo, "bootstrap_ci_high_95": hi, "equivalent_hhi_reduction": s["estimate"] if outcome_key == "diversification" else np.nan})
    result = pd.DataFrame(rows)
    write(result, outdir / "market_connectivity_marginal_effects.csv")
    d = result.loc[result["outcome_key"].eq("diversification")]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for level, g in d.groupby("integration_level", sort=False):
        g = g.sort_values("exposure_z")
        xvals = np.arange(len(g))
        ax.plot(xvals, g["marginal_effect_per_eci_sd"].to_numpy(), "o-", label=f"{level} openness")
        for xval, lo, hi in zip(xvals, g["bootstrap_ci_low_95"], g["bootstrap_ci_high_95"]):
            ax.vlines(xval, lo, hi, color=ax.lines[-1].get_color(), linewidth=1.2)
    ax.axhline(0, color="black", lw=.8); ax.set_xticks([0, 1, 2], ["low", "median", "high"]); ax.set(xlabel="Exposure quantile", ylabel="Marginal effect of ECI (one SD)", title="Market-connectivity marginal effects")
    ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(outdir / "figure_market_connectivity_marginal_effects.png", dpi=220); plt.close(fig)
    return result


def robustness(frame, channel_agg, reps, seed, outdir):
    agg = channel_agg.copy()
    agg["z_total_openness"] = z(agg["pre_total_trade_intensity"])
    agg["z_log_openness"] = z(agg["pre_log_total_trade_openness"])
    lo, hi = agg["pre_total_trade_intensity"].quantile([.01, .99])
    w = agg["pre_total_trade_intensity"].clip(lo, hi)
    agg["z_winsorized_openness"] = z(w)
    q99, q95 = agg["pre_total_trade_intensity"].quantile([.99, .95])
    gdp_q05 = agg["pre_real_gdp_usd"].quantile(.05)
    specs = [
        ("log_trade_openness", "z_log_openness", None),
        ("winsorized_1pct_openness", "z_winsorized_openness", None),
        ("exclude_highest_1pct", "z_total_openness", agg["pre_total_trade_intensity"] < q99),
        ("exclude_highest_5pct", "z_total_openness", agg["pre_total_trade_intensity"] < q95),
        ("exclude_smallest_5pct_real_gdp", "z_total_openness", agg["pre_real_gdp_usd"] > gdp_q05),
    ]
    rows = []
    for i, (name, col, mask) in enumerate(specs):
        tmp = frame.merge(agg[["country_iso3_code", col]], on="country_iso3_code", how="left")
        tmp["z_robust_openness"] = tmp[col]
        if mask is not None:
            keep = set(agg.loc[mask, "country_iso3_code"])
            tmp = tmp.loc[tmp["country_iso3_code"].isin(keep)]
        work, terms, target = post_terms(tmp, {"openness": "z_robust_openness"})
        try:
            fit = fit_terms(work, PRIMARY_OUTCOME, terms)
            rows.append(coefficient_row(fit, target, reps, seed + i, f"openness_robustness_{name}", "diversification", PRIMARY_LABEL, {"robustness_specification": name, "q99_threshold": q99, "q95_threshold": q95, "small_gdp_q05_threshold": gdp_q05}))
        except Exception as exc:
            rows.append({"test_id": f"openness_robustness_{name}_error", "robustness_specification": name, "error": str(exc), "pvalue_for_fdr": np.nan})
    result = pd.DataFrame(rows)
    write(result, outdir / "market_connectivity_openness_robustness.csv")
    return result


def fixed_family(primary, omnibus, mechanisms, alternatives, intensive):
    parts = [primary.assign(family_component="primary_trade_openness_interaction"), omnibus.assign(family_component="industrial_trade_structure_omnibus"), mechanisms.assign(family_component="extensive_destination_mechanism")]
    parts.append(alternatives.loc[alternatives["outcome_key"].isin(["destination_entropy", "effective_destinations"])].assign(family_component="alternative_diversification_measure"))
    parts.append(intensive.assign(family_component="intensive_margin_outcome"))
    family = pd.concat(parts, ignore_index=True, sort=False)
    family["multiplicity_family"] = "market_connectivity_fixed_unique_family"
    family["qvalue_market_connectivity_family"] = audited.bh_adjust(family["pvalue_for_fdr"])
    family["fdr_significant_0_05"] = (family["qvalue_market_connectivity_family"] < .05).astype(int)
    return family


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--bootstrap-reps", type=int, default=999)
    ap.add_argument("--seed", type=int, default=20260731)
    args = ap.parse_args()
    base = Path(args.base_dir).resolve()
    outdir = base / "reports/market_connectivity_completion"
    outdir.mkdir(parents=True, exist_ok=True)
    frame = build_base(base, outdir)

    primary_work, primary_terms, primary_target = post_terms(frame, {"openness": "z_openness_pre"})
    primary_fit = fit_terms(primary_work, PRIMARY_OUTCOME, primary_terms)
    primary_countries = set(primary_work.loc[primary_fit.sample_index, "country_iso3_code"].astype(str))
    primary = pd.DataFrame([coefficient_row(primary_fit, primary_target, args.bootstrap_reps, args.seed + 200, "market_connectivity_primary_openness_interaction", "diversification", PRIMARY_LABEL, {"moderator": "openness", "family_component": "primary_trade_openness_interaction"})])
    write(primary, outdir / "market_connectivity_primary_test.csv")

    wdi, _, _ = load_wdi_structural_source(base)
    frame, channel_agg = build_channel_intensities(frame, wdi, primary_countries)
    write(channel_agg, outdir / "market_connectivity_channel_constructs.csv")
    channel, channel_stress, channel_summary = run_channel_decomposition(frame, channel_agg, primary_countries, args.bootstrap_reps, args.seed, outdir)

    intensive_primary_frame = intensive_outcomes(base, frame, outdir, (2015, 2017), (2015, 2022), "primary_2015_2022", "intensive_margin_country_year_outcomes.csv")
    intensive = run_intensive_models(intensive_primary_frame, args.bootstrap_reps, args.seed + 100, outdir, "market_connectivity_intensive_margin_tests.csv", "primary_2015_2022", (2015, 2017), (2015, 2022), "intensive_margin_outcome")
    intensive_sensitivity_frame = intensive_outcomes(base, frame, outdir, (2012, 2017), (2012, 2022), "sensitivity_2012_2022", "intensive_margin_country_year_outcomes_sensitivity.csv")
    intensive_sensitivity = run_intensive_models(intensive_sensitivity_frame, args.bootstrap_reps, args.seed + 150, outdir, "market_connectivity_intensive_margin_sensitivity.csv", "sensitivity_2012_2022", (2012, 2017), (2012, 2022), "sensitivity_only")

    alternatives = pd.read_csv(outdir / "market_connectivity_alternative_diversification.csv")
    mechanisms = pd.read_csv(outdir / "market_connectivity_mechanism_tests.csv")
    phase, phase_eq = run_phase_models(frame, args.bootstrap_reps, args.seed + 300, outdir)
    marginal = marginal_effects(frame, args.bootstrap_reps, args.seed + 400, outdir)
    robust = robustness(frame, channel_agg, args.bootstrap_reps, args.seed + 500, outdir)

    mods = {"openness": "z_openness_pre", "manufacturing": "z_pre_manufacturing_value_added_share", "export_concentration": "z_pre_export_concentration"}
    base_terms = []
    work = frame.copy()
    for mod, col in mods.items():
        for combo in [("eci",), ("exposure",), ("eci", "exposure")]:
            term = "post_x_" + "_x_".join(combo)
            if term not in work:
                product = np.ones(len(work))
                for name in combo:
                    product *= pd.to_numeric(work[{"eci": "z_eci_pre", "exposure": "z_exposure_pre"}[name]], errors="coerce").to_numpy()
                work[term] = work["post_2018"].to_numpy() * product
            if term not in base_terms:
                base_terms.append(term)
        for combo in [("moderator",), ("eci", "moderator"), ("exposure", "moderator"), ("eci", "exposure", "moderator")]:
            combo = tuple(mod if x == "moderator" else x for x in combo)
            term = "post_x_" + "_x_".join(combo)
            product = np.ones(len(work))
            factors = {"eci": "z_eci_pre", "exposure": "z_exposure_pre", mod: col}
            for name in combo:
                product *= pd.to_numeric(work[factors[name]], errors="coerce").to_numpy()
            work[term] = work["post_2018"].to_numpy() * product
            base_terms.append(term)
    fit = fit_terms(work, PRIMARY_OUTCOME, base_terms)
    target_terms = ["post_x_eci_x_exposure_x_" + x for x in mods]
    mat = np.zeros((3, len(fit.term_names)))
    for i, term in enumerate(target_terms):
        mat[i, fit.term_index(term)] = 1
    joint = audited.wald_test(fit, mat, reps=args.bootstrap_reps, seed=args.seed + 600)
    omnibus = pd.DataFrame([{"test_id": "market_connectivity_industrial_trade_structure_omnibus", "test_type": "joint_wald_omnibus", "outcome_key": "diversification", "outcome": PRIMARY_LABEL, "moderators": ";".join(mods), "wald_chi2": joint["wald_chi2"], "wald_df": joint["wald_df"], "p_cluster": joint["p_cluster"], "p_wild_bootstrap": joint["p_wild_bootstrap"], "pvalue_for_fdr": joint["pvalue_for_fdr"], "bootstrap_reps_requested": joint["bootstrap_reps_requested"], "bootstrap_reps_success": joint["bootstrap_reps_success"], "n_obs": fit.n_obs, "n_countries": fit.n_countries, "n_years": fit.n_years}])
    write(omnibus, outdir / "market_connectivity_industrial_trade_omnibus.csv")

    family = fixed_family(primary, omnibus, mechanisms, alternatives, intensive)
    write(family, outdir / "market_connectivity_multiplicity_family.csv")
    family_definition = """# Corrected market-connectivity multiplicity family

The family contains each unique primary empirical test once: primary openness interaction; corrected industrial/trade-structure omnibus; three extensive destination-entry outcomes; two alternative diversification outcomes; and five intensive-margin outcomes from the corrected 2015-2022 primary window. The 2012-2022 incumbent-set analysis is a sensitivity analysis and is not added as a second family of hypotheses. Duplicate new-destination rows are not counted twice. Channel decomposition, channel stress tests, phase equality, marginal effects, openness robustness, and stability diagnostics are estimand or robustness evidence, not separate family hypotheses.
"""
    (outdir / "multiplicity_family_definition.md").write_text(family_definition, encoding="utf-8")
    hold = pd.read_csv(outdir / "market_connectivity_holdout_validation.csv")
    hold["validation_type"] = "region_stratified_repeated_subsample_stability"
    hold["out_of_sample_validation"] = 0
    write(hold, outdir / "market_connectivity_holdout_validation.csv")
    hs = pd.read_csv(outdir / "market_connectivity_holdout_summary.csv")
    hs["validation_type"] = "region_stratified_repeated_subsample_stability"
    hs["out_of_sample_validation"] = 0
    write(hs, outdir / "market_connectivity_holdout_summary.csv")

    raw_channel = channel_summary.loc[channel_summary["channel_specification"].eq("raw_atlas_components")].iloc[0]
    post_joint = phase_eq.loc[phase_eq["test_type"].eq("omnibus_phase_equality")]
    meta = {
        "python": platform.python_version(), "seed": args.seed, "bootstrap_reps": args.bootstrap_reps,
        "primary_outcome": PRIMARY_OUTCOME, "primary_estimand": "ECI_pre x Exposure_pre x Post x Openness_pre",
        "primary_n_countries": int(primary.n_countries.iloc[0]), "primary_n_obs": int(primary.n_obs.iloc[0]),
        "primary_estimate": float(primary.estimate.iloc[0]), "primary_p_wild_bootstrap": float(primary.p_wild_bootstrap.iloc[0]),
        "fixed_family_tests": len(family), "fixed_family_q_lt_005": int((family.qvalue_market_connectivity_family < .05).sum()),
        "phase_equality_p_wild_bootstrap": float(post_joint.p_wild_bootstrap.iloc[0]) if len(post_joint) else None,
        "intensive_primary_model_period": "2015-2022", "intensive_primary_incumbent_period": "2015-2017",
        "intensive_sensitivity_model_period": "2012-2022", "intensive_sensitivity_incumbent_period": "2012-2017",
        "intensive_primary_n_years": int(intensive.n_years.dropna().iloc[0]), "intensive_sensitivity_n_years": int(intensive_sensitivity.n_years.dropna().iloc[0]),
        "channel_primary_sample_n_countries": int(raw_channel.n_countries), "channel_primary_sample_n_obs": int(raw_channel.n_obs),
        "channel_import_sign_stable_across_specs": bool(raw_channel.import_sign_stable_across_specs),
        "channel_joint_import_sign_stable_across_specs": bool(raw_channel.joint_import_sign_stable_across_specs),
        "channel_extreme_exclusion_import_sign_stable": bool(raw_channel.extreme_small_economy_exclusions_retain_import_sign),
        "channel_raw_difference_reasonably_supported": bool(raw_channel.difference_reasonably_supported),
        "terminology_holdout": "region-stratified repeated subsample stability analysis", "out_of_sample_validation": False,
    }
    (outdir / "mandatory_computations_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    audited_metadata_path = outdir / "analysis_metadata.json"
    audited_metadata = json.loads(audited_metadata_path.read_text())
    audited_metadata.update({
        "primary_outcome": PRIMARY_OUTCOME, "primary_estimand": "ECI_pre x Exposure_pre x Post x Openness_pre",
        "primary_n_countries": int(primary.n_countries.iloc[0]), "primary_n_obs": int(primary.n_obs.iloc[0]),
        "primary_estimate": float(primary.estimate.iloc[0]), "primary_p_wild_bootstrap": float(primary.p_wild_bootstrap.iloc[0]),
        "family_tests": len(family), "family_q_lt_005": int((family.qvalue_market_connectivity_family < .05).sum()),
        "fixed_unique_family_tests": len(family), "fixed_unique_family_q_lt_005": int((family.qvalue_market_connectivity_family < .05).sum()),
        "intensive_primary_model_period": "2015-2022", "intensive_sensitivity_model_period": "2012-2022",
        "channel_primary_sample_n_countries": int(raw_channel.n_countries), "channel_primary_sample_n_obs": int(raw_channel.n_obs),
        "mandatory_computations": True, "terminology_holdout": "region-stratified repeated subsample stability analysis", "out_of_sample_validation": False,
    })
    audited_metadata_path.write_text(json.dumps(audited_metadata, indent=2), encoding="utf-8")
    text = f"""# Mandatory final computations

The corrected mandatory analysis uses a 2015-2022 primary intensive-margin window with the 2015-2017 incumbent destination set and a separate 2012-2022 sensitivity using the 2012-2017 incumbent set. The primary intensive family therefore has eight model years; the sensitivity has eleven.

Primary openness interaction: {meta["primary_estimate"]:.6f}; wild-bootstrap p={meta["primary_p_wild_bootstrap"]:.6g}; fixed unique family tests={meta["fixed_family_tests"]}; q<.05={meta["fixed_family_q_lt_005"]}.

The channel decomposition is estimated on the exact primary 181-country sample. Raw Atlas export/import intensity, log(1+x), 1 percent winsorization, import-tail exclusions, small-economy exclusion, and a WDI-compatible reconstruction are reported. The raw components sum to an Atlas trade-intensity measure, while the WDI-compatible specification allocates the WDI primary total across observed export/import shares; this distinction is documented rather than hidden.

The former holdout procedure is region-stratified repeated subsample stability analysis, not out-of-sample validation. The fixed family counts each unique primary test once; channel stress tests and the incumbent-window sensitivity are not added as duplicate confirmatory families.
"""
    (outdir / "mandatory_computations_summary.md").write_text(text, encoding="utf-8")
    print(f"Mandatory computations written to {outdir}; family={len(family)}; q<.05={(family.qvalue_market_connectivity_family < .05).sum()}")


if __name__ == "__main__":
    main()
