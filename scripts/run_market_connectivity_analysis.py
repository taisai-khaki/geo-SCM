from __future__ import annotations
import argparse, itertools, json, platform
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import run_final_audited_analysis as audited
from run_capability_conversion_analysis import contrast_statistics, wild_cluster_bootstrap_contrast
from run_structural_regime_analysis import fit_terms, load_wdi_structural_source, build_structural_profile

PRIMARY_OUTCOME = "partner_diversification_excl_us_china"
PRIMARY_LABEL = "Partner diversification excluding US and China"
EVENT_REFERENCE = 2017
OUTCOME_ALTS = {
    "destination_entropy": ("destination_entropy_excl_us_china", "Destination entropy excluding US and China"),
    "effective_destinations": ("effective_destinations_excl_us_china", "Effective number of destinations excluding US and China"),
    "new_destination_export_share": ("new_non_uschina_destination_export_share", "Export share through new non-US/China destinations"),
    "persistent_new_destinations": ("persistent_new_non_uschina_destination_count", "Persistent new non-US/China destination count"),
}
MECHANISMS = {
    "new_destination_entry": ("new_non_uschina_destination_count", "New non-US/China destination entry count"),
    "new_destination_export_share": ("new_non_uschina_destination_export_share", "Export share through new non-US/China destinations"),
    "persistent_destination_entry": ("persistent_new_non_uschina_destination_count", "Persistent new non-US/China destination count"),
}
LABELS = {
    "openness": "Pre-shock trade openness",
    "established_destinations": "Pre-shock established non-US/China destinations",
    "destination_entropy": "Pre-shock destination entropy excluding US and China",
    "outside_us_china_share": "Pre-shock export share outside US and China",
}


def z(x):
    x = pd.to_numeric(x, errors="coerce")
    sd = x.std(ddof=0)
    return (x - x.mean()) / sd if np.isfinite(sd) and sd else pd.Series(np.nan, index=x.index)


def write(df, path):
    df.to_csv(path, index=False)


def post_terms(frame, mods):
    out = frame.copy()
    factors = {"eci": "z_eci_pre", "exposure": "z_exposure_pre", **mods}
    names, terms = list(factors), []
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            term = "post_x_" + "_x_".join(combo)
            product = np.ones(len(out))
            for name in combo:
                product *= pd.to_numeric(out[factors[name]], errors="coerce").to_numpy()
            out[term] = out["post_2018"].to_numpy() * product
            terms.append(term)
    target = "post_x_eci_x_exposure_x_" + "_x_".join(mods)
    return out, terms, target


def row(fit, target, reps, seed, test_id, outcome_key, outcome, extra=None):
    c = np.zeros(len(fit.term_names))
    c[fit.term_index(target)] = 1
    stat = contrast_statistics(fit, c)
    boot = wild_cluster_bootstrap_contrast(fit, c, reps=reps, seed=seed, alternative="two-sided")
    result = {**stat, **boot, "test_id": test_id, "outcome_key": outcome_key, "outcome": outcome,
              "term": target, "pvalue_for_fdr": boot["p_wild_bootstrap"],
              "n_obs": fit.n_obs, "n_countries": fit.n_countries, "n_years": fit.n_years}
    if extra: result.update(extra)
    return result


def build(base, outdir):
    panel = pd.read_csv(base / "reports/final_design_completion/panel_with_completed_design_constructs.csv")
    wdi, _, _ = load_wdi_structural_source(base)
    profile, _, _ = build_structural_profile(panel, wdi)
    profile = audited.standardize_profile(profile)
    profile["z_openness_pre"] = profile["z_pre_trade_openness"]
    keep = ["country_iso3_code", "country_name", "wb_region", "z_eci_pre", "z_exposure_pre",
            "z_openness_pre", "z_pre_manufacturing_value_added_share", "z_pre_export_concentration",
            "pre_trade_openness", "eci_pre", "exposure_pre"]
    frame = panel.drop(columns=["wb_region"], errors="ignore").merge(profile[keep], on="country_iso3_code", how="inner")
    frame["post_2018"] = frame["year"].ge(2018).astype(float)
    frame["post_2019"] = frame["year"].ge(2019).astype(float)
    write(profile[["country_iso3_code", "country_name", "wb_region", "eci_pre", "exposure_pre",
                   "pre_trade_openness", "z_eci_pre", "z_exposure_pre", "z_openness_pre"]],
          outdir / "frozen_market_connectivity_profile.csv")
    return frame


def access_measures(frame):
    pre = frame.loc[frame.year.between(2015, 2017)]
    rows = []
    for code, g in pre.groupby("country_iso3_code"):
        rows.append({"country_iso3_code": code,
                     "pre_established_destinations": pd.to_numeric(g.n_non_uschina_destinations, errors="coerce").mean(),
                     "pre_destination_entropy": pd.to_numeric(g.destination_entropy_excl_us_china, errors="coerce").mean(),
                     "pre_outside_us_china_share": 1 - pd.to_numeric(g.us_export_market_dependence_pre, errors="coerce").mean() - pd.to_numeric(g.china_export_market_dependence_pre, errors="coerce").mean()})
    measures = pd.DataFrame(rows)
    coverage = pd.DataFrame([
        {"measure": "openness", "available": 1, "label": LABELS["openness"], "reason": "Primary frozen WDI profile measure"},
        {"measure": "established_destinations", "available": 1, "label": LABELS["established_destinations"], "reason": "Mean 2015-2017 established destinations"},
        {"measure": "destination_entropy", "available": 1, "label": LABELS["destination_entropy"], "reason": "Mean 2015-2017 destination entropy"},
        {"measure": "outside_us_china_share", "available": 1, "label": LABELS["outside_us_china_share"], "reason": "One minus pre-shock US and China export shares"},
        {"measure": "import_source_breadth", "available": 0, "label": "Pre-shock import-source breadth", "reason": "Frozen panel has no bilateral import-source counts"},
        {"measure": "logistics_trade_facilitation", "available": 0, "label": "Logistics/trade-facilitation capacity", "reason": "No harmonized frozen logistics series"},
    ])
    return measures, coverage


def model_family(frame, outcomes, modname, modcol, reps, seed, family):
    rows = []
    for i, (key, (outcome, label)) in enumerate(outcomes.items()):
        work, terms, target = post_terms(frame, {modname: modcol})
        try:
            fit = fit_terms(work, outcome, terms)
            rows.append(row(fit, target, reps, seed + i, f"market_connectivity_{family}_{key}", key, label,
                            {"moderator": modname, "moderator_label": LABELS.get(modname, modname), "family_component": family}))
        except Exception as exc:
            rows.append({"test_id": f"market_connectivity_{family}_{key}_error", "outcome_key": key, "outcome": label, "error": str(exc), "pvalue_for_fdr": np.nan})
    return pd.DataFrame(rows)


def industrial_omnibus(frame, reps, seed):
    mods = {"openness": "z_openness_pre", "manufacturing": "z_pre_manufacturing_value_added_share", "export_concentration": "z_pre_export_concentration"}
    work, terms, _ = post_terms(frame, mods)
    targets = ["post_x_eci_x_exposure_x_" + m for m in mods]
    try:
        fit = fit_terms(work, PRIMARY_OUTCOME, terms)
        mat = np.zeros((3, len(fit.term_names)))
        for i, target in enumerate(targets): mat[i, fit.term_index(target)] = 1
        joint = audited.wald_test(fit, mat, reps=reps, seed=seed)
        b = mat @ fit.beta
        cov = mat @ fit.covariance @ mat.T
        return pd.DataFrame([{"test_id": "market_connectivity_industrial_trade_structure_omnibus",
            "outcome_key": "diversification", "outcome": PRIMARY_LABEL, "test_type": "joint_wald_omnibus",
            "moderators": ";".join(mods), "wald_chi2": float(b @ np.linalg.pinv(cov) @ b), "wald_df": 3,
            "p_cluster": joint["p_cluster"], "p_wild_bootstrap": joint["p_wild_bootstrap"], "pvalue_for_fdr": joint["pvalue_for_fdr"],
            "bootstrap_reps_requested": joint["bootstrap_reps_requested"], "bootstrap_reps_success": joint["bootstrap_reps_success"],
            "n_obs": fit.n_obs, "n_countries": fit.n_countries, "n_years": fit.n_years, "family_component": "industrial_trade_structure_omnibus"}])
    except Exception as exc:
        return pd.DataFrame([{"test_id": "market_connectivity_industrial_trade_structure_omnibus_error", "error": str(exc), "pvalue_for_fdr": np.nan, "family_component": "industrial_trade_structure_omnibus"}])


def event_study(frame, reps, seed, outdir):
    out = frame.copy()
    factors = {"eci": "z_eci_pre", "exposure": "z_exposure_pre", "openness": "z_openness_pre"}
    terms, targets = [], {}
    for year in sorted(pd.to_numeric(out.year).dropna().astype(int).unique()):
        if year == EVENT_REFERENCE: continue
        flag = out.year.eq(year).astype(float).to_numpy()
        for r in range(1, 4):
            for combo in itertools.combinations(factors, r):
                term = "event_" + "_x_".join(combo) + f"_{year}"
                product = np.ones(len(out))
                for name in combo: product *= pd.to_numeric(out[factors[name]], errors="coerce").to_numpy()
                out[term] = flag * product
                terms.append(term)
                if r == 3: targets[year] = term
    try:
        fit = fit_terms(out, PRIMARY_OUTCOME, terms)
        coeff = []
        for year, target in targets.items():
            c = np.zeros(len(fit.term_names)); c[fit.term_index(target)] = 1
            s = contrast_statistics(fit, c)
            coeff.append({"test_id": f"market_connectivity_event_{year}", "event_year": year, "term": target,
                          "estimate": s["estimate"], "se": s["se"], "p_cluster": s["p_cluster"],
                          "ci_low_95": s["ci_low_95"], "ci_high_95": s["ci_high_95"], "n_obs": fit.n_obs,
                          "n_countries": fit.n_countries, "outcome_key": "diversification", "outcome": PRIMARY_LABEL,
                          "lower_order_terms_per_year": 7, "event_reference_year": EVENT_REFERENCE})
        joints = []
        for name, years in [("pretrend", range(2012, 2017)), ("post_period", range(2018, 2023))]:
            selected = [targets[y] for y in years if y in targets]
            mat = np.zeros((len(selected), len(fit.term_names)))
            for i, term in enumerate(selected): mat[i, fit.term_index(term)] = 1
            j = audited.wald_test(fit, mat, reps=reps, seed=seed + (name == "post_period"))
            joints.append({"test_id": f"market_connectivity_event_{name}_joint", "test_type": f"{name}_joint_wald",
                           "years": ";".join(map(str, years)), "outcome_key": "diversification", "outcome": PRIMARY_LABEL,
                           "wald_chi2": j["wald_chi2"], "wald_df": j["wald_df"], "p_cluster": j["p_cluster"],
                           "p_wild_bootstrap": j["p_wild_bootstrap"], "pvalue_for_fdr": j["pvalue_for_fdr"],
                           "bootstrap_reps_requested": j["bootstrap_reps_requested"], "bootstrap_reps_success": j["bootstrap_reps_success"],
                           "n_obs": fit.n_obs, "n_countries": fit.n_countries, "n_years": fit.n_years, "lower_order_terms_per_year": 7})
        coeff = pd.DataFrame(coeff)
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ax.axhline(0, color="black", lw=.8); ax.axvline(EVENT_REFERENCE + .5, color="0.6", ls="--", lw=.9)
        ax.errorbar(coeff.event_year, coeff.estimate, yerr=1.96 * coeff.se, fmt="o-", color="#176b87", capsize=3)
        ax.set(xlabel="Event year (2017 reference)", ylabel="Four-way coefficient", title="Market-connectivity event study")
        ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(outdir / "figure_market_connectivity_event_study.png", dpi=220); plt.close(fig)
        return coeff, pd.DataFrame(joints)
    except Exception as exc:
        return pd.DataFrame([{"test_id": "market_connectivity_event_error", "error": str(exc), "pvalue_for_fdr": np.nan}]), pd.DataFrame([{"test_id": "market_connectivity_event_joint_error", "error": str(exc), "pvalue_for_fdr": np.nan}])


def stat(frame, outcome, terms, target):
    fit = fit_terms(frame, outcome, terms)
    c = np.zeros(len(fit.term_names)); c[fit.term_index(target)] = 1
    return fit, contrast_statistics(fit, c)


def influence(frame, terms, target, baseline):
    countries = sorted(frame.country_iso3_code.dropna().astype(str).unique())
    cr = []
    for code in countries:
        try:
            f, s = stat(frame.loc[frame.country_iso3_code.astype(str).ne(code)], PRIMARY_OUTCOME, terms, target)
            cr.append({"excluded_country": code, "estimate": s["estimate"], "se": s["se"], "p_cluster": s["p_cluster"], "n_obs": f.n_obs, "n_countries": f.n_countries})
        except Exception as exc: cr.append({"excluded_country": code, "error": str(exc)})
    country = pd.DataFrame(cr); valid = country.loc[country.estimate.notna()]
    cs = pd.DataFrame([{"analysis": "leave_one_country_out_openness_interaction", "baseline_estimate": baseline, "min_estimate": valid.estimate.min(), "max_estimate": valid.estimate.max(), "positive_sign_proportion": (valid.estimate > 0).mean(), "successful_exclusions": len(valid), "total_countries": len(countries)}])
    regions = sorted(frame.wb_region.dropna().astype(str).unique())
    rr = []
    for reg in regions:
        try:
            f, s = stat(frame.loc[frame.wb_region.astype(str).ne(reg)], PRIMARY_OUTCOME, terms, target)
            rr.append({"excluded_region": reg, "estimate": s["estimate"], "se": s["se"], "p_cluster": s["p_cluster"], "n_obs": f.n_obs, "n_countries": f.n_countries})
        except Exception as exc: rr.append({"excluded_region": reg, "error": str(exc)})
    region = pd.DataFrame(rr); validr = region.loc[region.estimate.notna()]
    rs = pd.DataFrame([{"analysis": "leave_one_region_out_openness_interaction", "baseline_estimate": baseline, "min_estimate": validr.estimate.min(), "max_estimate": validr.estimate.max(), "positive_sign_proportion": (validr.estimate > 0).mean(), "successful_exclusions": len(validr), "total_regions": len(regions), "region_with_minimum": validr.loc[validr.estimate.idxmin(), "excluded_region"], "region_with_maximum": validr.loc[validr.estimate.idxmax(), "excluded_region"]}])
    return country, cs, region, rs


def holdouts(frame, terms, target, reps, seed):
    rng = np.random.default_rng(seed)
    groups = {str(r): sorted(g.country_iso3_code.dropna().astype(str).unique()) for r, g in frame.groupby("wb_region")}
    rows = []
    for rep in range(1, reps + 1):
        held = []
        for countries in groups.values():
            if len(countries) > 1:
                n = min(max(1, int(round(.2 * len(countries)))), len(countries) - 1)
                held.extend(rng.choice(countries, size=n, replace=False))
        held = sorted(set(held))
        try:
            f, s = stat(frame.loc[~frame.country_iso3_code.astype(str).isin(held)], PRIMARY_OUTCOME, terms, target)
            rows.append({"repetition": rep, "held_out_countries": len(held), "estimate": s["estimate"], "se": s["se"], "p_cluster": s["p_cluster"], "positive_sign": int(s["estimate"] > 0), "n_obs": f.n_obs, "n_countries": f.n_countries})
        except Exception as exc: rows.append({"repetition": rep, "held_out_countries": len(held), "error": str(exc)})
    detail = pd.DataFrame(rows); valid = detail.loc[detail.estimate.notna()]
    summary = pd.DataFrame([{"validation": "region_stratified_repeated_holdout", "repetitions_requested": reps, "repetitions_successful": len(valid), "positive_sign_proportion": (valid.estimate > 0).mean(), "min_estimate": valid.estimate.min(), "max_estimate": valid.estimate.max(), "median_estimate": valid.estimate.median(), "mean_estimate": valid.estimate.mean(), "held_out_countries_mean": valid.held_out_countries.mean()}])
    return detail, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--bootstrap-reps", type=int, default=999)
    ap.add_argument("--holdout-reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260731)
    args = ap.parse_args()
    base = Path(args.base_dir).resolve()
    outdir = base / "reports/market_connectivity_completion"
    outdir.mkdir(parents=True, exist_ok=True)
    frame = build(base, outdir)
    measures, coverage = access_measures(frame)
    write(measures, outdir / "market_access_pre_shock_constructs.csv"); write(coverage, outdir / "market_access_measure_coverage.csv")
    primary_frame, terms, target = post_terms(frame, {"openness": "z_openness_pre"})
    pf = fit_terms(primary_frame, PRIMARY_OUTCOME, terms)
    primary = pd.DataFrame([row(pf, target, args.bootstrap_reps, args.seed, "market_connectivity_primary_openness_interaction", "diversification", PRIMARY_LABEL, {"moderator": "openness", "hypothesis": "Market-connectivity hypothesis", "primary_outcome": PRIMARY_OUTCOME, "family_component": "primary_trade_openness_interaction"})])
    write(primary, outdir / "market_connectivity_primary_test.csv")
    alternatives = model_family(frame, OUTCOME_ALTS, "openness", "z_openness_pre", args.bootstrap_reps, args.seed + 100, "alternative_diversification"); write(alternatives, outdir / "market_connectivity_alternative_diversification.csv")
    mechanisms = model_family(frame, MECHANISMS, "openness", "z_openness_pre", args.bootstrap_reps, args.seed + 200, "conditional_destination_mechanism"); write(mechanisms, outdir / "market_connectivity_mechanism_tests.csv")
    omnibus = industrial_omnibus(frame, args.bootstrap_reps, args.seed + 300); write(omnibus, outdir / "market_connectivity_industrial_trade_omnibus.csv")
    ec, ej = event_study(frame, args.bootstrap_reps, args.seed + 400, outdir); write(ec, outdir / "market_connectivity_event_study_coefficients.csv"); write(ej, outdir / "market_connectivity_event_study_joint_tests.csv")
    effective_countries = set(primary_frame.loc[pf.sample_index, "country_iso3_code"].astype(str))
    effective_frame = primary_frame.loc[primary_frame["country_iso3_code"].astype(str).isin(effective_countries)].copy()
    baseline = float(primary.estimate.iloc[0])
    cd, cs, rd, rs = influence(effective_frame, terms, target, baseline)
    write(cd, outdir / "market_connectivity_leave_one_country_out.csv"); write(cs, outdir / "market_connectivity_leave_one_country_summary.csv"); write(rd, outdir / "market_connectivity_leave_one_region_out.csv"); write(rs, outdir / "market_connectivity_leave_one_region_summary.csv")
    hd, hs = holdouts(effective_frame, terms, target, args.holdout_reps, args.seed + 500); write(hd, outdir / "market_connectivity_holdout_validation.csv"); write(hs, outdir / "market_connectivity_holdout_summary.csv")
    access = frame.merge(measures, on="country_iso3_code", how="left"); ar = []
    for i, (name, col) in enumerate({"established_destinations": "pre_established_destinations", "destination_entropy": "pre_destination_entropy", "outside_us_china_share": "pre_outside_us_china_share"}.items()):
        access[f"z_market_{name}"] = z(access[col])
        try:
            mw, mt, mm = post_terms(access, {name: f"z_market_{name}"}); mf = fit_terms(mw, PRIMARY_OUTCOME, mt)
            ar.append(row(mf, mm, args.bootstrap_reps, args.seed + 600 + i, f"market_access_robustness_{name}", "diversification", PRIMARY_LABEL, {"measure": name, "measure_label": LABELS[name], "multiplicity_status": "robustness_only"}))
        except Exception as exc: ar.append({"test_id": f"market_access_robustness_{name}_error", "measure": name, "error": str(exc), "pvalue_for_fdr": np.nan})
    write(pd.DataFrame(ar), outdir / "market_connectivity_market_access_robustness.csv")
    family = pd.concat([primary.assign(family_component="primary_trade_openness_interaction"), omnibus, mechanisms.assign(family_component="destination_entry_mechanism"), alternatives.assign(family_component="alternative_diversification_measure")], ignore_index=True, sort=False)
    family["multiplicity_family"] = "market_connectivity_fixed_family"
    family["qvalue_market_connectivity_family"] = audited.bh_adjust(family.pvalue_for_fdr)
    family["fdr_significant_0_05"] = (family.qvalue_market_connectivity_family < .05).astype(int)
    write(family, outdir / "market_connectivity_multiplicity_family.csv")
    (outdir / "multiplicity_family_definition.md").write_text("# Market-connectivity fixed multiplicity family\n\nThe family contains exactly nine prespecified tests: the primary trade-openness interaction; one industrial/trade-structure omnibus; three conditional destination-entry mechanism tests; and four alternative diversification measures. Complementary market-access measures, event-year coefficients, holdouts, and influence diagnostics are robustness or validation evidence, not additional confirmatory hypotheses.\n", encoding="utf-8")
    post = ej.loc[ej.test_type.eq("post_period_joint_wald")]
    metadata = {"python": platform.python_version(), "seed": args.seed, "bootstrap_reps": args.bootstrap_reps, "holdout_reps": args.holdout_reps, "primary_outcome": PRIMARY_OUTCOME, "primary_estimand": "ECI_pre x Exposure_pre x Post x Openness_pre", "primary_n_countries": int(primary.n_countries.iloc[0]), "primary_n_obs": int(primary.n_obs.iloc[0]), "primary_estimate": float(primary.estimate.iloc[0]), "primary_p_cluster": float(primary.p_cluster.iloc[0]), "primary_p_wild_bootstrap": float(primary.p_wild_bootstrap.iloc[0]), "family_tests": len(family), "family_q_lt_005": int((family.qvalue_market_connectivity_family < .05).sum()), "event_years": ec.event_year.dropna().astype(int).tolist() if "event_year" in ec else [], "event_post_joint_p_wild": float(post.p_wild_bootstrap.iloc[0]) if len(post) else None, "holdout_positive_sign_proportion": float(hs.positive_sign_proportion.iloc[0])}
    (outdir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    text = f"""# Market-connectivity hypothesis extension

Frozen hypothesis: pre-shock trade openness positively moderates the relationship between productive complexity and post-shock export-partner diversification among countries exposed to the US-China tariff conflict.

Primary estimand: ECI_pre x Exposure_pre x Post x Openness_pre.
Primary outcome: {PRIMARY_LABEL}.
Primary estimate: {metadata["primary_estimate"]:.6f}; wild-bootstrap p-value: {metadata["primary_p_wild_bootstrap"]:.6g}; countries: {metadata["primary_n_countries"]}; observations: {metadata["primary_n_obs"]}.

The corrected event study estimates the four-way coefficient for 2012-2022 with 2017 as the reference year and all seven post-varying lower-order interactions each year. Conditional destination-entry mechanisms, region-stratified repeated holdouts, leave-one-country-out and leave-one-region-out estimates are reported separately. Complementary access measures are established-destination breadth, destination entropy, and export share outside US/China. Import-source breadth and logistics capacity are documented as unavailable rather than proxied.

The fixed multiplicity family contains {len(family)} tests: the primary interaction, one industrial/trade-structure omnibus, three mechanism tests, and four alternative diversification outcomes. This is an additional focused hypothesis, not retroactive confirmation of H1-H5.
"""
    (outdir / "README.md").write_text(text, encoding="utf-8")
    (outdir / "analysis_summary.md").write_text(text, encoding="utf-8")
    print(f"Market-connectivity analysis written to {outdir}; primary={metadata['primary_estimate']:.6f}; p_wild={metadata['primary_p_wild_bootstrap']:.6g}; family={metadata['family_tests']}; q<.05={metadata['family_q_lt_005']}")


if __name__ == "__main__":
    main()
