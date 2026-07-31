from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from run_final_audited_analysis import (
    OUTCOMES,
    STRUCTURAL_LABELS,
    STRUCTURAL_VARS,
    add_base_terms,
    add_moderator_terms,
    bh_adjust,
    coefficient_row,
    fit_terms,
    prepare_panel,
    corrected_regime_models,
    focal_event_terms,
    wald_test,
)
from run_capability_conversion_analysis import wild_cluster_bootstrap_contrast


def run_continuous(frame: pd.DataFrame, post_start: int, reps: int, seed: int) -> pd.DataFrame:
    work_frame = frame.copy()
    work_frame["post_2018"] = work_frame["year"].ge(post_start).astype(float)
    rows = []
    for oi, (outcome_key, (outcome, label)) in enumerate(OUTCOMES.items()):
        for vi, variable in enumerate(STRUCTURAL_VARS):
            work, terms = add_base_terms(work_frame)
            work, mod_terms = add_moderator_terms(work, "z_" + variable)
            terms += mod_terms
            fit = fit_terms(work, outcome, terms)
            row = coefficient_row(
                fit,
                "eci_exposure_post_x_moderator",
                reps,
                seed + oi * 100 + vi,
            )
            row.update(
                {
                    "test_id": f"post_continuous_{post_start}_{outcome_key}_{variable}",
                    "analysis": "continuous_moderator",
                    "outcome_key": outcome_key,
                    "outcome": label,
                    "entity": "z_" + variable,
                    "moderator": "z_" + variable,
                    "moderator_label": STRUCTURAL_LABELS[variable],
                    "post_start": post_start,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def run_event_post_sensitivity(
    frame: pd.DataFrame, regimes: list[str], post_start: int, reps: int, seed: int
) -> pd.DataFrame:
    rows = []
    for oi, (outcome_key, (outcome, label)) in enumerate(OUTCOMES.items()):
        variants = [("full_sample", frame, False), ("pooled_regime_interacted", frame, True)]
        variants += [
            (f"regime_{regime}", frame.loc[frame["structural_regime"].eq(regime)], False)
            for regime in regimes
        ]
        for vi, (variant, subset, interacted) in enumerate(variants):
            work, terms, triple = focal_event_terms(subset, interacted, regimes[0])
            fit = fit_terms(work, outcome, terms)
            selected = [term for year, term in triple.items() if year >= post_start]
            if interacted and len(regimes) == 2:
                selected += [
                    f"{term}_x_{regimes[1]}"
                    for year, term in triple.items()
                    if year >= post_start and f"{term}_x_{regimes[1]}" in fit.term_names
                ]
            matrix = np.zeros((len(selected), len(fit.term_names)))
            for idx, term in enumerate(selected):
                matrix[idx, fit.term_index(term)] = 1.0
            joint = wald_test(
                fit, matrix, reps=reps,
                seed=seed + oi * 1000 + vi,
            )
            rows.append(
                {
                    "test_id": f"corrected_event_post_period_{post_start}_{variant}_{outcome_key}",
                    "variant": variant,
                    "outcome_key": outcome_key,
                    "outcome": label,
                    "post_start": post_start,
                    "test_type": "joint_post_period",
                    **joint,
                    "n_obs": fit.n_obs,
                    "n_countries": fit.n_countries,
                }
            )
    return pd.DataFrame(rows)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--bootstrap-reps", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    base = Path(args.base_dir).resolve()
    out = base / "reports" / "structural_regime_completion"
    panel = pd.read_csv(base / "reports" / "final_design_completion" / "panel_with_completed_design_constructs.csv")
    profile = pd.read_csv(out / "structural_profile_final_audit_base.csv")
    assignment = pd.read_csv(out / "structural_regime_assignment_probabilities.csv")
    profile = profile.loc[profile["primary_structural_sample"].astype(bool)].copy()
    model_panel = prepare_panel(panel, profile, assignment)
    current = pd.read_csv(out / "final_continuous_structural_moderator_tests.csv")
    current = current.copy()
    current["analysis"] = "continuous_moderator"
    current["entity"] = current["moderator"]
    current["post_start"] = 2018
    current["pvalue_for_fdr"] = current["p_wild_bootstrap"]
    sensitivity = run_continuous(model_panel, 2019, args.bootstrap_reps, args.seed + 1900)
    sensitivity.to_csv(out / "post_period_2019_continuous_moderators.csv", index=False)
    regimes = sorted(model_panel["structural_regime"].dropna().unique())
    slopes_2019, differences_2019, validation_2019, _ = corrected_regime_models(
        model_panel, 2019, regimes, args.bootstrap_reps, args.seed + 2900
    )
    slopes_2019.to_csv(out / "structural_regime_specific_coefficients_2019.csv", index=False)
    differences_2019.to_csv(out / "structural_regime_difference_tests_2019.csv", index=False)
    validation_2019.to_csv(out / "structural_regime_omnibus_validation_2019.csv", index=False)
    event_post_2019 = run_event_post_sensitivity(
        model_panel, regimes, 2019, args.bootstrap_reps, args.seed + 3900
    )

    slopes_2018 = pd.read_csv(out / "structural_regime_specific_coefficients_corrected.csv")
    differences_2018 = pd.read_csv(out / "structural_regime_difference_tests_corrected.csv")

    regime_long_rows = []
    for table, analysis, post_start in [
        (slopes_2018, "regime_slope", 2018),
        (differences_2018, "regime_difference", 2018),
        (slopes_2019, "regime_slope", 2019),
        (differences_2019, "regime_difference", 2019),
    ]:
        for _, row in table.iterrows():
            if pd.isna(row.get("p_wild_bootstrap")):
                continue
            if analysis == "regime_slope":
                entity = f"regime_{row['regime']}"
            elif row["test_type"] == "pairwise_regime_difference":
                entity = f"{row['regime_left']}_minus_{row['regime_right']}"
            else:
                entity = "regime_omnibus"
            regime_long_rows.append(
                {
                    "analysis": analysis,
                    "outcome_key": row["outcome_key"],
                    "outcome": row["outcome"],
                    "entity": entity,
                    "post_start": post_start,
                    "estimate": row["estimate"],
                    "se": row["se"],
                    "p_wild_bootstrap": row["p_wild_bootstrap"],
                    "n_obs": row["n_obs"],
                    "n_countries": row["n_countries"],
                }
            )
    regime_long = pd.DataFrame(regime_long_rows)
    sensitivity["pvalue_for_fdr"] = sensitivity["p_wild_bootstrap"]

    long = pd.read_csv(out / "post_period_2018_2019_long.csv")
    long = pd.concat([long, regime_long], ignore_index=True)
    add = []
    for table in [current, sensitivity]:
        for _, row in table.iterrows():
            add.append(
                {
                    "analysis": "continuous_moderator",
                    "outcome_key": row["outcome_key"],
                    "outcome": row["outcome"],
                    "entity": row["entity"],
                    "post_start": row["post_start"],
                    "estimate": row["estimate"],
                    "se": row["se"],
                    "p_wild_bootstrap": row["p_wild_bootstrap"],
                    "n_obs": row["n_obs"],
                    "n_countries": row["n_countries"],
                }
            )
    long = pd.concat([long, pd.DataFrame(add)], ignore_index=True)
    summary_rows = []
    for keys, group in long.groupby(["analysis", "outcome_key", "entity"]):
        wide = group.drop_duplicates("post_start").set_index("post_start")
        if 2018 not in wide.index or 2019 not in wide.index:
            continue
        a, b = wide.loc[2018], wide.loc[2019]
        summary_rows.append(
            {
                "analysis": keys[0],
                "outcome_key": keys[1],
                "entity": keys[2],
                "estimate_2018": a["estimate"],
                "se_2018": a["se"],
                "p_wild_2018": a["p_wild_bootstrap"],
                "estimate_2019": b["estimate"],
                "se_2019": b["se"],
                "p_wild_2019": b["p_wild_bootstrap"],
                "sign_stability": np.sign(a["estimate"]) == np.sign(b["estimate"]),
                "n_obs_2018": a["n_obs"],
                "n_obs_2019": b["n_obs"],
                "n_countries_2018": a["n_countries"],
                "n_countries_2019": b["n_countries"],
                "pvalue_for_fdr": b["p_wild_bootstrap"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    long.to_csv(out / "post_period_2018_2019_long.csv", index=False)
    summary.to_csv(out / "post_period_2018_2019_comparison.csv", index=False)

    event_post_2018 = pd.read_csv(out / "corrected_event_study_post_tests.csv")
    event_post_2018["post_start"] = 2018
    event_post_summary = pd.concat([event_post_2018, event_post_2019], ignore_index=True)
    event_post_summary.to_csv(out / "corrected_event_study_post_period_summary.csv", index=False)

    family = pd.read_csv(out / "final_structural_multiplicity_family.csv")
    additions = []
    for _, row in sensitivity.iterrows():
        additions.append(row.to_dict())
    h6 = pd.read_csv(out / "h6_directional_2018_2019.csv")
    additions += h6.loc[h6["post_start"].eq(2019)].to_dict("records")
    additions += slopes_2019.to_dict("records")
    additions += differences_2019.to_dict("records")
    additions += event_post_2019.to_dict("records")
    additions = pd.DataFrame(additions)
    if not additions.empty:
        additions["multiplicity_family"] = "final_exploratory_structural_family"
        additions = additions.loc[
            ~additions["test_id"].isin(family["test_id"])
        ]
        family = pd.concat([family, additions], ignore_index=True)
    family["qvalue_final_structural_family"] = bh_adjust(family["pvalue_for_fdr"])
    family["fdr_significant_0_05"] = (family["qvalue_final_structural_family"] < 0.05).astype(int)
    family.to_csv(out / "final_structural_multiplicity_family.csv", index=False)

    log = out / "reproduction_log.txt"
    log.write_text(
        "python scripts/run_final_audited_analysis.py --bootstrap-reps 999 --mi-reps 20 --stability-reps 100 --seed 20260731\n"
        "python scripts/run_post_period_sensitivity.py --bootstrap-reps 999 --seed 20260731\n"
        "All inputs are repository-relative; no local absolute paths are required.\n",
        encoding="utf-8",
    )
    print("post-period sensitivity rows", len(summary))
    print("final structural family rows", len(family))
    print("q<.05", int((family["qvalue_final_structural_family"] < 0.05).sum()))


if __name__ == "__main__":
    main()
