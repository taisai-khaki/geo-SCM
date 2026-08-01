from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "market_connectivity_completion"


def test_market_connectivity_required_outputs():
    required = [
        "analysis_metadata.json",
        "README.md",
        "market_connectivity_primary_test.csv",
        "market_connectivity_alternative_diversification.csv",
        "market_connectivity_mechanism_tests.csv",
        "market_connectivity_industrial_trade_omnibus.csv",
        "market_connectivity_event_study_coefficients.csv",
        "market_connectivity_event_study_joint_tests.csv",
        "market_connectivity_holdout_validation.csv",
        "market_connectivity_holdout_summary.csv",
        "market_connectivity_leave_one_country_out.csv",
        "market_connectivity_leave_one_region_out.csv",
        "market_connectivity_market_access_robustness.csv",
        "market_connectivity_multiplicity_family.csv",
        "market_access_measure_coverage.csv",
        "figure_market_connectivity_event_study.png",
        "mandatory_computations_metadata.json",
        "mandatory_computations_summary.md",
        "market_connectivity_channel_constructs.csv",
        "market_connectivity_channel_decomposition.csv",
        "market_connectivity_channel_stress_tests.csv",
        "market_connectivity_channel_stress_summary.csv",
        "intensive_margin_country_year_outcomes.csv",
        "intensive_margin_country_year_outcomes_sensitivity.csv",
        "intensive_margin_measure_definitions.csv",
        "market_connectivity_intensive_margin_tests.csv",
        "market_connectivity_intensive_margin_sensitivity.csv",
        "market_connectivity_phase_results.csv",
        "market_connectivity_phase_equality_tests.csv",
        "market_connectivity_marginal_effects.csv",
        "figure_market_connectivity_marginal_effects.png",
        "market_connectivity_openness_robustness.csv",
    ]
    assert not [name for name in required if not (OUT / name).exists()]


def test_primary_estimand_and_outcome_are_frozen():
    metadata = json.loads((OUT / "analysis_metadata.json").read_text())
    mandatory = json.loads((OUT / "mandatory_computations_metadata.json").read_text())
    primary = pd.read_csv(OUT / "market_connectivity_primary_test.csv")
    assert metadata["primary_estimand"] == "ECI_pre x Exposure_pre x Post x Openness_pre"
    assert mandatory["primary_outcome"] == "partner_diversification_excl_us_china"
    assert primary.loc[0, "outcome_key"] == "diversification"
    assert primary.loc[0, "bootstrap_reps_requested"] == 999
    assert primary.loc[0, "n_countries"] == 181


def test_event_study_is_four_way_and_saturated_in_lower_orders():
    coefficients = pd.read_csv(OUT / "market_connectivity_event_study_coefficients.csv")
    assert set(coefficients["event_year"]) == {2012, 2013, 2014, 2015, 2016, 2018, 2019, 2020, 2021, 2022}
    assert coefficients["lower_order_terms_per_year"].eq(7).all()
    joints = pd.read_csv(OUT / "market_connectivity_event_study_joint_tests.csv")
    assert set(joints["test_type"]) == {"pretrend_joint_wald", "post_period_joint_wald"}
    assert joints["bootstrap_reps_requested"].eq(999).all()


def test_fixed_family_has_unique_tests_once():
    family = pd.read_csv(OUT / "market_connectivity_multiplicity_family.csv")
    assert len(family) == 12
    assert family["test_id"].is_unique
    assert family["qvalue_market_connectivity_family"].between(0, 1).all()
    assert set(family["family_component"]) == {
        "primary_trade_openness_interaction",
        "industrial_trade_structure_omnibus",
        "extensive_destination_mechanism",
        "alternative_diversification_measure",
        "intensive_margin_outcome",
    }


def test_mechanisms_and_alternatives_are_distinct():
    mechanisms = pd.read_csv(OUT / "market_connectivity_mechanism_tests.csv")
    alternatives = pd.read_csv(OUT / "market_connectivity_alternative_diversification.csv")
    assert len(mechanisms) == 3
    assert len(alternatives) == 4
    assert "partner_diversification_excl_us_china" not in set(alternatives["outcome"].astype(str))


def test_holdout_and_influence_diagnostics_cover_effective_sample():
    holdouts = pd.read_csv(OUT / "market_connectivity_holdout_validation.csv")
    holdout_summary = pd.read_csv(OUT / "market_connectivity_holdout_summary.csv")
    country_summary = pd.read_csv(OUT / "market_connectivity_leave_one_country_summary.csv")
    region_summary = pd.read_csv(OUT / "market_connectivity_leave_one_region_summary.csv")
    assert len(holdouts) == 100
    assert holdout_summary.loc[0, "repetitions_successful"] == 100
    assert country_summary.loc[0, "total_countries"] == 181
    assert region_summary.loc[0, "total_regions"] == 7
    assert holdout_summary.loc[0, "positive_sign_proportion"] == 1.0
    assert holdout_summary.loc[0, "validation_type"] == "region_stratified_repeated_subsample_stability"
    assert holdout_summary.loc[0, "out_of_sample_validation"] == 0
    assert country_summary.loc[0, "positive_sign_proportion"] == 1.0
    assert region_summary.loc[0, "positive_sign_proportion"] == 1.0


def test_market_access_coverage_is_explicit():
    coverage = pd.read_csv(OUT / "market_access_measure_coverage.csv")
    assert set(coverage.loc[coverage["available"].eq(0), "measure"]) == {
        "import_source_breadth",
        "logistics_trade_facilitation",
    }
    robustness = pd.read_csv(OUT / "market_connectivity_market_access_robustness.csv")
    assert len(robustness) == 3


def test_mandatory_decompositions_and_phase_outputs():
    metadata = json.loads((OUT / "mandatory_computations_metadata.json").read_text())
    assert metadata["bootstrap_reps"] == 999
    assert metadata["intensive_primary_model_period"] == "2015-2022"
    assert metadata["intensive_sensitivity_model_period"] == "2012-2022"
    channels = pd.read_csv(OUT / "market_connectivity_channel_decomposition.csv")
    assert len(channels) == 5
    assert channels["n_countries"].eq(181).all()
    assert channels["n_obs"].eq(1991).all()
    assert set(channels["channel_type"]) == {"individual", "joint_model", "formal_equality_test"}
    assert channels["bootstrap_reps_success"].eq(999).all()
    stress = pd.read_csv(OUT / "market_connectivity_channel_stress_tests.csv")
    summary = pd.read_csv(OUT / "market_connectivity_channel_stress_summary.csv")
    assert len(stress) == 35
    assert len(summary) == 7
    assert summary["import_estimate"].gt(0).all()
    assert summary["joint_import_estimate"].gt(0).all()
    intensive = pd.read_csv(OUT / "market_connectivity_intensive_margin_tests.csv")
    assert len(intensive) == 5
    assert intensive["design"].eq("primary_2015_2022").all()
    assert intensive["n_years"].eq(8).all()
    assert intensive["test_id"].is_unique
    sensitivity = pd.read_csv(OUT / "market_connectivity_intensive_margin_sensitivity.csv")
    assert len(sensitivity) == 5
    assert sensitivity["design"].eq("sensitivity_2012_2022").all()
    assert sensitivity["n_years"].eq(11).all()
    definitions = pd.read_csv(OUT / "intensive_margin_measure_definitions.csv")
    assert len(definitions) == 10
    assert set(definitions["design"]) == {"primary_2015_2022", "sensitivity_2012_2022"}
    phase = pd.read_csv(OUT / "market_connectivity_phase_results.csv")
    equality = pd.read_csv(OUT / "market_connectivity_phase_equality_tests.csv")
    assert len(phase) == 3
    assert len(equality) == 3
    marginal = pd.read_csv(OUT / "market_connectivity_marginal_effects.csv")
    assert len(marginal) == 18
    assert marginal["bootstrap_ci_low_95"].notna().all()
    robustness = pd.read_csv(OUT / "market_connectivity_openness_robustness.csv")
    assert len(robustness) == 5
    assert robustness["bootstrap_reps_success"].eq(999).all()
