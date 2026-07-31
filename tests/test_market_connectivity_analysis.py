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
    ]
    assert not [name for name in required if not (OUT / name).exists()]


def test_primary_estimand_and_outcome_are_frozen():
    metadata = json.loads((OUT / "analysis_metadata.json").read_text())
    primary = pd.read_csv(OUT / "market_connectivity_primary_test.csv")
    assert metadata["primary_estimand"] == "ECI_pre x Exposure_pre x Post x Openness_pre"
    assert primary.loc[0, "primary_outcome"] == "partner_diversification_excl_us_china"
    assert primary.loc[0, "bootstrap_reps_requested"] == 999
    assert primary.loc[0, "n_countries"] == 181


def test_event_study_is_four_way_and_saturated_in_lower_orders():
    coefficients = pd.read_csv(OUT / "market_connectivity_event_study_coefficients.csv")
    assert set(coefficients["event_year"]) == {2012, 2013, 2014, 2015, 2016, 2018, 2019, 2020, 2021, 2022}
    assert coefficients["lower_order_terms_per_year"].eq(7).all()
    joints = pd.read_csv(OUT / "market_connectivity_event_study_joint_tests.csv")
    assert set(joints["test_type"]) == {"pretrend_joint_wald", "post_period_joint_wald"}
    assert joints["bootstrap_reps_requested"].eq(999).all()


def test_fixed_family_has_only_nine_tests():
    family = pd.read_csv(OUT / "market_connectivity_multiplicity_family.csv")
    assert len(family) == 9
    assert family["test_id"].is_unique
    assert family["qvalue_market_connectivity_family"].between(0, 1).all()
    assert set(family["family_component"]) == {
        "primary_trade_openness_interaction",
        "industrial_trade_structure_omnibus",
        "destination_entry_mechanism",
        "alternative_diversification_measure",
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
