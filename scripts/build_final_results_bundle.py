from __future__ import annotations

import argparse
import json
import io
import subprocess
import math
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame({"missing_file": [str(path)]})
    try:
        return pd.read_csv(path)
    except PermissionError:
        repo = Path(__file__).resolve().parents[1]
        relative = path.resolve().relative_to(repo).as_posix()
        raw = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=repo)
        return pd.read_csv(io.BytesIO(raw))


def safe_read_text(path: Path) -> str:
    if not path.exists():
        return f"[Missing] {path}"
    return path.read_text(encoding="utf-8", errors="ignore")


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing_file": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def make_json_safe(value: Any) -> Any:
    """Convert pandas missing values and non-finite numbers to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if value is None:
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)):
        return value
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return make_json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def build_recommendation_lines(base_dir: Path) -> list[str]:
    summary_path = base_dir / "reports" / "outlier_diagnostics" / "outlier_stability_summary.csv"
    robust_path = (
        base_dir / "reports" / "outlier_diagnostics" / "outlier_robustness_reestimation.csv"
    )
    summary = safe_read_csv(summary_path)
    robust = safe_read_csv(robust_path)

    lines: list[str] = []
    lines.append(f"Final integrated package generated on {date.today().isoformat()}.")
    lines.append("Objective: comprehensive country-level analysis with outlier-aware inference.")
    lines.append("Core model term tracked: z_eci:pe in FE DiD-style regressions.")

    if {"dv", "coef_full_z_eci_pe", "p_full_z_eci_pe"}.issubset(summary.columns):
        lines.append("Main FE results:")
        for _, row in summary.iterrows():
            lines.append(
                f"- {row['dv']}: coef={row['coef_full_z_eci_pe']:.4f}, p={row['p_full_z_eci_pe']:.4g}"
            )

    if {"dv", "spec", "coef_z_eci_pe", "pvalue"}.issubset(robust.columns):
        lines.append("Outlier-robust evidence:")
        for dv in robust["dv"].dropna().unique():
            sub = robust[robust["dv"] == dv].copy()
            main = sub[sub["spec"] == "Main_FE_cluster"]
            if main.empty:
                continue
            main_coef = float(main["coef_z_eci_pe"].iloc[0])
            sig_count = int((sub["pvalue"] < 0.05).sum())
            lines.append(
                f"- {dv}: main coef={main_coef:.4f}; significant robust specs={sig_count}/{len(sub)}"
            )

    lines.extend(
        [
            "Legacy package interpretation (superseded by completed tariff-weighted redesign):",
            "- DV2 (Export Recovery): most stable signal; becomes stronger under outlier-robust specs.",
            "- DV1 (GVC Linkage Change): direction is sensitive to influential-country treatment.",
            "- DV3 (Partner Diversification): weak average effect; heterogeneity dominates.",
            "Recommendation for paper framing:",
            "- Present robustness table and heterogeneity narrative instead of single global-effect claim.",
            "- Report influential-country diagnostics transparently and keep baseline + robust specs side by side.",
        ]
    )

    capability_summary_path = base_dir / "reports" / "capability_conversion_redesign" / "analysis_summary.md"
    if capability_summary_path.exists():
        lines.extend(
            [
                "Current methodological-redesign priority:",
                "- Use the frozen pre-shock capability package as the current conservative evidence base.",
                "- Treat H6 as theory-refining/exploratory, not a retroactive confirmatory hypothesis.",
                "- The legacy tables remain useful as an audit trail but do not supersede the redesigned outputs.",
            ]
        )
    completed_summary_path = base_dir / "reports" / "final_design_completion" / "analysis_summary.md"
    if completed_summary_path.exists():
        lines.extend(
            [
                "Current authoritative completed-design interpretation:",
                "- The pre-shock, tariff-weighted design is the current evidence base and supersedes legacy global-effect tables.",
                "- No H1-H5 or H6 test survives the 66-test Benjamini-Hochberg correction at q < .05.",
                "- Direct-country GPR results and profile-matched GPR results must be reported as separate primary and sensitivity evidence.",
                "- Use the final-design output README and analysis summary as the entry point for external review.",
            ]
        )
    structural_summary_path = base_dir / "reports" / "structural_regime_completion" / "analysis_summary.md"
    if structural_summary_path.exists():
        lines.extend(
            [
                "Current structural-regime extension:",
                "- The 2012-2017 structural profile and k=2 regime selection are outcome-independent and must remain separate from the tariff-weighted confirmatory results.",
                "- The structural family contains 84 tests; use its full BH q-values before interpreting continuous moderators or regime differences.",
                "- No structural-family result is promoted to primary theory unless the complete decision rule in structural_design_protocol.md is satisfied.",
            ]
        )
    return lines


def build_metadata_rows(base_dir: Path) -> list[dict[str, Any]]:
    meta_rows: list[dict[str, Any]] = []
    files_to_note = [
        base_dir / "reports" / "comprehensive_analysis_summary.md",
        base_dir / "reports" / "regression_build_metadata.json",
        base_dir / "reports" / "outlier_diagnostics" / "outlier_recommendation_pack.md",
    ]
    for p in files_to_note:
        meta_rows.append(
            {
                "item": "exists",
                "path": str(p),
                "value": p.exists(),
            }
        )

    reg_meta_path = base_dir / "reports" / "regression_build_metadata.json"
    if reg_meta_path.exists():
        reg_meta = json.loads(reg_meta_path.read_text(encoding="utf-8"))
        for k, v in reg_meta.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                meta_rows.append({"item": f"reg_meta:{k}", "path": str(reg_meta_path), "value": v})

    completed_meta_path = base_dir / "reports" / "final_design_completion" / "analysis_metadata.json"
    if completed_meta_path.exists():
        completed_meta = safe_read_json(completed_meta_path)
        for k, v in completed_meta.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                meta_rows.append(
                    {"item": f"completed_design_meta:{k}", "path": str(completed_meta_path), "value": v}
                )
            elif isinstance(v, list):
                meta_rows.append(
                    {
                        "item": f"completed_design_meta:{k}",
                        "path": str(completed_meta_path),
                        "value": json.dumps(v),
                    }
                )

    meta_rows.append(
        {
            "item": "package_generated_date",
            "path": "",
            "value": date.today().isoformat(),
        }
    )
    return meta_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a single-file final results package for external review."
    )
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project base directory.",
    )
    parser.add_argument(
        "--out-file",
        default="reports/final_results_all_in_one.json",
        help="Output single-file package path relative to base-dir (.json or .xlsx).",
    )
    return parser.parse_args()


def build_package(base_dir: Path) -> dict[str, Any]:
    package: dict[str, Any] = {}
    package["assistant_input"] = build_recommendation_lines(base_dir)
    package["metadata"] = build_metadata_rows(base_dir)

    summary_md = safe_read_text(base_dir / "reports" / "comprehensive_analysis_summary.md")
    package["summary_text"] = summary_md
    package["additional_ideas_summary_text"] = safe_read_text(
        base_dir / "reports" / "additional_ideas" / "additional_ideas_summary.md"
    )
    package["advanced_methods_summary_text"] = safe_read_text(
        base_dir / "reports" / "advanced_methods" / "advanced_methods_summary.md"
    )
    package["final_inference_summary_text"] = safe_read_text(
        base_dir / "reports" / "final_inference" / "final_inference_summary.md"
    )
    package["confirmatory_hypotheses_summary_text"] = safe_read_text(
        base_dir / "reports" / "confirmatory_hypotheses" / "confirmatory_summary.md"
    )
    package["simpson_diagnostics_summary_text"] = safe_read_text(
        base_dir / "reports" / "simpson_diagnostics" / "simpson_diagnostics_summary.md"
    )
    package["simpson_factors_full_report_text"] = safe_read_text(
        base_dir / "reports" / "simpson_diagnostics" / "simpson_factors_full_report.md"
    )
    package["simpson_hypothesis_validity_report_text"] = safe_read_text(
        base_dir / "reports" / "simpson_diagnostics" / "simpson_hypothesis_validity_report.md"
    )
    package["income_group_summary_text"] = safe_read_text(
        base_dir / "reports" / "class_based_tests" / "income_group_summary.md"
    )
    package["rerun_comparison_summary_text"] = safe_read_text(
        base_dir
        / "reports"
        / "rerun_no_rents_vs_with_rents_comparison"
        / "comparison_summary.md"
    )
    package["confirmatory_profile_knn_summary_text"] = safe_read_text(
        base_dir
        / "reports"
        / "confirmatory_hypotheses_profile_knn_no_rents_999"
        / "confirmatory_summary.md"
    )
    package["method_choices_recommended_text"] = safe_read_text(
        base_dir / "reports" / "method_choices_recommended.md"
    )
    package["figure_notes_text"] = safe_read_text(
        base_dir / "reports" / "figures_paper" / "figure_notes.md"
    )
    package["covid_sensitivity_legacy_summary_text"] = safe_read_text(
        base_dir
        / "reports"
        / "covid_sensitivity_legacy_with_rents"
        / "covid_sensitivity_summary.md"
    )
    package["covid_sensitivity_no_rents_summary_text"] = safe_read_text(
        base_dir / "reports" / "covid_sensitivity_no_rents" / "covid_sensitivity_summary.md"
    )

    package["capability_conversion_summary_text"] = safe_read_text(
        base_dir / "reports" / "capability_conversion_redesign" / "analysis_summary.md"
    )
    package["capability_conversion_h6_theory_methods_addendum_text"] = safe_read_text(
        base_dir / "reports" / "capability_conversion_redesign" / "h6_theory_methods_addendum.tex"
    )

    package["capability_conversion_manuscript_decision_note_text"] = safe_read_text(
        base_dir / "reports" / "capability_conversion_redesign" / "manuscript_decision_note.md"
    )

    completed_design_dir = base_dir / "reports" / "final_design_completion"
    package["completed_design_summary_text"] = safe_read_text(completed_design_dir / "analysis_summary.md")
    package["completed_design_methods_addendum_text"] = safe_read_text(
        completed_design_dir / "completed_design_methods_addendum.tex"
    )
    package["completed_design_readme_text"] = safe_read_text(completed_design_dir / "README.md")
    package["completed_design_data_sources_text"] = safe_read_text(
        base_dir / "docs" / "completed_design_data_sources.md"
    )
    package["completed_design_metadata"] = safe_read_json(
        completed_design_dir / "analysis_metadata.json"
    )
    package["completed_design_visual_assets"] = [
        "reports/final_design_completion/figure_completed_confirmatory_coefficients.png",
        "reports/final_design_completion/figure_h6_tariff_weighted_threshold.png",
        "reports/final_design_completion/figure_targeted_exploratory_heterogeneity.png",
    ]

    structural_dir = base_dir / "reports" / "structural_regime_completion"
    package["structural_regime_summary_text"] = safe_read_text(structural_dir / "analysis_summary.md")
    package["structural_regime_protocol_text"] = safe_read_text(structural_dir / "structural_design_protocol.md")
    package["structural_regime_readme_text"] = safe_read_text(structural_dir / "README.md")
    package["structural_regime_visual_assets"] = [
        "reports/structural_regime_completion/figure_structural_regime_profiles.png",
        "reports/structural_regime_completion/figure_structural_regime_coefficients.png",
    ]
    package["final_audited_analysis_readme_text"] = safe_read_text(structural_dir / "FINAL_ANALYSIS_README.md")
    package["final_audited_family_definition_text"] = safe_read_text(structural_dir / "multiplicity_family_definition.md")
    package["final_audited_visual_assets"] = [
        "reports/structural_regime_completion/figure_corrected_event_study_gvc.png",
        "reports/structural_regime_completion/figure_corrected_event_study_recovery.png",
        "reports/structural_regime_completion/figure_corrected_event_study_diversification.png",
    ]
    csv_map = {
        "did_model_results": "reports/did_model_results.csv",
        "main_models_2_4": "reports/comprehensive_table2_4_main_models.csv",
        "detailed_terms_2_4": "reports/comprehensive_table2_4_detailed_terms.csv",
        "moderation_table5": "reports/comprehensive_table5_moderation.csv",
        "robustness_table6": "reports/comprehensive_table6_robustness.csv",
        "desc_table1": "reports/comprehensive_table1_descriptive.csv",
        "vif_table1": "reports/comprehensive_table1_vif.csv",
        "outlier_stability": "reports/outlier_diagnostics/outlier_stability_summary.csv",
        "outlier_robust_reest": "reports/outlier_diagnostics/outlier_robustness_reestimation.csv",
        "outlier_delta_main": "reports/outlier_diagnostics/outlier_robustness_delta_vs_main.csv",
        "influential_countries": "reports/outlier_diagnostics/top_influential_countries_all_dv.csv",
        "recurrent_influential": "reports/outlier_diagnostics/recurrent_influential_countries.csv",
        "weighted_unweighted": "reports/outlier_diagnostics/weighted_vs_unweighted_results.csv",
        "flag_share_size_q": "reports/outlier_diagnostics/flag_share_by_country_size_quartile.csv",
        "cluster_marginal_fx": "reports/cluster_regime_models/cluster_specific_marginal_effects.csv",
        "cluster_outcomes": "reports/exploratory_patterns/cluster_outcome_summary.csv",
        "top_moderators": "reports/exploratory_patterns/top_moderator_candidates.csv",
        "additional_event_study_coefs": "reports/additional_ideas/event_study_coefficients.csv",
        "additional_event_study_tests": "reports/additional_ideas/event_study_joint_tests.csv",
        "additional_sample_split": "reports/additional_ideas/sample_split_results.csv",
        "additional_quantile": "reports/additional_ideas/quantile_results.csv",
        "additional_regime_eci": "reports/additional_ideas/regime_eci_tercile_results.csv",
        "additional_post2022_validation": "reports/additional_ideas/post2022_external_validation.csv",
        "advanced_wild_bootstrap": "reports/advanced_methods/wild_cluster_bootstrap_results.csv",
        "advanced_robust_fe": "reports/advanced_methods/robust_fe_results.csv",
        "advanced_cate_summary": "reports/advanced_methods/causal_heterogeneity_summary.csv",
        "advanced_cate_groups": "reports/advanced_methods/causal_heterogeneity_group_means.csv",
        "advanced_cate_country_rank": "reports/advanced_methods/causal_heterogeneity_country_rank.csv",
        "final_dv2_high_precision_bootstrap": "reports/final_inference/dv2_high_precision_bootstrap.csv",
        "final_mediation_results": "reports/final_inference/mediation_results.csv",
        "final_fdr_all_tests": "reports/final_inference/fdr_all_tests.csv",
        "final_decision_table": "reports/final_inference/final_decision_table.csv",
        "confirmatory_primary_tests": "reports/confirmatory_hypotheses/confirmatory_primary_tests.csv",
        "confirmatory_moderation_tests": "reports/confirmatory_hypotheses/confirmatory_moderation_tests.csv",
        "confirmatory_pretrend_placebo": "reports/confirmatory_hypotheses/confirmatory_pretrend_placebo.csv",
        "confirmatory_multiple_testing": "reports/confirmatory_hypotheses/confirmatory_multiple_testing.csv",
        "confirmatory_robustness_layer": "reports/confirmatory_hypotheses/confirmatory_robustness_layer.csv",
        "confirmatory_hypothesis_matrix": "reports/confirmatory_hypotheses/hypothesis_test_matrix.csv",
        "simpson_raw_summary": "reports/simpson_diagnostics/simpson_screen_raw_summary.csv",
        "simpson_controls_yearfe_summary": "reports/simpson_diagnostics/simpson_screen_controls_yearfe_summary.csv",
        "simpson_within_between": "reports/simpson_diagnostics/simpson_within_between.csv",
        "simpson_extended_scan_summary": "reports/simpson_diagnostics/simpson_extended_scan_summary.csv",
        "simpson_extended_scan_shortlist": "reports/simpson_diagnostics/simpson_extended_scan_shortlist.csv",
        "simpson_factor_overall_by_mode": "reports/simpson_diagnostics/simpson_factor_overall_by_mode.csv",
        "simpson_factor_detail_sig_reversals": "reports/simpson_diagnostics/simpson_factor_detail_sig_reversals.csv",
        "simpson_dv_mode_overview": "reports/simpson_diagnostics/simpson_dv_mode_overview.csv",
        "simpson_hypothesis_validity_matrix": "reports/simpson_diagnostics/simpson_hypothesis_validity_matrix.csv",
        "simpson_hypothesis_validity_by_factor": "reports/simpson_diagnostics/simpson_hypothesis_validity_by_factor.csv",
        "simpson_controlled_positive_sig_with_fdr": "reports/simpson_diagnostics/simpson_controlled_positive_sig_with_fdr.csv",
        "income_group_reference": "reports/class_based_tests/world_bank_income_groups_reference.csv",
        "income_group_coverage": "reports/class_based_tests/income_group_coverage.csv",
        "income_h1_h3_by_group": "reports/class_based_tests/h1_h3_by_official_income_group.csv",
        "income_h4_h5_by_group": "reports/class_based_tests/h4_h5_by_official_income_group.csv",
        "income_h1_h3_support_flags": "reports/class_based_tests/h1_h3_income_group_support_flags.csv",
        "income_h4_h5_support_flags": "reports/class_based_tests/h4_h5_income_group_support_flags.csv",
        "income_h1_h3_fdr": "reports/class_based_tests/income_h1_h3_fdr.csv",
        "income_h4_h5_fdr": "reports/class_based_tests/income_h4_h5_fdr.csv",
        "rerun_no_rents_main_models_2_4": "reports/rerun_no_rents_baseline/comprehensive_table2_4_main_models.csv",
        "rerun_no_rents_detailed_terms_2_4": "reports/rerun_no_rents_baseline/comprehensive_table2_4_detailed_terms.csv",
        "rerun_no_rents_moderation_table5": "reports/rerun_no_rents_baseline/comprehensive_table5_moderation.csv",
        "rerun_no_rents_robustness_table6": "reports/rerun_no_rents_baseline/comprehensive_table6_robustness.csv",
        "rerun_with_rents_main_models_2_4": "reports/rerun_with_rents_baseline/comprehensive_table2_4_main_models.csv",
        "rerun_with_rents_detailed_terms_2_4": "reports/rerun_with_rents_baseline/comprehensive_table2_4_detailed_terms.csv",
        "rerun_with_rents_moderation_table5": "reports/rerun_with_rents_baseline/comprehensive_table5_moderation.csv",
        "rerun_with_rents_robustness_table6": "reports/rerun_with_rents_baseline/comprehensive_table6_robustness.csv",
        "rerun_main_models_comparison": "reports/rerun_no_rents_vs_with_rents_comparison/main_models_comparison.csv",
        "rerun_moderation_comparison": "reports/rerun_no_rents_vs_with_rents_comparison/moderation_comparison.csv",
        "data_audit_control_missing_by_year": "reports/data_audit/control_missing_by_year.csv",
        "data_audit_sample_inclusion_scenarios": "reports/data_audit/sample_inclusion_scenarios.csv",
        "data_audit_h1_h3_with_vs_without_rents": "reports/data_audit/h1_h3_with_vs_without_rents.csv",
        "confirmatory_profile_knn_primary_tests": "reports/confirmatory_hypotheses_profile_knn_no_rents_999/confirmatory_primary_tests.csv",
        "confirmatory_profile_knn_moderation_tests": "reports/confirmatory_hypotheses_profile_knn_no_rents_999/confirmatory_moderation_tests.csv",
        "confirmatory_profile_knn_pretrend_placebo": "reports/confirmatory_hypotheses_profile_knn_no_rents_999/confirmatory_pretrend_placebo.csv",
        "confirmatory_profile_knn_multiple_testing": "reports/confirmatory_hypotheses_profile_knn_no_rents_999/confirmatory_multiple_testing.csv",
        "confirmatory_profile_knn_robustness_layer": "reports/confirmatory_hypotheses_profile_knn_no_rents_999/confirmatory_robustness_layer.csv",
        "confirmatory_profile_knn_hypothesis_matrix": "reports/confirmatory_hypotheses_profile_knn_no_rents_999/hypothesis_test_matrix.csv",
        "confirmatory_profile_knn_gpr_audit": "reports/confirmatory_hypotheses_profile_knn_no_rents_999/gpr_profile_knn_audit_by_year_method.csv",
        "confirmatory_profile_knn_gpr_panel_extract": "reports/confirmatory_hypotheses_profile_knn_no_rents_999/gpr_profile_knn_panel_extract.csv",
        "figure_appendix_event_study_profile_coefficients": "reports/figures_paper/appendix_event_study_export_recovery_profile_knn_coefficients.csv",
        "figure_appendix_event_study_profile_joint_tests": "reports/figures_paper/appendix_event_study_export_recovery_profile_knn_joint_tests.csv",
        "figureX_spec_validation_comparison": "reports/figures_paper/figureX_spec_validation_comparison.csv",
        "covid_legacy_primary_terms": "reports/covid_sensitivity_legacy_with_rents/covid_sensitivity_primary_terms.csv",
        "covid_legacy_moderation_terms": "reports/covid_sensitivity_legacy_with_rents/covid_sensitivity_moderation_terms.csv",
        "covid_no_rents_primary_terms": "reports/covid_sensitivity_no_rents/covid_sensitivity_primary_terms.csv",
        "covid_no_rents_moderation_terms": "reports/covid_sensitivity_no_rents/covid_sensitivity_moderation_terms.csv",
        "capability_conversion_data_audit": "reports/capability_conversion_redesign/data_construction_audit.csv",
        "capability_frozen_country_constructs": "reports/capability_conversion_redesign/frozen_country_constructs.csv",
        "capability_threshold_summary": "reports/capability_conversion_redesign/h6_threshold_summary.csv",
        "capability_threshold_search": "reports/capability_conversion_redesign/h6_threshold_search.csv",
        "capability_threshold_bootstrap_distribution": "reports/capability_conversion_redesign/h6_threshold_bootstrap_distribution.csv",
        "capability_h6_primary_and_robustness": "reports/capability_conversion_redesign/h6_primary_and_robustness_tests.csv",
        "capability_h6_model_coefficients": "reports/capability_conversion_redesign/h6_model_coefficients.csv",
        "capability_h6_holdout_detail": "reports/capability_conversion_redesign/h6_holdout_validation.csv",
        "capability_h6_holdout_summary": "reports/capability_conversion_redesign/h6_holdout_validation_summary.csv",
        "capability_redesigned_h1_h3": "reports/capability_conversion_redesign/redesigned_h1_h3_tests.csv",
        "capability_equivalence_mde": "reports/capability_conversion_redesign/equivalence_mde.csv",
        "capability_h4_by_outcome": "reports/capability_conversion_redesign/h4_redesigned_tests.csv",
        "capability_h4_omnibus": "reports/capability_conversion_redesign/h4_omnibus_joint_test.csv",
        "capability_h5_observed_gpr": "reports/capability_conversion_redesign/h5_observed_gpr_exploratory.csv",
        "capability_gpr_coverage": "reports/capability_conversion_redesign/gpr_observed_coverage.csv",
        "capability_gpr_profile_validation": "reports/capability_conversion_redesign/gpr_pre_shock_profile_validation.csv",
        "capability_income_heterogeneity": "reports/capability_conversion_redesign/income_heterogeneity_log_recovery.csv",
        "capability_income_heterogeneity_omnibus": "reports/capability_conversion_redesign/income_heterogeneity_log_recovery_omnibus.csv",
        "capability_destination_mechanisms": "reports/capability_conversion_redesign/mechanism_destination_entry_tests.csv",
        "capability_h6_temporal_regimes": "reports/capability_conversion_redesign/h6_temporal_regime_exploration.csv",
        "world_bank_country_regions": "data/processed/world_bank_country_regions.csv",
        "completed_tariff_weighted_profiles": "data/processed/tariff_weighted_channel_exposure_2015_2017.csv",
        "completed_source_construction_audit": "reports/final_design_completion/source_and_construction_audit.csv",
        "completed_third_country_sample_audit": "reports/final_design_completion/third_country_sample_audit.csv",
        "completed_annual_section301_rate_schedule": "reports/final_design_completion/annual_section301_rate_schedule.csv",
        "completed_baci_tariff_profile_audit": "reports/final_design_completion/baci_tariff_profile_audit.csv",
        "completed_ustr_tariff_line_extraction_audit": "reports/final_design_completion/ustr_tariff_line_extraction_audit.csv",
        "completed_ustr_hs6_membership_audit": "reports/final_design_completion/ustr_section301_hs6_membership_audit.csv",
        "completed_ustr_tariff_lines_hs8": "reports/final_design_completion/ustr_section301_tariff_lines_hs8.csv",
        "completed_frozen_pre_shock_constructs": "reports/final_design_completion/frozen_pre_shock_constructs_completed.csv",
        "completed_historical_income_groups": "reports/final_design_completion/historical_income_groups_2015_2017.csv",
        "completed_historical_income_group_audit": "reports/final_design_completion/historical_income_group_audit.csv",
        "completed_frozen_regime_thresholds": "reports/final_design_completion/frozen_regime_thresholds.csv",
        "completed_confirmatory_tariff_weighted_tests": "reports/final_design_completion/confirmatory_tariff_weighted_tests.csv",
        "completed_equivalence_and_mde": "reports/final_design_completion/equivalence_and_mde_tariff_weighted.csv",
        "completed_h4_tariff_weighted_tests": "reports/final_design_completion/h4_tariff_weighted_tests.csv",
        "completed_h5_and_h4_omnibus_tests": "reports/final_design_completion/h5_and_h4_omnibus_tests.csv",
        "completed_h5_profile_matched_gpr_sensitivity": "reports/final_design_completion/h5_profile_matched_gpr_sensitivity.csv",
        "completed_h6_tariff_weighted_tests": "reports/final_design_completion/h6_tariff_weighted_tests.csv",
        "completed_h6_tariff_weighted_coefficients": "reports/final_design_completion/h6_tariff_weighted_coefficients.csv",
        "completed_h6_tariff_weighted_threshold_summary": "reports/final_design_completion/h6_tariff_weighted_threshold_summary.csv",
        "completed_h6_tariff_weighted_threshold_search": "reports/final_design_completion/h6_tariff_weighted_threshold_search.csv",
        "completed_h6_tariff_weighted_threshold_bootstrap": "reports/final_design_completion/h6_tariff_weighted_threshold_bootstrap.csv",
        "completed_E1_upper_middle_income_gvc_tests": "reports/final_design_completion/E1_upper_middle_income_gvc_tests.csv",
        "completed_E1_upper_middle_income_gvc_omnibus": "reports/final_design_completion/E1_upper_middle_income_gvc_omnibus.csv",
        "completed_E2_low_eci_export_recovery_tests": "reports/final_design_completion/E2_low_eci_export_recovery_tests.csv",
        "completed_E2_low_eci_export_recovery_omnibus": "reports/final_design_completion/E2_low_eci_export_recovery_omnibus.csv",
        "completed_E3_high_income_gpr_diversification_observed": "reports/final_design_completion/E3_high_income_gpr_diversification_observed.csv",
        "completed_E3_high_income_gpr_diversification_observed_omnibus": "reports/final_design_completion/E3_high_income_gpr_diversification_observed_omnibus.csv",
        "completed_E3_high_income_gpr_diversification_profile_matched": "reports/final_design_completion/E3_high_income_gpr_diversification_profile_matched.csv",
        "completed_E3_high_income_gpr_diversification_profile_matched_omnibus": "reports/final_design_completion/E3_high_income_gpr_diversification_profile_matched_omnibus.csv",
        "completed_frozen_controls_by_year_sensitivity": "reports/final_design_completion/frozen_pre_shock_controls_by_year_sensitivity.csv",
        "completed_destination_entry_mechanism_tests": "reports/final_design_completion/destination_entry_mechanism_tests.csv",
        "completed_gpr_coverage_audit": "reports/final_design_completion/gpr_coverage_audit.csv",
        "completed_gpr_profile_matching_validation": "reports/final_design_completion/gpr_profile_matching_validation.csv",
        "completed_gpr_profile_matching_validation_detail": "reports/final_design_completion/gpr_profile_matching_validation_detail.csv",
        "completed_gpr_profile_matched_imputed_values": "reports/final_design_completion/gpr_profile_matched_imputed_values.csv",
        "completed_full_reported_multiplicity_family": "reports/final_design_completion/full_reported_multiplicity_family.csv",
        "completed_design_panel": "reports/final_design_completion/panel_with_completed_design_constructs.csv",
        "structural_wdi_source": "data/raw/structural_wdi_2012_2017.csv",
        "structural_profile_2012_2017": "reports/structural_regime_completion/structural_profile_2012_2017.csv",
        "structural_profile_audit": "reports/structural_regime_completion/structural_profile_audit.csv",
        "structural_pre_outcome_coverage": "reports/structural_regime_completion/pre_outcome_coverage_audit.csv",
        "structural_exposure_balance": "reports/structural_regime_completion/exposure_balance.csv",
        "structural_exposure_smd": "reports/structural_regime_completion/exposure_smd.csv",
        "structural_exposure_correlations": "reports/structural_regime_completion/exposure_correlations.csv",
        "structural_pre_outcome_process_tests": "reports/structural_regime_completion/pre_outcome_process_tests.csv",
        "structural_pretrend_by_exposure_tercile": "reports/structural_regime_completion/pretrend_by_exposure_tercile.csv",
        "structural_vif": "reports/structural_regime_completion/structural_vif.csv",
        "structural_confounder_candidates": "reports/structural_regime_completion/confounder_candidates.csv",
        "structural_continuous_moderators": "reports/structural_regime_completion/continuous_structural_moderator_tests.csv",
        "structural_balancing_weight_definition": "reports/structural_regime_completion/balancing_weight_definition.csv",
        "structural_regime_selection": "reports/structural_regime_completion/structural_regime_selection.csv",
        "structural_regime_assignments": "reports/structural_regime_completion/structural_regime_assignments.csv",
        "structural_regime_profiles": "reports/structural_regime_completion/structural_regime_profiles.csv",
        "structural_regime_exposure_counts": "reports/structural_regime_completion/structural_regime_exposure_counts.csv",
        "structural_regime_geographic_composition": "reports/structural_regime_completion/structural_regime_geographic_composition.csv",
        "structural_regime_outcome_coverage": "reports/structural_regime_completion/structural_regime_outcome_coverage.csv",
        "structural_regime_coefficients": "reports/structural_regime_completion/structural_regime_specific_coefficients.csv",
        "structural_regime_difference_tests": "reports/structural_regime_completion/structural_regime_difference_tests.csv",
        "structural_regime_power_mde": "reports/structural_regime_completion/structural_regime_power_mde.csv",
        "structural_regime_event_study_coefficients": "reports/structural_regime_completion/structural_regime_event_study_coefficients.csv",
        "structural_regime_pretrend_tests": "reports/structural_regime_completion/structural_regime_event_study_pretrend_tests.csv",
        "structural_regime_placebo_tests": "reports/structural_regime_completion/structural_regime_placebo_tests.csv",
        "structural_regime_leave_one_out": "reports/structural_regime_completion/structural_regime_leave_one_country_out.csv",
        "structural_omitted_confounding": "reports/structural_regime_completion/omitted_confounding_sensitivity.csv",
        "structural_analysis_metadata": "reports/structural_regime_completion/structural_analysis_metadata.csv",
        "structural_full_multiplicity_family": "reports/structural_regime_completion/full_structural_multiplicity_family.csv",
        "structural_regression_panel": "reports/structural_regime_completion/panel_with_structural_regimes.csv",
        "final_audited_sample_audit": "reports/structural_regime_completion/valid_country_sample_audit.csv",
        "final_audited_excluded_entities": "reports/structural_regime_completion/excluded_entities.csv",
        "final_audited_missingness": "reports/structural_regime_completion/structural_missingness_by_country.csv",
        "final_audited_sample_comparison": "reports/structural_regime_completion/structural_sample_comparison.csv",
        "final_audited_profile_base": "reports/structural_regime_completion/structural_profile_final_audit_base.csv",
        "final_clean_exposure_counts": "reports/structural_regime_completion/structural_regime_exposure_counts_clean.csv",
        "final_consensus_selection": "reports/structural_regime_completion/structural_regime_selection_clean.csv",
        "final_consensus_matrix": "reports/structural_regime_completion/structural_regime_consensus_matrix.csv",
        "final_assignment_probabilities": "reports/structural_regime_completion/structural_regime_assignment_probabilities.csv",
        "final_assignments": "reports/structural_regime_completion/structural_regime_assignments_clean.csv",
        "final_profiles": "reports/structural_regime_completion/structural_regime_profiles_clean.csv",
        "final_focal_models": "reports/structural_regime_completion/final_focal_models_2018.csv",
        "final_continuous_moderators": "reports/structural_regime_completion/final_continuous_structural_moderator_tests.csv",
        "final_regime_slopes": "reports/structural_regime_completion/structural_regime_specific_coefficients_corrected.csv",
        "final_regime_differences": "reports/structural_regime_completion/structural_regime_difference_tests_corrected.csv",

        "final_regime_slopes_2019": "reports/structural_regime_completion/structural_regime_specific_coefficients_2019.csv",
        "final_regime_differences_2019": "reports/structural_regime_completion/structural_regime_difference_tests_2019.csv",
        "final_regime_omnibus_2019": "reports/structural_regime_completion/structural_regime_omnibus_validation_2019.csv",
        "final_omnibus_validation": "reports/structural_regime_completion/structural_regime_omnibus_validation.csv",
        "corrected_event_coefficients": "reports/structural_regime_completion/corrected_event_study_coefficients.csv",
        "corrected_event_pretrend": "reports/structural_regime_completion/corrected_event_study_pretrend_tests.csv",
        "corrected_event_post": "reports/structural_regime_completion/corrected_event_study_post_tests.csv",
        "corrected_event_post_summary": "reports/structural_regime_completion/corrected_event_study_post_period_summary.csv",
        "outcome_specific_power": "reports/structural_regime_completion/regime_power_mde_outcome_specific.csv",
        "regime_difference_power": "reports/structural_regime_completion/regime_difference_power_mde.csv",
        "common_sample_adjustments": "reports/structural_regime_completion/progressive_adjustment_common_sample.csv",
        "maximum_sample_adjustments": "reports/structural_regime_completion/progressive_adjustment_maximum_sample.csv",
        "coefficient_stability": "reports/structural_regime_completion/coefficient_stability_summary.csv",
        "weight_diagnostics": "reports/structural_regime_completion/residual_density_weight_diagnostics.csv",
        "weight_balance": "reports/structural_regime_completion/balance_before_after_weighting.csv",
        "theory_balance": "reports/structural_regime_completion/theory_based_balance_diagnostics.csv",
        "moderator_blocks": "reports/structural_regime_completion/structural_moderator_block_omnibus_tests.csv",
        "assignment_robustness": "reports/structural_regime_completion/regime_assignment_robustness.csv",
        "assignment_model_sensitivity": "reports/structural_regime_completion/regime_model_assignment_sensitivity.csv",
        "leave_one_country_summary": "reports/structural_regime_completion/leave_one_country_out_summary_stats.csv",
        "leave_one_country_detail": "reports/structural_regime_completion/leave_one_country_out_summary.csv",
        "leave_one_region": "reports/structural_regime_completion/leave_one_region_out_results.csv",
        "influence_audit": "reports/structural_regime_completion/influential_country_region_audit.csv",
        "placebo_comparison": "reports/structural_regime_completion/placebo_2014_2015_comparison.csv",
        "h6_directional": "reports/structural_regime_completion/h6_directional_2018_2019.csv",
        "equivalence": "reports/structural_regime_completion/final_equivalence_tests.csv",
        "mde": "reports/structural_regime_completion/final_minimum_detectable_effects.csv",
        "post_period_comparison": "reports/structural_regime_completion/post_period_2018_2019_comparison.csv",
        "post_period_long": "reports/structural_regime_completion/post_period_2018_2019_long.csv",
        "post_period_continuous": "reports/structural_regime_completion/post_period_2019_continuous_moderators.csv",
        "final_confirmatory_family": "reports/structural_regime_completion/final_confirmatory_multiplicity_family.csv",
        "final_structural_family": "reports/structural_regime_completion/final_structural_multiplicity_family.csv",
        "reproduction_manifest": "reports/structural_regime_completion/reproduction_manifest.csv",
        "output_file_hashes": "reports/structural_regime_completion/output_file_hashes.csv",
        "final_audit_summary": "reports/structural_regime_completion/final_audit_summary.csv",
    }

    tables: dict[str, Any] = {}
    for key, rel_path in csv_map.items():
        df = safe_read_csv(base_dir / rel_path)
        tables[key] = {
            "source": rel_path,
            "rows": int(len(df)),
            "columns": df.columns.tolist(),
            "records": df.to_dict(orient="records"),
        }
    package["tables"] = tables
    package["generated_on"] = date.today().isoformat()
    return package


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_path = (base_dir / args.out_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    package = build_package(base_dir)

    if out_path.suffix.lower() == ".json":
        out_path.write_text(json.dumps(make_json_safe(package), indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
        print(f"Wrote final JSON package: {out_path}")
        print(f"Tables included: {len(package['tables'])}")
        return

    # Optional XLSX branch if dependency is available.
    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            pd.DataFrame({"assistant_input": package["assistant_input"]}).to_excel(
                writer, sheet_name="assistant_input", index=False
            )
            pd.DataFrame(package["metadata"]).to_excel(writer, sheet_name="metadata", index=False)
            pd.DataFrame(
                {"comprehensive_analysis_summary_md": str(package["summary_text"]).splitlines()}
            ).to_excel(writer, sheet_name="summary_text", index=False)
            for key, value in package["tables"].items():
                df = pd.DataFrame(value["records"])
                if df.empty:
                    df = pd.DataFrame({"note": [f"No rows for {key}"]})
                df.to_excel(writer, sheet_name=key[:31], index=False)
        print(f"Wrote final XLSX package: {out_path}")
        print(f"Sheets: {3 + len(package['tables'])}")
    except ModuleNotFoundError:
        fallback = out_path.with_suffix(".json")
        fallback.write_text(json.dumps(make_json_safe(package), indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
        print(f"openpyxl not available. Wrote fallback JSON package: {fallback}")


if __name__ == "__main__":
    main()
