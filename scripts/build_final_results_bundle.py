from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame({"missing_file": [str(path)]})
    return pd.read_csv(path)


def safe_read_text(path: Path) -> str:
    if not path.exists():
        return f"[Missing] {path}"
    return path.read_text(encoding="utf-8", errors="ignore")


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
            "Final interpretation:",
            "- DV2 (Export Recovery): most stable signal; becomes stronger under outlier-robust specs.",
            "- DV1 (GVC Linkage Change): direction is sensitive to influential-country treatment.",
            "- DV3 (Partner Diversification): weak average effect; heterogeneity dominates.",
            "Recommendation for paper framing:",
            "- Present robustness table and heterogeneity narrative instead of single global-effect claim.",
            "- Report influential-country diagnostics transparently and keep baseline + robust specs side by side.",
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
        out_path.write_text(json.dumps(package, indent=2, ensure_ascii=True), encoding="utf-8")
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
        fallback.write_text(json.dumps(package, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"openpyxl not available. Wrote fallback JSON package: {fallback}")


if __name__ == "__main__":
    main()
