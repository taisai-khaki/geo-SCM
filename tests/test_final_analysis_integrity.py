from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'structural_regime_completion'

def test_required_final_outputs_exist():
    required = [
        'valid_country_sample_audit.csv', 'excluded_entities.csv',
        'structural_missingness_by_country.csv', 'structural_sample_comparison.csv',
        'structural_regime_selection_clean.csv', 'structural_regime_consensus_matrix.csv',
        'structural_regime_assignment_probabilities.csv', 'structural_regime_assignments_clean.csv',
        'structural_regime_profiles_clean.csv', 'structural_regime_difference_tests_corrected.csv',
        'structural_regime_omnibus_validation.csv', 'corrected_event_study_coefficients.csv',
        'corrected_event_study_pretrend_tests.csv', 'corrected_event_study_post_tests.csv',
        'post_period_2018_2019_comparison.csv',
        'structural_regime_specific_coefficients_2019.csv',
        'structural_regime_difference_tests_2019.csv',
        'structural_regime_omnibus_validation_2019.csv',
        'corrected_event_study_post_period_summary.csv', 'regime_power_mde_outcome_specific.csv',
        'regime_difference_power_mde.csv', 'progressive_adjustment_common_sample.csv',
        'progressive_adjustment_maximum_sample.csv', 'coefficient_stability_summary.csv',
        'residual_density_weight_diagnostics.csv', 'balance_before_after_weighting.csv',
'theory_based_balance_diagnostics.csv',
        'structural_moderator_block_omnibus_tests.csv', 'regime_assignment_robustness.csv',
        'regime_model_assignment_sensitivity.csv', 'leave_one_country_out_summary_stats.csv',
        'leave_one_region_out_results.csv', 'influential_country_region_audit.csv',
        'placebo_2014_2015_comparison.csv', 'final_equivalence_tests.csv',
        'final_minimum_detectable_effects.csv', 'final_confirmatory_multiplicity_family.csv',
        'final_structural_multiplicity_family.csv', 'multiplicity_family_definition.md',
        'reproduction_manifest.csv', 'output_file_hashes.csv', 'reproduction_log.txt',
        'reproducibility_proof.log', 'reproducibility_versions.csv',
    ]
    missing = [name for name in required if not (OUT / name).exists()]
    assert not missing, missing

def test_k2_omnibus_equals_pairwise():
    table = pd.read_csv(OUT / 'structural_regime_omnibus_validation.csv')
    assert table.loc[table['k'] == 2, 'validation_passed'].all()

def test_event_study_has_eci_exposure_terms():
    table = pd.read_csv(OUT / 'corrected_event_study_coefficients.csv')
    assert table['term'].astype(str).str.contains('eci_exposure').all()

def test_primary_sample_constraints():
    table = pd.read_csv(OUT / 'valid_country_sample_audit.csv')
    primary = table.loc[table['primary_structural_sample']]
    assert (primary['cluster_imputed_fields'] <= 2).all()
    assert primary['valid_entity'].all()
    assert primary['valid_exposure'].all()
    assert primary['valid_eci'].all()

def test_post_2019_outputs_present():
    table = pd.read_csv(OUT / 'post_period_2018_2019_comparison.csv')
    assert not table.empty
    assert {'estimate_2018', 'estimate_2019'}.issubset(table.columns)

def test_outcome_power_counts_are_valid():
    table = pd.read_csv(OUT / 'regime_power_mde_outcome_specific.csv')
    assert (table['n_countries_used'] == table['effective_number_of_clusters']).all()

def test_family_qvalues_are_valid():
    table = pd.read_csv(OUT / 'final_structural_multiplicity_family.csv')
    assert table['test_id'].notna().all()
    assert table['qvalue_final_structural_family'].between(0, 1).all()

def test_2019_regime_and_event_outputs():
    validation = pd.read_csv(OUT / 'structural_regime_omnibus_validation_2019.csv')
    assert validation['validation_passed'].all()
    event = pd.read_csv(OUT / 'corrected_event_study_post_period_summary.csv')
    assert 2019 in set(event['post_start'].astype(int))

def test_power_counts_do_not_exceed_observed_panel():
    power = pd.read_csv(OUT / 'regime_power_mde_outcome_specific.csv')
    panel = pd.read_csv(ROOT / 'reports' / 'final_design_completion' / 'panel_with_completed_design_constructs.csv')
    columns = {
        'gvc': 'gvc_adverse_deviation_stability',
        'recovery': 'log_export_recovery',
        'diversification': 'partner_diversification_excl_us_china',
    }
    for key, column in columns.items():
        observed_rows = int(panel[column].notna().sum())
        assert int(power.loc[power['outcome_key'].eq(key), 'n_country_year_observations'].max()) <= observed_rows

def test_family_has_no_obsolete_or_duplicate_ids():
    family = pd.read_csv(OUT / 'final_structural_multiplicity_family.csv')
    assert family['test_id'].notna().all()
    assert family['test_id'].is_unique
    assert family['qvalue_final_structural_family'].notna().all()

def test_frozen_focal_reference_values():
    focal = pd.read_csv(OUT / 'final_focal_models_2018.csv').set_index('test_id')
    expected = {
        'focal_2018_gvc_eci_exposure_post': 0.23079135951473873,
        'focal_2018_recovery_eci_exposure_post': -0.052508028999043645,
        'focal_2018_diversification_eci_exposure_post': -0.013137820287150031,
    }
    for test_id, value in expected.items():
        assert np.isclose(float(focal.loc[test_id, 'estimate']), value, atol=1e-9)
def test_country_validity_keeps_real_economies():
    audit = pd.read_csv(OUT / 'valid_country_sample_audit.csv')
    for code in ['CAF', 'ZAF']:
        row = audit.loc[audit['country_iso3_code'].eq(code)].iloc[0]
        assert bool(row['valid_entity'])
        assert bool(row['primary_structural_sample'])
        assert row['entity_validity_reason'] == 'valid_sovereign_or_analytical_economy'

def test_event_study_scope_matches_confirmatory_samples():
    coefficients = pd.read_csv(OUT / 'corrected_event_study_coefficients.csv')
    full = coefficients.loc[coefficients['variant'].eq('full_sample_confirmatory')]
    expected = {'gvc': 78, 'recovery': 228, 'diversification': 228}
    for key, n in expected.items():
        values = full.loc[full['outcome_key'].eq(key), 'n_countries'].unique()
        assert len(values) == 1 and int(values[0]) == n
    pretrend = pd.read_csv(OUT / 'corrected_event_study_pretrend_tests.csv')
    assert 'full_sample_cleaned_structural' in set(pretrend['variant'])
    assert 'pooled_regime_interacted' in set(pretrend['variant'])

def test_weighting_is_diagnostic_only():
    balance = pd.read_csv(OUT / 'balance_before_after_weighting.csv')
    assert 'joint_balance_pvalue' not in balance.columns
    assert balance['diagnostic_status'].eq('failed_balance').all()
    assert (balance['absolute_smd_after'] > 0.10).all()
    assert not (OUT / 'weighted_model_trimmed_sensitivity.csv').exists()

def test_reproducibility_proof_is_complete():
    log = (OUT / 'reproducibility_proof.log').read_text(encoding='utf-8')
    versions = pd.read_csv(OUT / 'reproducibility_versions.csv')
    assert 'EXIT_CODE: 0' in log
    assert {'python', 'package', 'version'}.issubset(versions.columns)
    assert versions['version'].notna().all()
