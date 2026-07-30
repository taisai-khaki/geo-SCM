# Simpson Paradox Factors: Full Results

This report summarizes sign-reversal diagnostics by stratifying factor.
A factor is considered a paradox candidate when subgroup estimates frequently reverse sign vs pooled.

## Overall Factor Ranking (Raw)
- rents_q4: avg reversal share=0.75, DV rows with >=50% reversals=3
- income_q4: avg reversal share=0.67, DV rows with >=50% reversals=3
- open_q4: avg reversal share=0.67, DV rows with >=50% reversals=2
- tariff_q4: avg reversal share=0.64, DV rows with >=50% reversals=2
- wgi_q4: avg reversal share=0.50, DV rows with >=50% reversals=2
- size_q4: avg reversal share=0.42, DV rows with >=50% reversals=2
- coi_q4: avg reversal share=0.42, DV rows with >=50% reversals=2
- post_group: avg reversal share=0.33, DV rows with >=50% reversals=1
- eci_t3: avg reversal share=0.33, DV rows with >=50% reversals=1
- intensity_q4: avg reversal share=0.33, DV rows with >=50% reversals=2

## Overall Factor Ranking (Controlled + Year FE)
- income_q4: avg reversal share=0.53, DV rows with >=50% reversals=2
- rents_q4: avg reversal share=0.50, DV rows with >=50% reversals=2
- wgi_q4: avg reversal share=0.44, DV rows with >=50% reversals=2
- eci_t3: avg reversal share=0.44, DV rows with >=50% reversals=1
- gpr_time_q4: avg reversal share=0.33, DV rows with >=50% reversals=2
- intensity_q4: avg reversal share=0.33, DV rows with >=50% reversals=2
- open_q4: avg reversal share=0.25, DV rows with >=50% reversals=1
- tariff_q4: avg reversal share=0.19, DV rows with >=50% reversals=0
- size_q4: avg reversal share=0.17, DV rows with >=50% reversals=1
- coi_q4: avg reversal share=0.17, DV rows with >=50% reversals=1

## Within vs Between Conflict
- H1_DV1_Stability: within=0.4377, between=-0.2051, opposite_sign=1
- H2_DV2_ExportRecovery: within=0.2458, between=-0.1521, opposite_sign=1
- H3_DV3_Diversification: within=0.0132, between=-0.0020, opposite_sign=1

## Files
- simpson_factor_overall_by_mode.csv
- simpson_factor_detail_sig_reversals.csv
- simpson_dv_mode_overview.csv
- simpson_extended_scan_summary.csv
- simpson_within_between.csv