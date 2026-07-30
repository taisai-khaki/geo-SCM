# Final Inference Round

This round includes:
- High-precision wild cluster bootstrap for DV2.
- Mediation test (X=z_eci:pe, M=DV2, Y=DV3) with cluster bootstrap.
- Global Benjamini-Hochberg FDR correction across main and robustness tests.

## DV2 high-precision bootstrap
- coef=-0.1031, cluster-p=0.1827, wild-bootstrap p=0.1962, CI=[-0.1637, 0.1481], reps=999

## Mediation (DV2 -> DV3)
- Path a (X->M): coef=-0.1031, p=0.1827
- Path b (M->Y|X): coef=-0.0245, p=3.106e-09
- Indirect a*b=0.0025, boot-p=0.1855, CI=[-0.0014, 0.0074]

## FDR correction
- Total hypotheses: 61
- Raw p<0.05: 16
- FDR q<0.05: 7

## Decision table
- C1 NOT SUPPORTED: DV2 channel effect exists and is robust (high_precision_wild_bootstrap_p=0.1961961961961962)
- C2 NOT SUPPORTED: Effect on DV3 is mediated by DV2 (mediation_indirect_boot_p=0.18546365914786966)
- C3 SUPPORTED: Meaningful evidence remains after FDR correction (count_fdr_significant_tests=7.0)