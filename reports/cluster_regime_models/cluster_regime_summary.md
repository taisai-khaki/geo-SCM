# Cluster-Regime Heterogeneity Results

Model core:
- Outcome ~ z(ECI) + (Post×Exposed) + z(ECI)×(Post×Exposed)
- Plus cluster interaction: z(ECI)×(Post×Exposed)×Cluster
- Country FE + Year FE + clustered SE by country
- Post period: 2019–2022

Primary output:
- `cluster_specific_marginal_effects.csv`
- `cluster_specific_effects_plot.png`