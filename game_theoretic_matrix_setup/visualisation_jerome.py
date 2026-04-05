# =========================
# PREP: group_label
# =========================

if 'group_label' not in df_gt_summary.columns:
    def create_group_label(row):
        return f"See:{row['see_emotions']}, α:{row['alpha']}"
    
    df_gt_summary['group_label'] = df_gt_summary.apply(create_group_label, axis=1)


# =========================
# PLOT 1 — GINI
# =========================

fig_gini = viz.evolution_plot(
    data=df_gt_summary,
    x='episode',
    y='gini_coef',
    hue='group_label',
    
    title='Évolution de l’inégalité (Coefficient de Gini) par groupe expérimental',
    xlabel='Épisode',
    ylabel='Coefficient de Gini',
    
    smooth=True,
    smoothing_window=15,          # 👈 plus doux (important)
    smooth_method="exponential",  # 👈 meilleur pour learning curves
    
    confidence_interval=95,       # 👈 IC 95%
    
    markers=False,                # 👈 évite surcharge visuelle
    palette='colorblind',
    figsize=(14, 7)
)


# =========================
# PLOT 2 — STEPS
# =========================

fig_steps = viz.evolution_plot(
    data=df_gt_summary,
    x='episode',
    y='steps_final',
    hue='group_label',
    
    title='Évolution de la durée des épisodes (Steps) par groupe expérimental',
    xlabel='Épisode',
    ylabel='Nombre de steps',
    
    smooth=True,
    smoothing_window=15,
    smooth_method="exponential",
    
    confidence_interval=95,
    
    markers=False,
    palette='colorblind',
    figsize=(14, 7)
)