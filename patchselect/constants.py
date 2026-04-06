"""Constants shared across the patch selection pipeline."""

BASE_FEATURE_NAMES = [
    "tissue_frac",
    "odsum_mean",
    "gray_mean",
    "sat_mean",
    "h_mean",
    "h_std",
    "h_q10",
    "h_q50",
    "h_q90",
    "d_mean",
    "d_std",
    "d_q10",
    "d_q50",
    "d_q90",
    "residual_mean",
    "residual_q90",
    "h_hist_00_25",
    "h_hist_25_50",
    "h_hist_50_75",
    "h_hist_75_100",
    "d_hist_00_25",
    "d_hist_25_50",
    "d_hist_50_75",
    "d_hist_75_100",
    "d_pos_mean",
    "nuclei_frac",
    "nuclei_count_density",
    "nuclei_area_mean",
    "nuclei_area_cv",
    "dab_in_nuc_frac",
    "dab_ring_frac",
    "dab_extra_frac",
    "dab_cc_density",
    "dab_edge_to_area",
    "log_lap_var",
    "grad_mean",
    "grad_p90",
    "hole_frac",
    "unexpected_color_frac",
    "fold_frac",
    "border_tissue_frac",
    "compartment_margin",
]

NEIGHBOR_FEATURE_NAMES = [
    "neigh_sem_l1_mean",
    "neigh_sem_l1_max",
    "neigh_d_mean_diff",
    "neigh_posfrac_diff",
    "neigh_nuc_frac_diff",
    "neigh_state_change_frac",
]

FEATURE_NAMES = BASE_FEATURE_NAMES + NEIGHBOR_FEATURE_NAMES

BASE_FEATURE_TO_INDEX = {name: idx for idx, name in enumerate(BASE_FEATURE_NAMES)}
FEATURE_TO_INDEX = {name: idx for idx, name in enumerate(FEATURE_NAMES)}

SEMANTIC_FEATURE_NAMES = [
    "h_mean",
    "h_q10",
    "h_q90",
    "d_mean",
    "d_q10",
    "d_q90",
    "d_hist_25_50",
    "d_hist_50_75",
    "d_hist_75_100",
    "d_pos_mean",
    "nuclei_frac",
    "dab_in_nuc_frac",
    "dab_ring_frac",
    "dab_extra_frac",
    "compartment_margin",
]

SEMANTIC_FEATURE_INDICES = [
    BASE_FEATURE_NAMES.index(name) for name in SEMANTIC_FEATURE_NAMES
]

ROLE_NAMES = (
    "prototype",
    "positive_tail",
    "interface",
    "rare_state",
)
