"""Measured first-round bounds and deliberately unset failure gates."""

BASELINE_INITIAL_MASK_IOU = 0.3749
THRESHOLDS = {
    'penetration_m': 0.01, 'resting_height_error_m': 0.01,
    'final_speed_m_s': 0.001, 'energy_gain_j_kg': 0.001,
    'ground_fit_rms_m': 0.01, 'final_angular_speed_rad_s': 0.001,
    'coast_distance_error_m': 0.02, 'flight_time_error_s': 0.02,
    'range_error_m': 0.05, 'parabola_vertical_residual_m': 0.01,
    'parabola_horizontal_residual_m': 1e-6,
    # First image: accepted IoU .831735; .80 leaves 3.17 percentage points.
    # First road grid: 0/52 false positives; .02 allows about one road cell.
    # These are scene-specific prototype bounds, not general accuracy claims.
    'initial_mask_iou_min': 0.80, 'road_false_positive_ratio_max': 0.02,
}
