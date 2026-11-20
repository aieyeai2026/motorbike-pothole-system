#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lane { Left, Center, Right }

impl Lane {
    pub fn as_str(&self) -> &'static str {
        match self {
            Lane::Left => "LEFT",
            Lane::Center => "CENTER",
            Lane::Right => "RIGHT",
        }
    }
}

pub fn classify_lane(x_center_norm: f32) -> Lane {
    let clamped = x_center_norm.clamp(0.0, 1.0);
    if clamped < 0.33 { Lane::Left }
    else if clamped <= 0.66 { Lane::Center }
    else { Lane::Right }
}

pub fn estimate_distance_meters(
    y_bottom_norm: f32,
    roi_horizon_norm: f32,
    min_distance_m: f32,
    max_distance_m: f32,
) -> f32 {
    let horizon = roi_horizon_norm.clamp(0.05, 0.90);
    let y = y_bottom_norm.clamp(horizon, 1.0);
    let road_progress = (y - horizon) / (1.0 - horizon);
    let curved = 1.0 - road_progress.powf(0.65);
    let distance = min_distance_m + curved * (max_distance_m - min_distance_m);
    distance.clamp(min_distance_m, max_distance_m)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CenteringStatus {
    Centered,
    DriftingLeft,
    DriftingRight,
}

impl CenteringStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            CenteringStatus::Centered => "CENTERED",
            CenteringStatus::DriftingLeft => "DRIFTING_LEFT",
            CenteringStatus::DriftingRight => "DRIFTING_RIGHT",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaneCentering {
    pub offset_meters: f32,
    pub dist_left_m: f32,
    pub dist_right_m: f32,
    pub status: CenteringStatus,
}

pub fn compute_lane_centering(
    x_ego: f32,
    x_left_border: f32,
    x_right_border: f32,
    road_width_m: f32,
) -> LaneCentering {
    let road_span = (x_right_border - x_left_border).max(0.01);
    let norm_pos = ((x_ego - x_left_border) / road_span).clamp(0.0, 1.0);
    let dist_left_m = norm_pos * road_width_m;
    let dist_right_m = (1.0 - norm_pos) * road_width_m;
    let offset_meters = (norm_pos - 0.5) * road_width_m;

    let status = if offset_meters.abs() <= 0.25 {
        CenteringStatus::Centered
    } else if offset_meters < -0.25 {
        CenteringStatus::DriftingLeft
    } else {
        CenteringStatus::DriftingRight
    };

    LaneCentering {
        offset_meters,
        dist_left_m,
        dist_right_m,
        status,
    }
}

pub fn compensate_horizon_for_pitch(base_horizon_norm: f32, pitch_deg: f32) -> f32 {
    (base_horizon_norm + pitch_deg * 0.006).clamp(0.15, 0.85)
}

/// 2nd-degree polynomial curve `x(y) = a * y^2 + b * y + c` modeling curving road boundaries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadraticCurve {
    pub a: f32,
    pub b: f32,
    pub c: f32,
}

impl QuadraticCurve {
    pub fn eval(&self, y: f32) -> f32 {
        self.a * y * y + self.b * y + self.c
    }
}

/// Returns true only if ground contact point `(x, y)` lies strictly inside the curved drivable
/// road polygon between `y_horizon` and `y_bottom`, excluding off-road shoulders, walls, and trees.
pub fn is_point_in_drivable_corridor(
    x: f32,
    y: f32,
    y_horizon: f32,
    y_bottom: f32,
    left_curve: &QuadraticCurve,
    right_curve: &QuadraticCurve,
    inner_margin: f32,
) -> bool {
    if y < y_horizon || y > y_bottom {
        return false;
    }
    let left_x = left_curve.eval(y) + inner_margin;
    let right_x = right_curve.eval(y) - inner_margin;
    right_x > left_x && x >= left_x && x <= right_x
}

/// Computes the motorcycle's dynamic projected tire-track corridor `(traj_left_x, traj_right_x)`
/// at normalized vertical coordinate `y`, curving by `steering_curvature` towards the horizon.
pub fn compute_trajectory_corridor(
    y: f32,
    y_horizon: f32,
    y_bottom: f32,
    ego_x: f32,
    steering_curvature: f32,
    base_half_width: f32,
) -> (f32, f32) {
    let span = (y_bottom - y_horizon).max(0.05);
    let progress = ((y - y_horizon) / span).clamp(0.0, 1.0); // 0 at horizon, 1 at bottom
    let dist_from_bike = 1.0 - progress;
    let center_x = ego_x + steering_curvature * dist_from_bike * dist_from_bike;
    let half_w = base_half_width * (0.22 + 0.78 * progress);
    (center_x - half_w, center_x + half_w)
}

/// Checks whether a hazard's horizontal span `[x_min, x_max]` at `y_bottom` intersects
/// the motorcycle's projected trajectory corridor.
pub fn intersects_trajectory_corridor(
    x_min: f32,
    x_max: f32,
    y_bottom: f32,
    y_horizon: f32,
    frame_bottom: f32,
    ego_x: f32,
    steering_curvature: f32,
    base_half_width: f32,
) -> bool {
    if y_bottom < y_horizon || y_bottom > frame_bottom {
        return false;
    }
    let (traj_l, traj_r) = compute_trajectory_corridor(
        y_bottom,
        y_horizon,
        frame_bottom,
        ego_x,
        steering_curvature,
        base_half_width,
    );
    x_max >= traj_l && x_min <= traj_r
}

/// Exact inverse of `estimate_distance_meters`: maps distance in meters (e.g. 5m, 10m, 15m)
/// to normalized image vertical coordinate `y` for backup-camera trajectory distance ticks.
pub fn inverse_perspective_y_for_distance(
    distance_m: f32,
    roi_horizon_norm: f32,
    min_distance_m: f32,
    max_distance_m: f32,
) -> f32 {
    let horizon = roi_horizon_norm.clamp(0.05, 0.90);
    let d = distance_m.clamp(min_distance_m, max_distance_m);
    let curved = (d - min_distance_m) / (max_distance_m - min_distance_m).max(0.01);
    let road_progress = (1.0 - curved).clamp(0.0, 1.0).powf(1.0 / 0.65);
    (horizon + road_progress * (1.0 - horizon)).clamp(horizon, 1.0)
}

/// Computes smoothed road vanishing point `(x_vp, y_vp)` from left/right boundary tangents
/// `x(y) = m * y + c`, and estimates camera banking tilt angle `theta_bank` in radians.
pub fn compute_vanishing_point_and_bank(
    m_left: f32,
    c_left: f32,
    m_right: f32,
    c_right: f32,
    prev_x_vp: f32,
    prev_y_vp: f32,
) -> (f32, f32, f32) {
    let denom = m_left - m_right;
    let (x_meas, y_meas) = if denom.abs() > 1e-4 {
        let y_int = ((c_right - c_left) / denom).clamp(0.48, 0.58);
        let x_int = (0.5 * ((m_left * y_int + c_left) + (m_right * y_int + c_right))).clamp(0.15, 0.85);
        (x_int, y_int)
    } else {
        (prev_x_vp.clamp(0.15, 0.85), prev_y_vp.clamp(0.48, 0.58))
    };

    let x_vp = (0.85 * prev_x_vp + 0.15 * x_meas).clamp(0.15, 0.85);
    let y_vp = (0.85 * prev_y_vp + 0.15 * y_meas).clamp(0.48, 0.58);
    let theta_bank = ((m_left + m_right) * 0.5).atan();
    (x_vp, y_vp, theta_bank)
}

/// Evaluates Phase 8 cyan curved road boundary `x(y)` converging cleanly at `(x_vp, y_vp)`.
pub fn eval_cyan_road_boundary(
    y: f32,
    x_vp: f32,
    y_vp: f32,
    y_bot: f32,
    x_bot: f32,
    kappa_road: f32,
) -> f32 {
    let span = (y_bot - y_vp).max(0.05);
    let t = ((y - y_vp) / span).clamp(0.0, 1.0);
    let curve_offset = kappa_road * t * (std::f32::consts::PI * t).sin();
    x_vp + (x_bot - x_vp) * t.powf(1.3) + curve_offset
}

/// Computes Phase 8 "Green Carpet" left/right edges `(carpet_left, carpet_right)` at row `y`,
/// tapering with exponent 1.4 from `w_near` at `y_bot` to `w_far` at `y_vp` while weaving toward `target_gap_x`.
#[allow(clippy::too_many_arguments)]
pub fn compute_carpet_bounds(
    y: f32,
    _x_vp: f32,
    y_vp: f32,
    y_bot: f32,
    x_ego: f32,
    target_gap_x: f32,
    kappa_steer: f32,
    w_near: f32,
    w_far: f32,
) -> (f32, f32) {
    let span = (y_bot - y_vp).max(0.05);
    let t = ((y - y_vp) / span).clamp(0.0, 1.0);
    let forward = 1.0 - t;
    let width = w_far + (w_near - w_far) * t.powf(1.4);
    let half_w = 0.5 * width;
    let x_path = x_ego + (target_gap_x - x_ego) * forward + kappa_steer * forward * forward;
    (x_path - half_w, x_path + half_w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_1_1_1_estimate_distance_meters_is_monotonic_and_bounded() {
        let dist_near_horizon = estimate_distance_meters(0.42, 0.40, 2.0, 30.0);
        let dist_mid_road = estimate_distance_meters(0.70, 0.40, 2.0, 30.0);
        let dist_bottom_frame = estimate_distance_meters(0.98, 0.40, 2.0, 30.0);
        assert!(dist_bottom_frame < dist_mid_road);
        assert!(dist_mid_road < dist_near_horizon);
        assert!(dist_bottom_frame >= 2.0 && dist_near_horizon <= 30.0);
    }

    #[test]
    fn t_1_1_2_classify_lane_maps_thirds_correctly() {
        assert_eq!(classify_lane(0.10), Lane::Left);
        assert_eq!(classify_lane(0.50), Lane::Center);
        assert_eq!(classify_lane(0.95), Lane::Right);
    }

    #[test]
    fn t_1_1_3_compute_lane_centering_detects_center_and_drift() {
        let center = compute_lane_centering(0.50, 0.0, 1.0, 3.6);
        assert_eq!(center.status, CenteringStatus::Centered);
        assert!((center.offset_meters).abs() < 1e-4);
        assert!((center.dist_left_m - 1.8).abs() < 1e-4);
        assert!((center.dist_right_m - 1.8).abs() < 1e-4);

        let drift_left = compute_lane_centering(0.20, 0.0, 1.0, 3.6);
        assert_eq!(drift_left.status, CenteringStatus::DriftingLeft);
        assert!(drift_left.offset_meters < -0.25);

        let drift_right = compute_lane_centering(0.80, 0.0, 1.0, 3.6);
        assert_eq!(drift_right.status, CenteringStatus::DriftingRight);
        assert!(drift_right.offset_meters > 0.25);
    }

    #[test]
    fn t_5_1_1_pitch_compensated_horizon_shifts_with_braking_and_clamps() {
        let level = compensate_horizon_for_pitch(0.40, 0.0);
        let nose_dive = compensate_horizon_for_pitch(0.40, -10.0); // braking pitch down shifts horizon up
        let wheelie = compensate_horizon_for_pitch(0.40, 10.0);    // acceleration pitch up shifts horizon down
        assert!((level - 0.40).abs() < 1e-5);
        assert!(nose_dive < level);
        assert!(wheelie > level);
        assert!(compensate_horizon_for_pitch(0.40, -200.0) >= 0.15);
        assert!(compensate_horizon_for_pitch(0.40, 200.0) <= 0.85);
    }

    #[test]
    fn t_6_1_1_quadratic_road_boundary_and_drivable_corridor_filter_rejects_offroad_points() {
        // Left boundary curves from x=0.38 at y=0.54 (horizon) down to x=0.12 at y=0.85
        // Right boundary curves from x=0.62 at y=0.54 (horizon) down to x=0.88 at y=0.85
        let left_curve = QuadraticCurve { a: -0.20, b: -0.55, c: 0.735 };
        let right_curve = QuadraticCurve { a: 0.20, b: 0.55, c: 0.265 };

        // On-road point at (0.50, 0.72) is strictly inside drivable polygon
        assert!(is_point_in_drivable_corridor(0.50, 0.72, 0.54, 0.85, &left_curve, &right_curve, 0.02));
        // Off-road parked car on left shoulder (x=0.08, y=0.72) is rejected
        assert!(!is_point_in_drivable_corridor(0.08, 0.72, 0.54, 0.85, &left_curve, &right_curve, 0.02));
        // Off-road fence on right (x=0.93, y=0.72) is rejected
        assert!(!is_point_in_drivable_corridor(0.93, 0.72, 0.54, 0.85, &left_curve, &right_curve, 0.02));
        // Tree/sky above road horizon (y=0.42 < 0.54) is rejected
        assert!(!is_point_in_drivable_corridor(0.50, 0.42, 0.54, 0.85, &left_curve, &right_curve, 0.02));
    }

    #[test]
    fn t_6_1_2_dynamic_trajectory_corridor_and_collision_path_intersection() {
        // Straight trajectory at y=0.75 centered around x=0.50
        let (tl, tr) = compute_trajectory_corridor(0.75, 0.54, 0.85, 0.50, 0.0, 0.11);
        assert!(tl < 0.50 && tr > 0.50);

        // Right-curving trajectory shifts corridor right near horizon
        let (tl_curve, tr_curve) = compute_trajectory_corridor(0.60, 0.54, 0.85, 0.50, 0.25, 0.11);
        let (tl_str, tr_str) = compute_trajectory_corridor(0.60, 0.54, 0.85, 0.50, 0.0, 0.11);
        assert!((tl_curve + tr_curve) > (tl_str + tr_str));

        // Pothole directly in path [0.46, 0.54] at y=0.75 intersects trajectory
        assert!(intersects_trajectory_corridor(0.46, 0.54, 0.75, 0.54, 0.85, 0.50, 0.0, 0.11));
        // Pothole on far left lane [0.22, 0.30] at y=0.75 does NOT intersect trajectory
        assert!(!intersects_trajectory_corridor(0.22, 0.30, 0.75, 0.54, 0.85, 0.50, 0.0, 0.11));

        // Distance tick y-projection is inverse of estimate_distance_meters
        let y_5m = inverse_perspective_y_for_distance(5.0, 0.54, 2.0, 30.0);
        let y_10m = inverse_perspective_y_for_distance(10.0, 0.54, 2.0, 30.0);
        let y_15m = inverse_perspective_y_for_distance(15.0, 0.54, 2.0, 30.0);
        assert!(y_5m > y_10m && y_10m > y_15m && y_15m > 0.54);
    }

    #[test]
    fn t_8_1_1_vanishing_point_tilting_horizon_and_cyan_boundaries_converge() {
        // Left boundary: x(0.54) = 0.50, x(0.88) = 0.16 -> m_left = -1.0, c_left = 1.04
        // Right boundary: x(0.54) = 0.50, x(0.88) = 0.84 -> m_right = 1.0, c_right = -0.04
        let (x_vp, y_vp, theta_bank) =
            compute_vanishing_point_and_bank(-1.0, 1.04, 1.0, -0.04, 0.50, 0.54);
        assert!((x_vp - 0.50).abs() < 1e-4);
        assert!((y_vp - 0.54).abs() < 1e-4);
        assert!(y_vp >= 0.48 && y_vp <= 0.58);
        assert!(theta_bank.abs() < 1e-4);

        // Tilted boundaries & temporal smoothing (0.85 * prev + 0.15 * meas)
        // Intersection at y_meas = 0.50, x_meas = 0.45, banking slope = (-0.8 + 1.2)/2 = 0.2
        let (x_vp_s, y_vp_s, theta_s) =
            compute_vanishing_point_and_bank(-0.8, 0.85, 1.2, -0.15, 0.50, 0.54);
        assert!((x_vp_s - (0.85 * 0.50 + 0.15 * 0.45)).abs() < 1e-3);
        assert!((y_vp_s - (0.85 * 0.54 + 0.15 * 0.50)).abs() < 1e-3);
        assert!((theta_s - 0.2f32.atan()).abs() < 1e-4);

        // Extreme intersection clamps y_vp inside [0.48, 0.58]
        let (_, y_clamped, _) =
            compute_vanishing_point_and_bank(-1.0, 0.70, 1.0, 0.10, 0.50, 0.48);
        assert!(y_clamped >= 0.48 && y_clamped <= 0.58);

        // Cyan curved road boundaries converge at (x_vp, y_vp) and never cross
        let xl_top = eval_cyan_road_boundary(0.54, 0.50, 0.54, 0.88, 0.12, 0.05);
        let xr_top = eval_cyan_road_boundary(0.54, 0.50, 0.54, 0.88, 0.88, 0.05);
        assert!((xl_top - 0.50).abs() < 1e-5);
        assert!((xr_top - 0.50).abs() < 1e-5);

        let xl_bot = eval_cyan_road_boundary(0.88, 0.50, 0.54, 0.88, 0.12, 0.05);
        let xr_bot = eval_cyan_road_boundary(0.88, 0.50, 0.54, 0.88, 0.88, 0.05);
        assert!((xl_bot - 0.12).abs() < 1e-5);
        assert!((xr_bot - 0.88).abs() < 1e-5);

        for kappa in [0.14f32, -0.14f32] {
            for step in 1..=100 {
                let y = 0.54 + (0.88 - 0.54) * (step as f32 / 100.0);
                let xl = eval_cyan_road_boundary(y, 0.50, 0.54, 0.88, 0.12, kappa);
                let xr = eval_cyan_road_boundary(y, 0.50, 0.54, 0.88, 0.88, kappa);
                assert!(xl < xr);
                assert!(xl <= 0.50 + 1e-4, "left curve hooked past x_vp at y={y} kappa={kappa}: xl={xl}");
                assert!(xr >= 0.50 - 1e-4, "right curve hooked past x_vp at y={y} kappa={kappa}: xr={xr}");
            }
        }

        // Green carpet width tapers with exponent 1.4 and weaves toward target_gap_x
        let (cl_bot, cr_bot) =
            compute_carpet_bounds(0.88, 0.50, 0.54, 0.88, 0.50, 0.62, 0.0, 0.18, 0.02);
        assert!(((cr_bot - cl_bot) - 0.18).abs() < 1e-4);
        assert!((((cl_bot + cr_bot) * 0.5) - 0.50).abs() < 1e-4);

        let (cl_top, cr_top) =
            compute_carpet_bounds(0.54, 0.50, 0.54, 0.88, 0.50, 0.62, 0.0, 0.18, 0.02);
        assert!(((cr_top - cl_top) - 0.02).abs() < 1e-4);
        assert!(((cl_top + cr_top) * 0.5) > 0.58);
    }
}
