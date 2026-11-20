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
}
