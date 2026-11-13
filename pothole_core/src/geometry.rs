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
}
