/// Crops the upper sky region above `roi_top_ratio` (e.g. 0.40) from an RGB byte buffer
/// (`width * height * 3` bytes) and normalizes pixel values `[0..255]` into `[0.0..1.0]` floats.
pub fn crop_and_normalize_roi(
    rgb_bytes: &[u8],
    width: usize,
    height: usize,
    roi_top_ratio: f32,
) -> Vec<f32> {
    let expected_len = width.saturating_mul(height).saturating_mul(3);
    if rgb_bytes.len() < expected_len || width == 0 || height == 0 {
        return Vec::new();
    }

    let clamped_ratio = roi_top_ratio.clamp(0.0, 0.95);
    let start_row = ((height as f32) * clamped_ratio).floor() as usize;
    let roi_rows = height.saturating_sub(start_row);
    let start_byte = start_row * width * 3;

    let mut normalized = Vec::with_capacity(roi_rows * width * 3);
    for &byte in &rgb_bytes[start_byte..expected_len] {
        normalized.push((byte as f32) / 255.0);
    }
    normalized
}

/// Combines normalized bounding box area (`[0.0, 1.0]`), distance in meters,
/// and optional IMU Z-axis peak acceleration (in G-force, where `1.0` is smooth flat road)
/// into a bounded `[0.0, 10.0]` severity score.
pub fn compute_severity_score(
    box_area_norm: f32,
    distance_meters: f32,
    imu_z_peak_g: Option<f32>,
) -> f32 {
    let area_component = (box_area_norm.clamp(0.0, 0.25) / 0.25) * 5.0;
    let proximity_factor = (1.0 - (distance_meters.clamp(2.0, 30.0) - 2.0) / 28.0) * 3.0;
    let imu_spike = imu_z_peak_g
        .map(|g| ((g - 1.0).max(0.0)).min(2.0))
        .unwrap_or(0.0);

    (area_component + proximity_factor + imu_spike).clamp(0.0, 10.0)
}

/// Filters out steady-state high-frequency engine harmonic vibration from an accelerometer
/// window and returns the true suspension impact peak in G-force (`>= 1.0`).
pub fn filter_imu_engine_vibration(samples_g: &[f32]) -> f32 {
    if samples_g.is_empty() {
        return 1.0;
    }
    let mut sorted: Vec<f32> = samples_g.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];
    let peak = sorted.last().copied().unwrap_or(1.0);
    (1.0 + (peak - median).max(0.0)).max(1.0)
}

/// Rejects false-positive pothole triggers identified in field footage (`VIDEO_ANALYSIS_PHASE5.md`):
/// - Painted crosswalk stripes / elongated markings (`w / h > 5.5`).
/// - Low-confidence canopy shadows (`conf < 0.60`) with flat IMU (`<= 1.05G`).
pub fn is_likely_false_positive_pothole(
    box_w_norm: f32,
    box_h_norm: f32,
    imu_z_peak_g: Option<f32>,
    confidence: f32,
) -> bool {
    let aspect = box_w_norm / box_h_norm.max(1e-4);
    if aspect > 5.5 {
        return true;
    }
    let imu_g = imu_z_peak_g.unwrap_or(1.0);
    if confidence < 0.60 && imu_g <= 1.05 {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_1_2_1_crop_and_normalize_roi_strips_sky_and_normalizes_floats() {
        // 2x10 RGB image (20 pixels = 60 bytes): top 4 rows = 255 (sky), bottom 6 rows = 128 (road)
        let width = 2;
        let height = 10;
        let mut frame = vec![255u8; 4 * width * 3];
        frame.extend(vec![128u8; 6 * width * 3]);

        let tensor = crop_and_normalize_roi(&frame, width, height, 0.40);
        // Only bottom 6 rows remain: 6 * 2 * 3 = 36 floats
        assert_eq!(tensor.len(), 36);
        assert!((tensor[0] - (128.0 / 255.0)).abs() < 1e-5);
        assert!(tensor.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn t_1_2_2_compute_severity_score_is_bounded_and_increases_with_imu_spike() {
        let visual_only = compute_severity_score(0.05, 10.0, None);
        let with_imu_impact = compute_severity_score(0.05, 10.0, Some(2.4));
        let extreme_max = compute_severity_score(0.50, 2.0, Some(5.0));

        assert!(visual_only >= 0.0 && visual_only <= 10.0);
        assert!(with_imu_impact > visual_only);
        assert_eq!(extreme_max, 10.0);
    }

    #[test]
    fn t_5_1_2_imu_engine_vibration_filter_and_fp_suppression() {
        // Steady high-frequency engine buzz oscillating around 1.35G produces minimal transient impulse (~1.0G)
        let engine_buzz = vec![1.34, 1.36, 1.35, 1.34, 1.36, 1.35];
        let filtered_buzz = filter_imu_engine_vibration(&engine_buzz);
        assert!(filtered_buzz < 1.15);

        // Real pothole suspension spike (2.4G) on top of 1.1G baseline passes through
        let pothole_hit = vec![1.10, 1.12, 1.09, 2.45, 1.15, 1.10];
        let filtered_hit = filter_imu_engine_vibration(&pothole_hit);
        assert!(filtered_hit > 1.80);

        // False-positive suppression: zebra crosswalk stripe (wide aspect ratio w/h > 5.5) is rejected
        assert!(is_likely_false_positive_pothole(0.60, 0.08, Some(1.02), 0.75));
        // Low-confidence canopy shadow (<0.60) with flat IMU (<=1.05G) is rejected
        assert!(is_likely_false_positive_pothole(0.20, 0.15, Some(1.01), 0.54));
        // True pothole with normal aspect ratio and high confidence is kept
        assert!(!is_likely_false_positive_pothole(0.22, 0.16, Some(1.75), 0.89));
    }
}
