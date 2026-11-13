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
}
