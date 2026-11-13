pub mod geometry;
pub mod nms;
pub mod preprocess;

use geometry::{classify_lane, estimate_distance_meters, Lane};
use nms::{non_max_suppression, RawBoundingBox};
use preprocess::compute_severity_score;

#[derive(Debug, Clone, PartialEq)]
pub struct DetectedPothole {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
    pub confidence: f32,
    pub distance_meters: f32,
    pub lane: Lane,
    pub severity_score: f32,
}

pub fn process_detections(
    candidates: &[RawBoundingBox],
    conf_threshold: f32,
    iou_threshold: f32,
    roi_horizon_norm: f32,
    imu_z_peak_g: Option<f32>,
) -> Vec<DetectedPothole> {
    non_max_suppression(candidates, conf_threshold, iou_threshold)
        .into_iter()
        .map(|b| {
            let distance = estimate_distance_meters(b.y_bottom(), roi_horizon_norm, 2.0, 30.0);
            let severity = compute_severity_score(b.area(), distance, imu_z_peak_g);
            DetectedPothole {
                x_min: b.x_min,
                y_min: b.y_min,
                x_max: b.x_max,
                y_max: b.y_max,
                confidence: b.confidence,
                distance_meters: distance,
                lane: classify_lane(b.x_center()),
                severity_score: severity,
            }
        })
        .collect()
}
