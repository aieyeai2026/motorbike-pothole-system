pub mod geometry;
pub mod nms;

use geometry::{classify_lane, estimate_distance_meters, Lane};
use nms::{non_max_suppression, RawBoundingBox};

#[derive(Debug, Clone, PartialEq)]
pub struct DetectedPothole {
    pub x_min: f32, pub y_min: f32, pub x_max: f32, pub y_max: f32,
    pub confidence: f32, pub distance_meters: f32, pub lane: Lane,
}

pub fn process_detections(
    candidates: &[RawBoundingBox],
    conf_threshold: f32,
    iou_threshold: f32,
    roi_horizon_norm: f32,
) -> Vec<DetectedPothole> {
    non_max_suppression(candidates, conf_threshold, iou_threshold)
        .into_iter()
        .map(|b| DetectedPothole {
            x_min: b.x_min, y_min: b.y_min, x_max: b.x_max, y_max: b.y_max,
            confidence: b.confidence,
            distance_meters: estimate_distance_meters(b.y_bottom(), roi_horizon_norm, 2.0, 30.0),
            lane: classify_lane(b.x_center()),
        })
        .collect()
}
