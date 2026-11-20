use crate::nms::RawBoundingBox;
use crate::{process_class_detections, DetectedPothole};

pub fn parse_yolov8_transposed_output(
    tensor: &[f32],
    num_anchors: usize,
    input_size: f32,
    conf_threshold: f32,
) -> Vec<RawBoundingBox> {
    parse_yolov8_multiclass_output(tensor, num_anchors, input_size, conf_threshold)
        .into_iter()
        .map(|(_, b)| b)
        .collect()
}

pub fn parse_yolov8_multiclass_output(
    tensor: &[f32],
    num_anchors: usize,
    input_size: f32,
    conf_threshold: f32,
) -> Vec<(i32, RawBoundingBox)> {
    if num_anchors == 0 || input_size <= 0.0 || tensor.len() < 5 * num_anchors {
        return Vec::new();
    }
    let num_channels = tensor.len() / num_anchors;
    let num_classes = num_channels.saturating_sub(4).max(1);

    let mut boxes = Vec::new();
    for i in 0..num_anchors {
        let mut best_class: i32 = 0;
        let mut best_conf: f32 = -1.0;
        for c in 0..num_classes {
            let score = tensor[(4 + c) * num_anchors + i];
            if score > best_conf {
                best_conf = score;
                best_class = c as i32;
            }
        }
        if best_conf < conf_threshold {
            continue;
        }

        let cx = tensor[i] / input_size;
        let cy = tensor[num_anchors + i] / input_size;
        let w = tensor[2 * num_anchors + i] / input_size;
        let h = tensor[3 * num_anchors + i] / input_size;

        let x_min = (cx - w * 0.5).clamp(0.0, 1.0);
        let y_min = (cy - h * 0.5).clamp(0.0, 1.0);
        let x_max = (cx + w * 0.5).clamp(0.0, 1.0);
        let y_max = (cy + h * 0.5).clamp(0.0, 1.0);

        if x_max > x_min && y_max > y_min {
            boxes.push((
                best_class,
                RawBoundingBox {
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                    confidence: best_conf,
                },
            ));
        }
    }

    boxes
}

pub fn run_yolo_postprocess(
    tensor: &[f32],
    num_anchors: usize,
    input_size: f32,
    conf_threshold: f32,
    iou_threshold: f32,
    roi_horizon_norm: f32,
    imu_z_peak_g: Option<f32>,
) -> Vec<DetectedPothole> {
    let raw = parse_yolov8_multiclass_output(tensor, num_anchors, input_size, conf_threshold);
    if raw.is_empty() {
        return Vec::new();
    }
    let max_class = raw.iter().map(|(c, _)| *c).max().unwrap_or(0);
    let mut results = Vec::new();
    for c in 0..=max_class {
        let class_boxes: Vec<RawBoundingBox> = raw
            .iter()
            .filter(|(cid, _)| *cid == c)
            .map(|(_, b)| *b)
            .collect();
        if !class_boxes.is_empty() {
            results.extend(process_class_detections(
                &class_boxes,
                c,
                conf_threshold,
                iou_threshold,
                roi_horizon_norm,
                imu_z_peak_g,
            ));
        }
    }
    results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Lane;

    #[test]
    fn t_1_3_1_parse_yolov8_transposed_output_decodes_and_runs_nms() {
        let num_anchors = 3;
        let tensor: Vec<f32> = vec![
            320.0, 324.0, 100.0, // cx
            480.0, 482.0, 200.0, // cy
            128.0, 130.0, 40.0,  // w
            64.0,  66.0,  40.0,  // h
            0.88,  0.72,  0.12,  // conf
        ];

        let hazards = run_yolo_postprocess(&tensor, num_anchors, 640.0, 0.50, 0.45, 0.40, Some(1.5));
        assert_eq!(hazards.len(), 1);
        assert_eq!(hazards[0].lane, Lane::Center);
        assert!((hazards[0].confidence - 0.88).abs() < 1e-5);
        assert!(hazards[0].distance_meters >= 2.0 && hazards[0].distance_meters <= 30.0);
    }

    #[test]
    fn t_4_1_1_multiclass_yolo_parses_pothole_car_debris_and_per_class_nms() {
        // [1, 7, 3] tensor: 4 box coords + 3 classes (0: pothole, 1: car, 2: debris)
        // Anchor 0 & Anchor 1 overlap at (320, 480), but Anchor 0 is pothole (0.91) and Anchor 1 is car (0.85).
        // Anchor 2 is debris at (150, 500) with conf 0.80.
        // Per-class NMS must keep BOTH the overlapping pothole and car, plus the debris (3 total).
        let num_anchors = 3;
        let tensor: Vec<f32> = vec![
            320.0, 322.0, 150.0, // cx
            480.0, 481.0, 500.0, // cy
            120.0, 120.0,  60.0, // w
             60.0,  60.0,  40.0, // h
             0.91,  0.10,  0.05, // class 0: pothole
             0.12,  0.85,  0.10, // class 1: car
             0.05,  0.08,  0.80, // class 2: debris
        ];

        let hazards = run_yolo_postprocess(&tensor, num_anchors, 640.0, 0.50, 0.45, 0.40, Some(1.4));
        assert_eq!(hazards.len(), 3);
        assert_eq!(hazards[0].class_id, 0); // pothole
        assert_eq!(hazards[1].class_id, 1); // car
        assert_eq!(hazards[2].class_id, 2); // debris
    }
}
