use crate::nms::RawBoundingBox;
use crate::{process_detections, DetectedPothole};

pub fn parse_yolov8_transposed_output(
    tensor: &[f32],
    num_anchors: usize,
    input_size: f32,
    conf_threshold: f32,
) -> Vec<RawBoundingBox> {
    if num_anchors == 0 || input_size <= 0.0 || tensor.len() < 5 * num_anchors {
        return Vec::new();
    }

    let mut boxes = Vec::new();
    for i in 0..num_anchors {
        let conf = tensor[4 * num_anchors + i];
        if conf < conf_threshold {
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
            boxes.push(RawBoundingBox {
                x_min,
                y_min,
                x_max,
                y_max,
                confidence: conf,
            });
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
    let raw = parse_yolov8_transposed_output(tensor, num_anchors, input_size, conf_threshold);
    process_detections(&raw, conf_threshold, iou_threshold, roi_horizon_norm, imu_z_peak_g)
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
}
