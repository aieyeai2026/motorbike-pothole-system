use pothole_core::geometry::{classify_lane, estimate_distance_meters};
use pothole_core::inference::parse_yolov8_transposed_output;
use pothole_core::nms::non_max_suppression;
use pothole_core::preprocess::{compute_severity_score, crop_and_normalize_roi};
use std::time::Instant;

#[test]
fn bench_core_pipeline_micro_timings() {
    let num_anchors = 8400; // full 8400 anchors for 640x640 YOLOv8
    let mut tensor = vec![0.0f32; num_anchors * 5];
    for i in 0..num_anchors {
        tensor[i] = 320.0 + (i % 60) as f32; // cx
        tensor[num_anchors + i] = 420.0 + (i % 80) as f32; // cy
        tensor[2 * num_anchors + i] = 60.0 + (i % 20) as f32; // w
        tensor[3 * num_anchors + i] = 40.0 + (i % 15) as f32; // h
        tensor[4 * num_anchors + i] = if i % 150 == 0 { 0.85 } else { 0.05 }; // conf
    }

    // 1. Measure Tensor Parse Time
    let start_parse = Instant::now();
    let raw_boxes = parse_yolov8_transposed_output(&tensor, num_anchors, 640.0, 0.50);
    let parse_duration = start_parse.elapsed();

    // 2. Measure NMS Time
    let start_nms = Instant::now();
    let nms_boxes = non_max_suppression(&raw_boxes, 0.50, 0.45);
    let nms_duration = start_nms.elapsed();

    // 3. Measure Distance and Lane Math Time
    let start_math = Instant::now();
    for b in &nms_boxes {
        let _d = estimate_distance_meters(b.y_max, 0.40, 2.0, 35.0);
        let _l = classify_lane((b.x_min + b.x_max) * 0.5);
    }
    let math_duration = start_math.elapsed();

    // 4. Measure Preprocess (ROI crop + float normalization on 640x480 frame)
    let fake_frame = vec![128u8; 640 * 480 * 3];
    let start_prep = Instant::now();
    let _roi = crop_and_normalize_roi(&fake_frame, 640, 480, 0.40);
    let prep_duration = start_prep.elapsed();

    // 5. Measure Severity Scoring
    let start_sev = Instant::now();
    for b in &nms_boxes {
        let area = (b.x_max - b.x_min) * (b.y_max - b.y_min);
        let _s = compute_severity_score(area, 10.0, Some(1.6));
    }
    let sev_duration = start_sev.elapsed();

    println!(
        "RUST_MICRO_BENCH_JSON:{{\"parse_us\":{},\"nms_us\":{},\"math_us\":{},\"preprocess_us\":{},\"severity_us\":{}}}",
        parse_duration.as_micros(),
        nms_duration.as_micros(),
        math_duration.as_micros(),
        prep_duration.as_micros(),
        sev_duration.as_micros()
    );
}
