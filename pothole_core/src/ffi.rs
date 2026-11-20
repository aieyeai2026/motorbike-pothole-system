use crate::geometry::{
    compute_lane_centering, intersects_trajectory_corridor, is_point_in_drivable_corridor,
    CenteringStatus, Lane, QuadraticCurve,
};
use crate::inference::run_yolo_postprocess;
use std::slice;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CDetectedPothole {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
    pub confidence: f32,
    pub distance_meters: f32,
    pub severity_score: f32,
    pub lane_code: i32,
    pub class_id: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CLaneCentering {
    pub offset_meters: f32,
    pub dist_left_m: f32,
    pub dist_right_m: f32,
    pub status_code: i32,
}

#[no_mangle]
pub unsafe extern "C" fn pothole_core_compute_lane_centering(
    x_ego: f32,
    x_left_border: f32,
    x_right_border: f32,
    road_width_m: f32,
    out_ptr: *mut CLaneCentering,
) -> i32 {
    if out_ptr.is_null() {
        return -1;
    }
    let res = compute_lane_centering(x_ego, x_left_border, x_right_border, road_width_m);
    let status_code = match res.status {
        CenteringStatus::Centered => 0,
        CenteringStatus::DriftingLeft => 1,
        CenteringStatus::DriftingRight => 2,
    };
    *out_ptr = CLaneCentering {
        offset_meters: res.offset_meters,
        dist_left_m: res.dist_left_m,
        dist_right_m: res.dist_right_m,
        status_code,
    };
    0
}

/// Evaluates whether a hazard box with bottom span `[x_min, x_max]` at `y_bottom` is:
/// - `0`: Off-road (outside curved drivable road polygon `[left_curve(y), right_curve(y)]`)
/// - `1`: On drivable road surface, but outside the motorcycle's trajectory corridor
/// - `2`: On drivable road surface AND intersecting the motorcycle's trajectory corridor
#[no_mangle]
pub unsafe extern "C" fn pothole_core_classify_hazard_corridor(
    x_min: f32,
    x_max: f32,
    y_bottom: f32,
    y_horizon: f32,
    frame_bottom: f32,
    left_poly_ptr: *const f32,
    right_poly_ptr: *const f32,
    ego_x: f32,
    steering_curvature: f32,
    base_half_width: f32,
) -> i32 {
    if left_poly_ptr.is_null() || right_poly_ptr.is_null() {
        return 0;
    }
    let l = slice::from_raw_parts(left_poly_ptr, 3);
    let r = slice::from_raw_parts(right_poly_ptr, 3);
    let left_curve = QuadraticCurve { a: l[0], b: l[1], c: l[2] };
    let right_curve = QuadraticCurve { a: r[0], b: r[1], c: r[2] };

    let x_center = (x_min + x_max) * 0.5;
    if !is_point_in_drivable_corridor(
        x_center,
        y_bottom,
        y_horizon,
        frame_bottom,
        &left_curve,
        &right_curve,
        0.02,
    ) {
        return 0;
    }
    if intersects_trajectory_corridor(
        x_min,
        x_max,
        y_bottom,
        y_horizon,
        frame_bottom,
        ego_x,
        steering_curvature,
        base_half_width,
    ) {
        2
    } else {
        1
    }
}

#[no_mangle]
pub unsafe extern "C" fn pothole_core_run_postprocess(
    tensor_ptr: *const f32,
    tensor_len: usize,
    num_anchors: usize,
    input_size: f32,
    conf_threshold: f32,
    iou_threshold: f32,
    roi_horizon_norm: f32,
    imu_z_peak_g: f32,
    out_buffer_ptr: *mut CDetectedPothole,
    max_out: usize,
) -> usize {
    if tensor_ptr.is_null() || out_buffer_ptr.is_null() || max_out == 0 {
        return 0;
    }

    let tensor = slice::from_raw_parts(tensor_ptr, tensor_len);
    let imu_opt = if imu_z_peak_g > 0.0 { Some(imu_z_peak_g) } else { None };

    let detections = run_yolo_postprocess(
        tensor,
        num_anchors,
        input_size,
        conf_threshold,
        iou_threshold,
        roi_horizon_norm,
        imu_opt,
    );

    let out_slice = slice::from_raw_parts_mut(out_buffer_ptr, max_out);
    let count = detections.len().min(max_out);

    for i in 0..count {
        let d = &detections[i];
        let lane_code = match d.lane {
            Lane::Left => 0,
            Lane::Center => 1,
            Lane::Right => 2,
        };
        out_slice[i] = CDetectedPothole {
            x_min: d.x_min,
            y_min: d.y_min,
            x_max: d.x_max,
            y_max: d.y_max,
            confidence: d.confidence,
            distance_meters: d.distance_meters,
            severity_score: d.severity_score,
            lane_code,
            class_id: d.class_id,
        };
    }

    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_2_1_1_c_ffi_postprocess_writes_detections_to_out_buffer() {
        let num_anchors = 2;
        let tensor: Vec<f32> = vec![
            320.0, 100.0,
            480.0, 200.0,
            128.0, 40.0,
            64.0,  40.0,
            0.91,  0.15,
        ];
        let mut out = vec![
            CDetectedPothole {
                x_min: 0.0, y_min: 0.0, x_max: 0.0, y_max: 0.0,
                confidence: 0.0, distance_meters: 0.0, severity_score: 0.0, lane_code: -1, class_id: -1,
            };
            5
        ];

        let count = unsafe {
            pothole_core_run_postprocess(
                tensor.as_ptr(),
                tensor.len(),
                num_anchors,
                640.0,
                0.50,
                0.45,
                0.40,
                1.8,
                out.as_mut_ptr(),
                out.len(),
            )
        };

        assert_eq!(count, 1);
        assert_eq!(out[0].lane_code, 1);
        assert!((out[0].confidence - 0.91).abs() < 1e-5);
        assert!(out[0].distance_meters >= 2.0);
        assert!(out[0].severity_score > 0.0);
    }

    #[test]
    fn t_4_1_2_c_ffi_exports_class_id_and_lane_centering() {
        let mut centering = CLaneCentering {
            offset_meters: 0.0,
            dist_left_m: 0.0,
            dist_right_m: 0.0,
            status_code: -1,
        };
        let rc = unsafe {
            pothole_core_compute_lane_centering(0.20, 0.0, 1.0, 3.6, &mut centering as *mut CLaneCentering)
        };
        assert_eq!(rc, 0);
        assert_eq!(centering.status_code, 1); // 1 = DRIFTING_LEFT
        assert!(centering.offset_meters < -0.25);
        assert!((centering.dist_left_m - 0.72).abs() < 1e-3);
    }

    #[test]
    fn t_6_1_3_c_ffi_evaluates_curved_corridor_and_trajectory_intersection() {
        // Check FFI drivable + trajectory intersection evaluation
        // code 2 = inside drivable road AND inside motorcycle trajectory path
        // code 1 = inside drivable road, outside trajectory path
        // code 0 = off-road (outside drivable polygon)
        let left_coeffs: [f32; 3] = [0.0, -0.80, 0.82];
        let right_coeffs: [f32; 3] = [0.0, 0.80, 0.18];

        let in_path = unsafe {
            pothole_core_classify_hazard_corridor(
                0.47, 0.53, 0.75, 0.54, 0.85,
                left_coeffs.as_ptr(), right_coeffs.as_ptr(),
                0.50, 0.0, 0.11,
            )
        };
        let on_road_out_of_path = unsafe {
            pothole_core_classify_hazard_corridor(
                0.26, 0.32, 0.75, 0.54, 0.85,
                left_coeffs.as_ptr(), right_coeffs.as_ptr(),
                0.50, 0.0, 0.11,
            )
        };
        let off_road = unsafe {
            pothole_core_classify_hazard_corridor(
                0.04, 0.10, 0.75, 0.54, 0.85,
                left_coeffs.as_ptr(), right_coeffs.as_ptr(),
                0.50, 0.0, 0.11,
            )
        };

        assert_eq!(in_path, 2);
        assert_eq!(on_road_out_of_path, 1);
        assert_eq!(off_road, 0);
    }
}
