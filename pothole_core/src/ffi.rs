use crate::geometry::Lane;
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
                confidence: 0.0, distance_meters: 0.0, severity_score: 0.0, lane_code: -1,
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
}
