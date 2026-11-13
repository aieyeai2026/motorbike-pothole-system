#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawBoundingBox {
    pub x_min: f32, pub y_min: f32, pub x_max: f32, pub y_max: f32, pub confidence: f32,
}

impl RawBoundingBox {
    pub fn area(&self) -> f32 { (self.x_max - self.x_min).max(0.0) * (self.y_max - self.y_min).max(0.0) }
    pub fn x_center(&self) -> f32 { (self.x_min + self.x_max) * 0.5 }
    pub fn y_bottom(&self) -> f32 { self.y_max }
}

pub fn compute_iou(a: &RawBoundingBox, b: &RawBoundingBox) -> f32 {
    let inter_w = (a.x_max.min(b.x_max) - a.x_min.max(b.x_min)).max(0.0);
    let inter_h = (a.y_max.min(b.y_max) - a.y_min.max(b.y_min)).max(0.0);
    let inter_area = inter_w * inter_h;
    if inter_area <= 0.0 { return 0.0; }
    let union_area = a.area() + b.area() - inter_area;
    if union_area <= 0.0 { 0.0 } else { inter_area / union_area }
}

pub fn non_max_suppression(candidates: &[RawBoundingBox], conf_threshold: f32, iou_threshold: f32) -> Vec<RawBoundingBox> {
    let mut filtered: Vec<RawBoundingBox> = candidates.iter().copied()
        .filter(|b| b.confidence >= conf_threshold && b.area() > 0.0)
        .collect();
    filtered.sort_unstable_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept: Vec<RawBoundingBox> = Vec::with_capacity(filtered.len());
    'outer: for candidate in filtered {
        for existing in &kept {
            if compute_iou(&candidate, existing) > iou_threshold { continue 'outer; }
        }
        kept.push(candidate);
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_1_1_3_compute_iou_handles_identical_disjoint_and_partial_boxes() {
        let box_a = RawBoundingBox { x_min: 0.2, y_min: 0.5, x_max: 0.4, y_max: 0.7, confidence: 0.9 };
        let box_disjoint = RawBoundingBox { x_min: 0.6, y_min: 0.6, x_max: 0.8, y_max: 0.8, confidence: 0.8 };
        let box_partial = RawBoundingBox { x_min: 0.3, y_min: 0.5, x_max: 0.5, y_max: 0.7, confidence: 0.75 };
        assert!((compute_iou(&box_a, &box_a) - 1.0).abs() < 1e-5);
        assert_eq!(compute_iou(&box_a, &box_disjoint), 0.0);
        assert!((compute_iou(&box_a, &box_partial) - (1.0 / 3.0)).abs() < 1e-5);
    }

    #[test]
    fn t_1_1_4_non_max_suppression_filters_low_conf_and_merges_overlaps() {
        let candidates = vec![
            RawBoundingBox { x_min: 0.40, y_min: 0.60, x_max: 0.55, y_max: 0.75, confidence: 0.92 },
            RawBoundingBox { x_min: 0.41, y_min: 0.61, x_max: 0.56, y_max: 0.76, confidence: 0.78 },
            RawBoundingBox { x_min: 0.75, y_min: 0.65, x_max: 0.88, y_max: 0.80, confidence: 0.81 },
            RawBoundingBox { x_min: 0.10, y_min: 0.50, x_max: 0.20, y_max: 0.60, confidence: 0.25 },
        ];
        let kept = non_max_suppression(&candidates, 0.50, 0.45);
        assert_eq!(kept.len(), 2);
        assert!((kept[0].confidence - 0.92).abs() < 1e-5);
        assert!((kept[1].confidence - 0.81).abs() < 1e-5);
    }
}
