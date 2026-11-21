"""Road gate for pothole boxes: a box is kept when it lies between the MEASURED road edges.

Regression: the old gate used a horizon clamped to y in [0.48, 0.56] and a drivable band that ended at
y = 0.75, so every pothole whose bottom edge was above mid-frame (far ahead) or below 75% of the frame was
dropped even though the model had found it. The measured road tracker puts the horizon near y = 0.05.
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from video_evaluator.road_boundaries import point_on_measured_road  # noqa: E402


def _road(left_end=0.50, right_end=0.50, ok=True):
    """Road converging to a vanishing point at (0.49, 0.05); left edge reaches x=0.05, right x=0.95 at y=0.50."""
    def edge(x_bottom, y_end):
        ys = [0.05 + i * (y_end - 0.05) / 9 for i in range(10)]
        return {"ys": ys, "xs": [0.49 + (x_bottom - 0.49) * (y - 0.05) / 0.45 for y in ys]}

    return {"ok": ok, "vp": (0.49, 0.05), "left": edge(0.05, left_end), "right": edge(0.95, right_end)}


def test_unknown_road_keeps_the_box():
    assert point_on_measured_road(_road(ok=False), 0.10, 0.90) is True


def test_far_pothole_above_the_old_mid_frame_horizon_is_kept():
    # The pothole from the bug report: bottom edge at y~0.35 (the old gate demanded y >= 0.48).
    assert point_on_measured_road(_road(), 0.376, 0.35) is True


def test_near_pothole_below_the_edges_end_is_kept():
    assert point_on_measured_road(_road(left_end=0.45, right_end=0.45), 0.50, 0.82) is True


def test_box_left_of_the_left_edge_is_dropped():
    assert point_on_measured_road(_road(), 0.05, 0.30) is False


def test_box_right_of_the_right_edge_is_dropped():
    assert point_on_measured_road(_road(), 0.97, 0.30) is False


def test_box_above_the_vanishing_point_is_dropped():
    assert point_on_measured_road(_road(), 0.49, 0.01) is False


def test_margin_forgives_a_box_just_outside_the_edge():
    # left edge at y=0.30 is x~0.27; 0.26 is inside the default 0.02 margin
    assert point_on_measured_road(_road(), 0.26, 0.30) is True


def test_truncated_edge_leaves_that_side_unconstrained_but_the_other_still_applies():
    road = _road(left_end=0.50, right_end=0.35)
    assert point_on_measured_road(road, 0.97, 0.45) is True   # right edge ended above y=0.45
    assert point_on_measured_road(road, 0.01, 0.45) is False  # left edge still reaches y=0.45
