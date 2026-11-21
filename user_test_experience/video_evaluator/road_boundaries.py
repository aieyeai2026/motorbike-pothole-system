"""
Curved road-edge boundaries that converge at a *measured* vanishing point (Phase 2).

Model (all coordinates normalised to the frame, t = y - y_vp):
    left(y)  = x_vp + s_left  * t + k * t^2
    right(y) = x_vp + s_right * t + k * t^2
Both edges share the vanishing point (x_vp, y_vp) and the curvature k, so they converge exactly at the
vanishing point and bend together with the road; they cannot flare out or cross.

Per frame, asphalt (low-saturation grey) is scanned row by row in the band above the ego bike. The road
run containing the tracked road centre gives one (left, right) edge pair per row. An edge is kept only when
what lies beyond it is roadside terrain (grass, dirt shoulder, white line); an edge that touches a vehicle,
the bike or the frame border is an occlusion, not a road edge, and is dropped. The robust fit is then
smoothed over time and, because vehicles only ever make the road look narrower, an edge may move outward
quickly but inward only slowly. Finally each drawn edge is cut where it would run through a vehicle or the bike.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

SCAN_W = 320                # analysis width (px)
SCAN_Y_MIN = 0.10           # topmost row scanned (just under the far horizon)
SCAN_Y_MAX = 0.44           # lowest row scanned: above the ego bike's dash / mirrors (camera-mount assumption)
Y_VP_RANGE = (-0.05, 0.40)  # allowed vanishing-point heights (normalised)
K_LIMIT = 0.8               # |curvature| bound
Y_END_MAX = 0.75            # boundaries are never drawn below this row
NUM_POINTS = 28
K_SMOOTHING = 0.10         # per-frame blend of the measured curvature into the tracked one
K_PRIOR_WEIGHT = 0.05      # ridge weight pulling the curvature toward its previous value (0 = free, 1 = as stiff as the slopes)


def _terrain_masks(hsv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(terrain, white_line): terrain = grass or dirt shoulder; vehicles (saturated red, dark wheels, white panels) are neither."""
    h, s, v = hsv[..., 0].astype(np.int16), hsv[..., 1].astype(np.int16), hsv[..., 2].astype(np.int16)
    grass = (h >= 28) & (h <= 95) & (s >= 55)
    dirt = (h >= 8) & (h <= 30) & (s >= 40) & (s <= 135) & (v >= 70) & (v <= 215)
    return grass | dirt, (v >= 185) & (s <= 70)


def _runs(row: np.ndarray) -> List[Tuple[int, int]]:
    """[(start, end_inclusive)] of True runs in a 1-D bool array."""
    if not row.any():
        return []
    d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
    return list(zip(np.where(d == 1)[0].tolist(), (np.where(d == -1)[0] - 1).tolist()))


def prepare_masks(frame: np.ndarray) -> Dict[str, np.ndarray]:
    """One preprocessing pass at SCAN_W: asphalt (raw + gap-bridged), terrain, white line, and 'unknown' (none of those)."""
    h0, w0 = frame.shape[:2]
    scale = SCAN_W / float(w0)
    small = cv2.resize(frame, (SCAN_W, max(8, int(round(h0 * scale)))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)
    asphalt_raw = (hsv[..., 1] <= 75) & (hsv[..., 2] >= 35) & (hsv[..., 2] <= 200)
    # bridge potholes, lane paint, shadows and tyre marks horizontally so they do not split the road
    asphalt = cv2.morphologyEx(
        asphalt_raw.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    ) > 0
    terrain, white = _terrain_masks(hsv)
    unknown = ~(asphalt_raw | terrain | white)  # vehicles, wheels, the ego bike, deep shadow
    return {"asphalt": asphalt, "terrain": terrain, "white": white, "unknown": unknown}


def scan_road_edges(masks: Dict[str, np.ndarray], x_center_hint: float = 0.5) -> Dict[str, np.ndarray]:
    """Per-row road edges -> {'yl','xl','yr','xr'} (normalised); rows whose edge is occluded/off-frame are omitted."""
    asphalt, terrain, white = masks["asphalt"], masks["terrain"], masks["white"]
    h, w = asphalt.shape

    pad = 7      # width of the terrain check window beyond an edge
    max_line = 5  # a painted edge line may sit between the asphalt and the terrain
    min_w = int(0.03 * w)
    yl, xl, yr, xr = [], [], [], []
    x_c = x_center_hint * w
    for py in range(int(SCAN_Y_MAX * h), int(SCAN_Y_MIN * h) - 1, -1):
        runs = [r for r in _runs(asphalt[py]) if r[1] - r[0] >= min_w]
        if not runs:
            continue
        cont = [r for r in runs if r[0] <= x_c <= r[1]]
        run = cont[0] if cont else min(runs, key=lambda r: min(abs(r[0] - x_c), abs(r[1] - x_c)))
        if not cont and min(abs(run[0] - x_c), abs(run[1] - x_c)) > 0.12 * w:
            continue
        a, b = run
        x_c = 0.6 * x_c + 0.4 * 0.5 * (a + b)
        y_n = py / float(h)
        # an edge counts only when grass/dirt lies just beyond it (after skipping a painted line):
        # not the frame border, a vehicle or the bike
        la = a
        while la > 0 and a - la < max_line and white[py, la - 1]:
            la -= 1
        if la > pad and terrain[py, la - pad:la].mean() >= 0.6:
            yl.append(y_n); xl.append(a / float(w))
        rb = b
        while rb < w - 1 and rb - b < max_line and white[py, rb + 1]:
            rb += 1
        if rb < w - 1 - pad and terrain[py, rb + 1:rb + 1 + pad].mean() >= 0.6:
            yr.append(y_n); xr.append(b / float(w))
    return {k: np.asarray(v, dtype=np.float64) for k, v in (("yl", yl), ("xl", xl), ("yr", yr), ("xr", xr))}


MIN_SIDE_POINTS = 12


def _fit_shared_vp(
    pts: Dict[str, np.ndarray], prior: Optional[Dict[str, float]]
) -> Optional[Dict[str, float]]:
    """
    Robust (Huber-IRLS) fit of x_vp, y_vp, s_left, s_right, k over a grid of y_vp. None if under-determined.
    With both edges observed all five parameters are fitted. If only one edge is observed (the other is hidden by a
    vehicle / the frame border) the hidden edge keeps its previous slope; with no previous fit there is nothing to hold,
    so nothing is fitted rather than hugging the occluder.
    """
    nl, nr = len(pts["yl"]), len(pts["yr"])
    if nl < MIN_SIDE_POINTS and nr < MIN_SIDE_POINTS:
        return None
    if (nl < MIN_SIDE_POINTS or nr < MIN_SIDE_POINTS) and prior is None:
        return None
    hold_left, hold_right = nl < MIN_SIDE_POINTS, nr < MIN_SIDE_POINTS
    if hold_left:
        pts = {**pts, "yl": pts["yl"][:0], "xl": pts["xl"][:0]}; nl = 0
    if hold_right:
        pts = {**pts, "yr": pts["yr"][:0], "xr": pts["xr"][:0]}; nr = 0
    best: Optional[Tuple[float, Dict[str, float]]] = None
    lam = 0.02  # weak pull toward the previous frame's parameters (keeps a one-sided fit sane)
    for y_vp in np.linspace(Y_VP_RANGE[0], Y_VP_RANGE[1], 46):
        tl, tr = pts["yl"] - y_vp, pts["yr"] - y_vp
        A = np.vstack([
            np.stack([np.ones(nl), tl, np.zeros(nl), tl ** 2], axis=1),
            np.stack([np.ones(nr), np.zeros(nr), tr, tr ** 2], axis=1),
        ])
        y = np.concatenate([pts["xl"], pts["xr"]])
        w = np.ones(len(y))
        P = np.array([prior["x_vp"], prior["s_left"], prior["s_right"], prior["k"]]) if prior else np.array([0.5, -1.0, 1.0, 0.0])
        theta = P.copy()
        for _ in range(4):
            Aw = A * w[:, None]
            # the t^2 column is ~10-50x smaller than the others, so it gets a proportionally lighter prior
            reg = lam * np.diag([1.0, 1.0, 1.0, K_PRIOR_WEIGHT]) * len(y)
            theta = np.linalg.solve(Aw.T @ A + reg, Aw.T @ y + reg @ P)
            res = y - A @ theta
            w = 1.0 / np.maximum(1.0, np.abs(res) / 0.012)  # Huber, 1.2% of frame width
        if hold_left:
            theta[1] = prior["s_left"]
        if hold_right:
            theta[2] = prior["s_right"]
        theta[3] = float(np.clip(theta[3], -K_LIMIT, K_LIMIT))
        res = y - A @ theta
        cost = float(np.sum(np.minimum(res ** 2, 0.02 ** 2)))
        x_vp, s_l, s_r, k = (float(v) for v in theta)
        if not (s_l < -0.15 and s_r > 0.15 and 0.05 <= x_vp <= 0.95):  # edges must diverge downward
            continue
        if best is None or cost < best[0]:
            best = (cost, {"x_vp": x_vp, "y_vp": float(y_vp), "s_left": s_l, "s_right": s_r, "k": k})
    return best[1] if best else None


class RoadBoundaryTracker:
    """Stateful per-video estimator; call `update(frame)` for every processed frame, in order."""

    def __init__(self) -> None:
        self.params: Optional[Dict[str, float]] = None

    def update(self, frame: np.ndarray) -> Dict[str, object]:
        hint = 0.5
        if self.params:
            p = self.params
            t = SCAN_Y_MAX - p["y_vp"]
            hint = float(np.clip(p["x_vp"] + 0.5 * (p["s_left"] + p["s_right"]) * t + p["k"] * t * t, 0.2, 0.8))
        masks = prepare_masks(frame)
        pts = scan_road_edges(masks, hint)
        fit = _fit_shared_vp(pts, self.params)
        if fit is not None:
            if self.params is None:
                self.params = fit
            else:
                a = 0.35
                new = {}
                for key in ("x_vp", "y_vp"):
                    new[key] = (1 - a) * self.params[key] + a * fit[key]
                new["k"] = (1 - K_SMOOTHING) * self.params["k"] + K_SMOOTHING * fit["k"]  # curvature is the noisiest term
                # an edge may move outward (road wider) quickly, inward (a vehicle occluding it) only slowly
                new["s_left"] = (1 - (0.6 if fit["s_left"] < self.params["s_left"] else 0.15)) * self.params["s_left"] + (0.6 if fit["s_left"] < self.params["s_left"] else 0.15) * fit["s_left"]
                new["s_right"] = (1 - (0.6 if fit["s_right"] > self.params["s_right"] else 0.15)) * self.params["s_right"] + (0.6 if fit["s_right"] > self.params["s_right"] else 0.15) * fit["s_right"]
                self.params = new
        return self.geometry(masks, len(pts["yl"]), len(pts["yr"]))

    def geometry(self, masks: Dict[str, np.ndarray], n_left: int = 0, n_right: int = 0) -> Dict[str, object]:
        """
        Per-edge polylines `{"left": {"ys", "xs"}, "right": {...}}` from the vanishing point downward. An edge ends
        where it leaves the frame, passes through a vehicle / the ego bike (pixels that are not road, terrain or
        paint), or reaches Y_END_MAX. Off-frame x is left unclipped (it is clipped when drawn).
        """
        empty = {"ok": False, "vp": None, "left": {"ys": [], "xs": []}, "right": {"ys": [], "xs": []}, "n_left": n_left, "n_right": n_right}
        if not self.params:
            return empty
        p = self.params
        h, w = masks["unknown"].shape
        occ = cv2.blur(masks["unknown"].astype(np.float32), (7, 7))
        y0 = max(p["y_vp"], 0.0)  # a vanishing point above the frame is clipped to the top edge
        ys = np.linspace(y0, Y_END_MAX, 200)
        t = ys - p["y_vp"]
        out: Dict[str, object] = {"ok": True, "vp": (float(p["x_vp"]), float(p["y_vp"])), "n_left": n_left, "n_right": n_right}
        for name, slope, leaves in (("left", p["s_left"], lambda x: x < 0.0), ("right", p["s_right"], lambda x: x > 1.0)):
            xs = p["x_vp"] + slope * t + p["k"] * t * t
            end = len(ys) - 1
            run = 0
            for i in range(len(ys)):
                if leaves(xs[i]):
                    end = max(i - 1, 0)
                    break
                if ys[i] >= 0.20:  # far field (trees / sky around the vanishing point) is not tested
                    cx, cy = int(np.clip(xs[i] * w, 0, w - 1)), int(np.clip(ys[i] * h, 0, h - 1))
                    run = run + 1 if occ[cy, cx] >= 0.6 else 0
                    if run >= 3:
                        end = max(i - 3, 0)
                        break
            idx = np.linspace(0, end, NUM_POINTS).astype(int)
            out[name] = {"ys": [float(v) for v in ys[idx]], "xs": [float(v) for v in xs[idx]]}
        return out


def point_on_measured_road(road: Dict[str, object], x: float, y: float, margin: float = 0.02) -> bool:
    """True when the ground point `(x, y)` (normalised) lies between the measured road edges.

    `road` is the dict from `RoadBoundaryTracker.update`. An unknown road (tracker not warmed up) keeps the point.
    A point above the vanishing point is not on the road. An edge that has ended (it left the frame or ran into a
    vehicle / the bike) no longer constrains points below its end, so close potholes are not dropped.
    """
    if not road.get("ok"):
        return True
    left, right = road["left"], road["right"]  # type: ignore[index]
    tops = [e["ys"][0] for e in (left, right) if len(e["ys"])]  # type: ignore[index]
    if tops and y < min(tops) - 0.005:
        return False
    if len(left["ys"]) and y <= left["ys"][-1] and x < float(np.interp(y, left["ys"], left["xs"])) - margin:  # type: ignore[index]
        return False
    if len(right["ys"]) and y <= right["ys"][-1] and x > float(np.interp(y, right["ys"], right["xs"])) + margin:  # type: ignore[index]
        return False
    return True
