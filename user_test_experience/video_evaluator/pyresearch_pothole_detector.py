"""
PyResearch YOLOv8 neural pothole detector (Phase 1).

Integrates trained weights directly from:
  https://github.com/pyresearch/Pothole-Computer-Vision-Project (best.pt, class {0: 'Pothole'})
with no morphological / Gaussian-background / gradient heuristics.

Inference is plain full-frame YOLO on a 1280x736 letterbox of the camera frame. A detection is
kept only when the model's own raw confidence is >= the threshold (default 0.45, never lower).
There is no confidence remapping: the number shown in the UI is the raw YOLO score.
"""

import copy
import io
import math
import os
import pickle
import types
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS_PATH = os.path.join(EVAL_DIR, "models", "pyresearch_pothole_best.pt")
PYRESEARCH_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/pyresearch/Pothole-Computer-Vision-Project/main/best.pt"
)
REPO_ROOT = os.path.abspath(os.path.join(EVAL_DIR, "..", ".."))
DEFAULT_CONF_THRESHOLD: float = 0.45  # raw YOLO confidence gate (no remapping); callers may only raise it
INFER_W, INFER_H = 1280, 736  # full-frame letterbox, multiple of 32
MAX_DETECTIONS = 5
NMS_IOU = 0.45
MAX_BOX_W_NORM, MAX_BOX_H_NORM = 0.35, 0.30  # a pothole box larger than this is not a pothole


# ── Pure-PyTorch YOLOv12s layers compatible with PyResearch best.pt ──────────
class _Conv(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if hasattr(self, "bn"):
            out = self.bn(out)
        return self.act(out) if hasattr(self, "act") else out


class _DWConv(_Conv):
    pass


class _Concat(nn.Module):
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, self.d)


class _Bottleneck(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if getattr(self, "add", False) else y


class _C3k(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class _C3k2(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class _AAttn(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)
        area = getattr(self, "area", 1)
        if area > 1 and n % area == 0:
            qk = qk.reshape(b * area, n // area, c * 2)
            v = v.reshape(b * area, n // area, c)
            b_eff, n_eff, _ = qk.shape
        else:
            b_eff, n_eff = b, n
        q, k = qk.split([c, c], dim=-1)
        nh, hd = self.num_heads, self.head_dim
        q = q.view(b_eff, n_eff, nh, hd).permute(0, 2, 1, 3)
        k = k.view(b_eff, n_eff, nh, hd).permute(0, 2, 1, 3)
        v = v.view(b_eff, n_eff, nh, hd).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 2, 1, 3).reshape(b, h, w, c).permute(0, 3, 1, 2)
        return self.proj(attn + pp)


class _ABlock(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.mlp(x)


class _A2C2f(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if getattr(self, "gamma", None) is not None:
            return x + self.gamma.view(1, -1, 1, 1) * y
        return y


class _DFL(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class _Detect(nn.Module):
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        shape = x[0].shape
        xi = [torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) for i in range(self.nl)]
        box, cls = torch.cat([v.view(shape[0], self.no, -1) for v in xi], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        dfl_out = self.dfl(box)
        anchors, strides = [], []
        for i in range(self.nl):
            _, _, h, w = x[i].shape
            sx = torch.arange(w, dtype=dfl_out.dtype, device=dfl_out.device) + 0.5
            sy = torch.arange(h, dtype=dfl_out.dtype, device=dfl_out.device) + 0.5
            gy, gx = torch.meshgrid(sy, sx, indexing="ij")
            anchors.append(torch.stack((gx, gy), -1).reshape(-1, 2))
            strides.append(
                torch.full((h * w, 1), self.stride[i], dtype=dfl_out.dtype, device=dfl_out.device)
            )
        anc = torch.cat(anchors, 0).transpose(0, 1).unsqueeze(0)
        strd = torch.cat(strides, 0).transpose(0, 1)
        lt, rb = dfl_out.chunk(2, 1)
        x1y1 = anc - lt
        x2y2 = anc + rb
        c_xy = (x1y1 + x2y2) * 0.5 * strd
        wh = (x2y2 - x1y1) * strd
        return torch.cat((c_xy, wh, cls.sigmoid()), 1)


class _DetectionModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: List[Optional[torch.Tensor]] = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x


_CLASS_MAP = {
    "Conv": _Conv,
    "DWConv": _DWConv,
    "Concat": _Concat,
    "Bottleneck": _Bottleneck,
    "C3k": _C3k,
    "C3k2": _C3k2,
    "AAttn": _AAttn,
    "ABlock": _ABlock,
    "A2C2f": _A2C2f,
    "DFL": _DFL,
    "Detect": _Detect,
    "DetectionModel": _DetectionModel,
}


class _YoloUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("ultralytics") and name in _CLASS_MAP:
            return _CLASS_MAP[name]
        return super().find_class(module, name)


_CUSTOM_PICKLE = types.SimpleNamespace(
    __name__="pyresearch_yolo_pickle",
    Unpickler=_YoloUnpickler,
    load=lambda f, **kw: _YoloUnpickler(f).load(),
    loads=lambda b, **kw: _YoloUnpickler(io.BytesIO(b)).load(),
)

_MODEL_CACHE: Dict[Tuple[str, str], nn.Module] = {}


def load_pyresearch_yolo_model(weights_path: Optional[str] = None, device: str = "cpu") -> nn.Module:
    """
    Loads the PyResearch YOLOv12s pothole model (`best.pt`, class `{0: 'Pothole'}`).
    Automatically downloads from PyResearch's GitHub repository if not cached locally.
    """
    target_path = os.path.abspath(weights_path or DEFAULT_WEIGHTS_PATH)
    dev_key = device if (device == "cpu" or torch.backends.mps.is_available()) else "cpu"
    if (target_path, dev_key) in _MODEL_CACHE:
        return _MODEL_CACHE[(target_path, dev_key)]

    if ("cpu" != dev_key) and ((target_path, "cpu") in _MODEL_CACHE):
        dev_model = copy.deepcopy(_MODEL_CACHE[(target_path, "cpu")]).to(dev_key).eval()
        _MODEL_CACHE[(target_path, dev_key)] = dev_model
        return dev_model

    if not os.path.isfile(target_path):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        raw = urllib.request.urlopen(PYRESEARCH_WEIGHTS_URL, timeout=20).read()
        with open(target_path, "wb") as f:
            f.write(raw)

    ckpt = torch.load(
        target_path,
        map_location="cpu",
        pickle_module=_CUSTOM_PICKLE,
        weights_only=False,
    )
    cpu_model = ckpt["model"].float().eval()
    _MODEL_CACHE[(target_path, "cpu")] = cpu_model
    if dev_key == "cpu":
        return cpu_model
    dev_model = copy.deepcopy(cpu_model).to(dev_key).eval()
    _MODEL_CACHE[(target_path, dev_key)] = dev_model
    return dev_model


def _fit_3pt_physical_bounds(
    asphalt: np.ndarray,
    x_vp: float,
    y_vp: float,
    x_left_bot: float,
    x_right_bot: float,
    kappa_road: float = 0.0,
    y_bot_ref: float = 0.88,
    num_steps: int = 24,
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """
    Fits smooth converging Phase 8 / Rust-parity cyan road boundaries anchored at
    `(x_vp, y_vp)` down to scan-outward vehicle-avoiding bottom road edges `[xl_bot, xr_bot]`.
    Returns `(ys, left_xs, right_xs, left_poly, right_poly)`.
    """
    h, w = asphalt.shape
    asphalt_c = cv2.morphologyEx(
        asphalt, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (29, 7))
    )
    cx_col = int(np.clip(x_vp, 0.30, 0.70) * w)
    seed_span = max(8, int(0.20 * w))
    l_edges: List[float] = []
    r_edges: List[float] = []

    for yn in np.linspace(max(y_vp + 0.12, 0.64), y_bot_ref, 8):
        py = int(np.clip(yn, 0.0, 0.99) * h)
        strip = np.mean(asphalt_c[max(0, py - 4):min(h, py + 5), :] > 0, axis=0)
        ok = strip >= 0.35

        l_start = next((idx for idx in range(cx_col, max(-1, cx_col - seed_span), -1) if ok[idx]), None)
        if l_start is not None:
            bad, edge_l = 0, l_start
            for idx in range(l_start, -1, -1):
                if ok[idx]:
                    bad = 0
                    edge_l = idx
                else:
                    bad += 1
                    if bad >= 6:
                        break
            if (l_start - edge_l) >= int(0.05 * w):
                l_edges.append(float(edge_l) / float(w))

        r_start = next((idx for idx in range(cx_col, min(w, cx_col + seed_span)) if ok[idx]), None)
        if r_start is not None:
            bad, edge_r = 0, r_start
            for idx in range(r_start, w):
                if ok[idx]:
                    bad = 0
                    edge_r = idx
                else:
                    bad += 1
                    if bad >= 6:
                        break
            if (edge_r - r_start) >= int(0.05 * w):
                r_edges.append(float(edge_r) / float(w))

    xl_bot = float(np.clip(np.median(l_edges) if l_edges else x_left_bot, 0.04, 0.36))
    xr_bot = float(np.clip(np.median(r_edges) if r_edges else x_right_bot, 0.64, 0.96))

    bound_ys = [float(v) for v in np.linspace(y_vp, y_bot_ref, num_steps)]
    left_xs: List[float] = []
    right_xs: List[float] = []
    span = max(0.05, y_bot_ref - y_vp)
    for yn in bound_ys:
        t = max(0.0, min(1.0, (yn - y_vp) / span))
        curve_bias = kappa_road * t * math.sin(math.pi * t)
        lx = x_vp + (xl_bot - x_vp) * (t ** 1.3) + curve_bias
        rx = x_vp + (xr_bot - x_vp) * (t ** 1.3) + curve_bias
        if t > 0.0 and rx < lx:
            mid = 0.5 * (lx + rx)
            lx = mid - 0.01 * t
            rx = mid + 0.01 * t
        left_xs.append(float(np.clip(lx, 0.01, 0.85)))
        right_xs.append(float(np.clip(rx, 0.15, 0.99)))
    left_xs[0] = float(np.clip(x_vp, 0.01, 0.85))
    right_xs[0] = float(np.clip(x_vp, 0.15, 0.99))

    left_poly = [float(c) for c in np.polyfit(bound_ys, left_xs, 2)]
    right_poly = [float(c) for c in np.polyfit(bound_ys, right_xs, 2)]
    return bound_ys, left_xs, right_xs, left_poly, right_poly


def _letterbox(frame: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, float, int, int]:
    """Aspect-preserving resize of `frame` onto a gray `height x width` canvas -> (canvas, scale, pad_x, pad_y)."""
    h, w = frame.shape[:2]
    r = min(width / float(w), height / float(h))
    nw, nh = int(round(w * r)), int(round(h * r))
    px, py = (width - nw) // 2, (height - nh) // 2
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, r, px, py


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """IoU of two `(cx, cy, w, h)` boxes."""
    ix = max(0.0, min(a[0] + a[2] / 2, b[0] + b[2] / 2) - max(a[0] - a[2] / 2, b[0] - b[2] / 2))
    iy = max(0.0, min(a[1] + a[3] / 2, b[1] + b[3] / 2) - max(a[1] - a[3] / 2, b[1] - b[3] / 2))
    inter = ix * iy
    return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-9)


def detect_pyresearch_and_road_potholes(
    frame: np.ndarray,
    use_yolo: bool = True,
    fg_obstacles: Optional[List[Dict[str, Any]]] = None,
    weights_path: Optional[str] = None,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    max_detections: int = MAX_DETECTIONS,
) -> Tuple[List[float], int]:
    """
    Full-frame YOLOv8 pothole detection.

    Keeps raw YOLO confidence >= max(DEFAULT_CONF_THRESHOLD, conf_threshold), drops absurdly large
    boxes and boxes centred inside a detected foreground vehicle, applies greedy NMS and packs the
    survivors as a flat `[7, N]` channel-major tensor (`cx, cy, w, h, conf, 0.03, 0.03`, coordinates
    normalised to the original frame then scaled by 640) for the Rust C-ABI post-process.
    """
    if not use_yolo:
        return [], 0

    h, w = frame.shape[:2]
    thresh = max(DEFAULT_CONF_THRESHOLD, float(conf_threshold))
    canvas, r, px, py = _letterbox(frame, INFER_W, INFER_H)
    infer_device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_pyresearch_yolo_model(weights_path, device=infer_device)
    x = torch.from_numpy(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).float()[None] / 255.0
    with torch.no_grad():
        out = model(x.to(infer_device)).cpu()[0]  # [5, N]: cx, cy, w, h, conf in letterbox pixels

    candidates: List[Tuple[float, float, float, float, float]] = []
    for j in torch.nonzero(out[4] >= thresh, as_tuple=False).flatten().tolist():
        cx = (float(out[0, j]) - px) / r / w
        cy = (float(out[1, j]) - py) / r / h
        bw = float(out[2, j]) / r / w
        bh = float(out[3, j]) / r / h
        if bw > MAX_BOX_W_NORM or bh > MAX_BOX_H_NORM:
            continue
        if any(
            o.get("x_min", 1.0) <= cx <= o.get("x_max", 0.0) and o.get("y_min", 1.0) <= cy <= o.get("y_max", 0.0)
            for o in (fg_obstacles or [])
            if o.get("class_id", 1) != 0
        ):
            continue
        candidates.append((float(out[4, j]), cx, cy, bw, bh))

    candidates.sort(key=lambda c: c[0], reverse=True)
    kept: List[Tuple[float, float, float, float, float]] = []
    for c in candidates:
        if all(_iou(c[1:], k[1:]) < NMS_IOU for k in kept):
            kept.append(c)
        if len(kept) >= max_detections:
            break
    if not kept:
        return [], 0

    tensor: List[float] = []
    for ch in (1, 2, 3, 4, 0):  # cx, cy, w, h, conf
        scale = 1.0 if ch == 0 else 640.0
        tensor.extend(float(k[ch]) * scale for k in kept)
    tensor.extend([0.03] * len(kept) * 2)
    return tensor, len(kept)


def evaluate_video_detections(
    video_path: str,
    stride: int = 6,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    weights_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs the detector over every `stride`-th frame of a video -> per-frame `[(conf, cx, cy, w, h), ...]` (normalised)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    per_frame: List[Dict[str, Any]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            tensor, n = detect_pyresearch_and_road_potholes(frame, conf_threshold=conf_threshold, weights_path=weights_path)
            boxes = [tuple(round(tensor[ch * n + i] / (1.0 if ch == 4 else 640.0), 4) for ch in (4, 0, 1, 2, 3)) for i in range(n)]
            per_frame.append({"frame": idx, "detections": boxes})
        idx += 1
    cap.release()
    return {"frames_scanned": len(per_frame), "per_frame": per_frame}

