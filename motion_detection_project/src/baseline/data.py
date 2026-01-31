from __future__ import annotations
import os
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import cv2

from .utils import overlap_fraction


def load_annotations(csv_path: str) -> Dict[str, List[Tuple[float, float]]]:
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    ann: Dict[str, List[Tuple[float, float]]] = {}
    for _, r in df.iterrows():
        video = str(r["video"])
        ann.setdefault(video, []).append(
            (float(r["t_start_sec"]), float(r["t_end_sec"]))
        )
    return ann


def resize_with_aspect(
    img: np.ndarray, width: int, height: int, keep_aspect: bool
) -> np.ndarray:
    if not keep_aspect:
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=resized.dtype)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def roi_mask(h: int, w: int, roi_polygon_norm: List[List[float]]) -> np.ndarray:
    if not roi_polygon_norm:
        return np.ones((h, w), dtype=np.uint8)
    pts = np.array(
        [[int(x * w), int(y * h)] for x, y in roi_polygon_norm], dtype=np.int32
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def extract_motion_scores(
    video_path: str, cfg: Dict[str, Any]
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Return motion_score per time window and corresponding (t0,t1) windows."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 0:
        src_fps = 25.0

    target_fps = float(cfg["preprocess"]["target_fps"])
    step = max(1, int(round(src_fps / target_fps)))

    W = int(cfg["preprocess"]["resize"]["width"])
    H = int(cfg["preprocess"]["resize"]["height"])
    keep_aspect = bool(cfg["preprocess"]["resize"]["keep_aspect"])

    win_sec = float(cfg["features"]["window_sec"])
    blur_k = int(cfg["features"]["blur_ksize"])
    diff_thr = int(cfg["features"]["diff_threshold"])

    ok, prev = cap.read()
    if not ok:
        cap.release()
        return np.zeros((0,), dtype=float), []

    prev = resize_with_aspect(prev, W, H, keep_aspect)
    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    if blur_k > 0:
        prev_g = cv2.GaussianBlur(prev_g, (blur_k, blur_k), 0)

    mask = roi_mask(H, W, cfg["preprocess"].get("roi_polygon_norm", []))
    total = float(np.sum(mask))

    frame_idx = 0
    sampled_t: List[float] = []
    sampled_frac: List[float] = []

    while True:
        for _ in range(step):
            ok, frame = cap.read()
            frame_idx += 1
            if not ok:
                break
        if not ok:
            break

        t_sec = frame_idx / src_fps
        frame = resize_with_aspect(frame, W, H, keep_aspect)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if blur_k > 0:
            g = cv2.GaussianBlur(g, (blur_k, blur_k), 0)

        diff = cv2.absdiff(g, prev_g)
        _, binm = cv2.threshold(diff, diff_thr, 1, cv2.THRESH_BINARY)
        binm = binm.astype(np.uint8) * mask
        frac = float(np.sum(binm)) / max(1.0, total)

        sampled_t.append(float(t_sec))
        sampled_frac.append(float(frac))
        prev_g = g

    cap.release()

    if not sampled_t:
        return np.zeros((0,), dtype=float), []

    scores: List[float] = []
    windows: List[Tuple[float, float]] = []

    cur = sampled_t[0]
    end = sampled_t[-1]
    i = 0
    while cur <= end + 1e-9:
        w0, w1 = cur, cur + win_sec
        vals = []
        while i < len(sampled_t) and sampled_t[i] < w1:
            if sampled_t[i] >= w0:
                vals.append(sampled_frac[i])
            i += 1
        scores.append(float(np.mean(vals)) if vals else 0.0)
        windows.append((float(w0), float(w1)))
        cur = w1

    return np.array(scores, dtype=float), windows


def label_windows(
    video_rel: str,
    windows: List[Tuple[float, float]],
    annotations: Dict[str, List[Tuple[float, float]]],
    overlap_thr: float,
) -> np.ndarray:
    ints = annotations.get(video_rel, [])
    y = []
    for w0, w1 in windows:
        pos = False
        for a0, a1 in ints:
            if overlap_fraction(w0, w1, a0, a1) >= overlap_thr:
                pos = True
                break
        y.append(1 if pos else 0)
    return np.array(y, dtype=int)
