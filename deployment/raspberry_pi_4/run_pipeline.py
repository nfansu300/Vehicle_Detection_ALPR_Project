#!/usr/bin/env python3
"""
================================================================================
MOROCCAN ALPR — FULL TWO-STAGE INFERENCE PIPELINE (RASPBERRY PI 4, CPU)
FIXED VERSION — see CHANGELOG at bottom of this docstring
================================================================================

STAGE 1 : YOLOv8n-100ep, FP16 ONNX  -> vehicle + license-plate detection
STAGE 2 : YOLO11-OCR,     INT8 ONNX -> character detection on cropped plate

USAGE
-----
    python3 alpr_pipeline.py

Just edit the CONFIG block below (model paths + image folder) and run.

REQUIREMENTS (on Pi, no sudo/apt — use pip --break-system-packages)
---------------------------------------------------------------------
    pip install --break-system-packages onnxruntime opencv-python-headless numpy

"""

import os
import sys
import json
import time
import argparse
import statistics
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import cv2
import onnxruntime as ort


# ==============================================================================
# CONFIG — EDIT THESE PATHS FOR YOUR PI
# ==============================================================================

class Config:
    # ---- Model paths (UPDATE THESE) ----
    DETECTION_MODEL_PATH = "/home/pfe_pi/models/yolov8n_100ep_fp16.onnx"
    OCR_MODEL_PATH        = "/home/pfe_pi/models/yolo11_ocr_int8.onnx"

    # ---- Input images ----
    IMAGES_DIR = "/home/pfe_pi/test_images"
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

    # ---- Output ----
    OUTPUT_DIR = "/home/pfe_pi/pipeline_outputs"
    SAVE_ANNOTATED_IMAGES = True
    SAVE_PLATE_CROPS = True

    # ---- Stage 1: Detection (YOLOv8n-100ep FP16) ----
    DET_INPUT_SIZE = 640
    DET_CONF_THRES = 0.25
    DET_IOU_THRES = 0.70
    DET_INPUT_DTYPE = np.float32
    DET_CLASS_NAMES = {
        0: "bus", 1: "car", 2: "license-plate", 3: "motorcycle", 4: "truck",
    }
    PLATE_CLASS_ID = 2

    # NEW: when a vehicle image yields multiple plate boxes, keep only the
    # single highest-confidence plate box (recommended -- see CHANGELOG #2).
    # Set True only if you specifically need every plate in multi-vehicle
    # scenes and are prepared to filter empty/garbage results downstream.
    KEEP_ALL_PLATES = False
    # Minimum confidence a *secondary* plate box must clear to even be
    # considered when KEEP_ALL_PLATES is True (ignored otherwise).
    SECONDARY_PLATE_MIN_CONF = 0.60

    # ---- Stage 2: OCR (YOLO11-OCR INT8) ----
    OCR_INPUT_SIZE = 608
    OCR_CONF_THRES = 0.30          # raised from 0.25 -> fewer noisy/garbage chars
    OCR_IOU_THRES = 0.45
    OCR_INPUT_DTYPE = np.float32
    OCR_CLASS_NAMES = {
        0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
        5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
        10: "a", 11: "b", 12: "d", 13: "gr", 14: "h",
        15: "t", 16: "w", 17: "waw", 18: "y",
    }
    DISPLAY_ARABIC = True
    LATIN_TO_ARABIC = {
        "a": "أ", "b": "ب", "d": "د", "h": "ه", "w": "و",
        "waw": "و", "t": "ط", "gr": "ج", "y": "ي",
    }
    PLATE_CROP_PAD_RATIO = 0.06

    # NEW: plausible Moroccan plate character-count range, used only to
    # flag (not discard) suspicious results for manual audit.
    PLATE_LEN_MIN = 5
    PLATE_LEN_MAX = 9

    # ---- Runtime ----
    ORT_INTRA_THREADS = 4
    ORT_INTER_THREADS = 1
    WARMUP_RUNS = 2


# ==============================================================================
# LOW-LEVEL: LETTERBOX PREPROCESSING (Ultralytics-compatible)
# ==============================================================================

def letterbox(img: np.ndarray, new_size: int, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_size - new_unpad[0], new_size - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def preprocess(img_bgr: np.ndarray, size: int, dtype):
    padded, ratio, (pad_x, pad_y) = letterbox(img_bgr, size)
    img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = img_rgb.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[None, ...]
    tensor = tensor.astype(dtype)
    return tensor, ratio, (pad_x, pad_y)


# ==============================================================================
# LOW-LEVEL: NMS + YOLO OUTPUT DECODE
# ==============================================================================

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep


def decode_ocr_output(raw_output: np.ndarray, class_names: dict,
                       conf_thres: float, ratio: float, pad: Tuple[int, int],
                       orig_shape: Tuple[int, int]):
    """
    YOLO11-OCR ONNX output is end-to-end/NMS-baked: [1, 300, 6] rows of
    [x1, y1, x2, y2, confidence, class_id] in letterboxed (608x608) space.
    No manual NMS needed here, only confidence filtering + unletterboxing.
    """
    out = np.asarray(raw_output)
    if out.ndim == 3:
        out = out[0]
    if out.ndim != 2 or out.shape[1] < 6:
        raise RuntimeError(f"Unexpected OCR output shape: {raw_output.shape}")

    x1 = out[:, 0].astype(np.float32)
    y1 = out[:, 1].astype(np.float32)
    x2 = out[:, 2].astype(np.float32)
    y2 = out[:, 3].astype(np.float32)
    confidence = out[:, 4].astype(np.float32)
    class_id = np.round(out[:, 5]).astype(int)

    mask = confidence >= conf_thres
    if not np.any(mask):
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    boxes = np.stack([x1[mask], y1[mask], x2[mask], y2[mask]], axis=1)
    scores = confidence[mask]
    class_ids = class_id[mask]

    pad_x, pad_y = pad
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= ratio

    H, W = orig_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)

    return boxes, scores, class_ids


def decode_yolo_output(raw_output: np.ndarray, num_classes: int,
                        conf_thres: float, iou_thres: float,
                        ratio: float, pad: Tuple[int, int],
                        orig_shape: Tuple[int, int]):
    """
    YOLOv8/YOLO11-style head. Raw shape [1, 4+num_classes, num_anchors].
    Returns boxes [N,4] xyxy in ORIGINAL image coords, scores [N], class_ids [N]
    """
    out = raw_output
    if out.ndim == 3:
        out = out[0]
    if out.shape[0] != 4 + num_classes and out.shape[-1] == 4 + num_classes:
        out = out.T
    out = out.astype(np.float32)

    boxes_xywh = out[0:4, :].T
    class_scores = out[4:4 + num_classes, :].T

    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    mask = scores >= conf_thres
    if not np.any(mask):
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    x_c, y_c, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    pad_x, pad_y = pad
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= ratio

    H, W = orig_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)

    final_boxes, final_scores, final_classes = [], [], []
    for cls in np.unique(class_ids):
        cls_mask = class_ids == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        keep = nms(cls_boxes, cls_scores, iou_thres)
        final_boxes.append(cls_boxes[keep])
        final_scores.append(cls_scores[keep])
        final_classes.append(np.full(len(keep), cls))

    if not final_boxes:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    return (np.concatenate(final_boxes), np.concatenate(final_scores),
            np.concatenate(final_classes))


# ==============================================================================
# ONNX RUNTIME SESSION WRAPPER
# ==============================================================================

def build_session(model_path: str, cfg: Config) -> ort.InferenceSession:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    so = ort.SessionOptions()
    so.intra_op_num_threads = cfg.ORT_INTRA_THREADS
    so.inter_op_num_threads = cfg.ORT_INTER_THREADS
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    return ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])


_ORT_TYPE_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8,
}


def resolve_input_dtype(session: ort.InferenceSession, configured_dtype, label: str):
    ort_type = session.get_inputs()[0].type
    actual_dtype = _ORT_TYPE_TO_NUMPY.get(ort_type)
    if actual_dtype is None:
        print(f"  ⚠ {label}: unrecognized ORT input type '{ort_type}', "
              f"falling back to configured dtype {configured_dtype}")
        return configured_dtype
    if actual_dtype != configured_dtype:
        print(f"  ⚠ {label}: configured dtype {np.dtype(configured_dtype)} "
              f"does not match model's actual input type {ort_type} "
              f"-> using {np.dtype(actual_dtype)} instead")
    return actual_dtype


# ==============================================================================
# STAGE 1: DETECTION
# ==============================================================================

@dataclass
class Detection:
    box: Tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str


def run_detection(session, img_bgr, cfg: Config, input_dtype) -> Tuple[List[Detection], float]:
    input_name = session.get_inputs()[0].name

    t0 = time.perf_counter()
    tensor, ratio, pad = preprocess(img_bgr, cfg.DET_INPUT_SIZE, input_dtype)
    outputs = session.run(None, {input_name: tensor})
    latency_ms = (time.perf_counter() - t0) * 1000.0

    raw = outputs[0]
    boxes, scores, class_ids = decode_yolo_output(
        raw, num_classes=len(cfg.DET_CLASS_NAMES),
        conf_thres=cfg.DET_CONF_THRES, iou_thres=cfg.DET_IOU_THRES,
        ratio=ratio, pad=pad, orig_shape=img_bgr.shape[:2],
    )

    detections = [
        Detection(
            box=tuple(boxes[i]), score=float(scores[i]), class_id=int(class_ids[i]),
            class_name=cfg.DET_CLASS_NAMES.get(int(class_ids[i]), f"cls{class_ids[i]}"),
        )
        for i in range(len(boxes))
    ]
    return detections, latency_ms


def select_plates_to_ocr(detections: List[Detection], cfg: Config) -> List[Detection]:
    """
    NEW (CHANGELOG #2): decide which 'license-plate' detections actually get
    sent to Stage 2. By default, keep only the single highest-confidence
    plate box per image -- this eliminates the dead-weight OCR calls on
    ghost/duplicate plate boxes that were costing ~360-500ms each and
    frequently producing empty-string results in your log.
    """
    plate_dets = [d for d in detections if d.class_id == cfg.PLATE_CLASS_ID]
    if not plate_dets:
        return []

    plate_dets.sort(key=lambda d: d.score, reverse=True)

    if not cfg.KEEP_ALL_PLATES:
        return [plate_dets[0]]

    # KEEP_ALL_PLATES=True: keep the best one, plus any secondary boxes
    # that clear a stricter confidence bar (avoids OCR'ing near-zero-conf
    # ghost boxes while still supporting genuine multi-vehicle frames).
    kept = [plate_dets[0]]
    kept += [d for d in plate_dets[1:] if d.score >= cfg.SECONDARY_PLATE_MIN_CONF]
    return kept


# ==============================================================================
# STAGE 2: OCR (character detection + assembly)
# ==============================================================================

@dataclass
class CharDetection:
    box: Tuple[float, float, float, float]
    score: float
    class_id: int
    char: str


def run_ocr(session, plate_crop_bgr, cfg: Config, input_dtype) -> Tuple[List[CharDetection], float]:
    input_name = session.get_inputs()[0].name

    t0 = time.perf_counter()
    tensor, ratio, pad = preprocess(plate_crop_bgr, cfg.OCR_INPUT_SIZE, input_dtype)
    outputs = session.run(None, {input_name: tensor})
    latency_ms = (time.perf_counter() - t0) * 1000.0

    raw = outputs[0]
    boxes, scores, class_ids = decode_ocr_output(
        raw, class_names=cfg.OCR_CLASS_NAMES,
        conf_thres=cfg.OCR_CONF_THRES,
        ratio=ratio, pad=pad, orig_shape=plate_crop_bgr.shape[:2],
    )

    chars = [
        CharDetection(
            box=tuple(boxes[i]), score=float(scores[i]), class_id=int(class_ids[i]),
            char=cfg.OCR_CLASS_NAMES.get(int(class_ids[i]), "?"),
        )
        for i in range(len(boxes))
    ]
    return chars, latency_ms


def assemble_plate_text(chars: List[CharDetection], cfg: Config) -> Tuple[str, bool]:
    """Order left-to-right by horizontal box center. Returns (text, len_flag)
    where len_flag=True if char count falls outside PLATE_LEN_MIN/MAX
    (NEW -- CHANGELOG #3: for audit, not for silent discarding)."""
    if not chars:
        return "", True

    ordered = sorted(chars, key=lambda c: (c.box[0] + c.box[2]) / 2.0)
    tokens = []
    for c in ordered:
        tok = c.char
        if cfg.DISPLAY_ARABIC:
            tok = cfg.LATIN_TO_ARABIC.get(tok, tok)
        tokens.append(tok)

    flag = not (cfg.PLATE_LEN_MIN <= len(tokens) <= cfg.PLATE_LEN_MAX)
    return " ".join(tokens), flag


# ==============================================================================
# CROPPING HELPER
# ==============================================================================

def crop_plate(img_bgr, box, pad_ratio) -> Optional[np.ndarray]:
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 1 or h <= 1:
        return None
    pad_x, pad_y = w * pad_ratio, h * pad_ratio
    x1 = int(max(0, x1 - pad_x))
    y1 = int(max(0, y1 - pad_y))
    x2 = int(min(W, x2 + pad_x))
    y2 = int(min(H, y2 + pad_y))
    if x2 <= x1 or y2 <= y1:
        return None
    return img_bgr[y1:y2, x1:x2].copy()


# ==============================================================================
# ANNOTATION / VISUALIZATION
# ==============================================================================

def draw_annotations(img_bgr, detections: List[Detection], plate_texts: List[str],
                      ocr_plate_ids: List[int]) -> np.ndarray:
    out = img_bgr.copy()
    text_by_id = dict(zip(ocr_plate_ids, plate_texts))
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in det.box]
        color = (0, 200, 0) if det.class_name != "license-plate" else (0, 140, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.score:.2f}"
        if idx in text_by_id:
            label += f" | {text_by_id[idx]}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# ==============================================================================
# METRICS
# ==============================================================================

class LatencyTracker:
    def __init__(self, warmup_runs: int = 0):
        self.warmup_runs = warmup_runs
        self._all: List[float] = []

    def add(self, ms: float):
        self._all.append(ms)

    @property
    def values(self):
        return self._all[self.warmup_runs:] if len(self._all) > self.warmup_runs else self._all

    def summary(self):
        v = self.values
        if not v:
            return {}
        return {
            "count": len(v),
            "mean_ms": statistics.mean(v),
            "median_ms": statistics.median(v),
            "min_ms": min(v),
            "max_ms": max(v),
            "p95_ms": sorted(v)[int(0.95 * (len(v) - 1))] if len(v) > 1 else v[0],
            "fps": 1000.0 / statistics.mean(v) if statistics.mean(v) > 0 else 0.0,
        }


# ==============================================================================
# JSON SAFETY (FIX for CHANGELOG #1)
# ==============================================================================

def to_py(obj):
    """Recursively convert numpy scalar/array types to native Python types
    so json.dump never chokes on np.float32 / np.int64 / etc."""
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def safe_round(v, ndigits=3):
    """round() that always returns a native Python float, never np.float32."""
    return round(float(v), ndigits)


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def find_images(images_dir: str, extensions: Tuple[str, ...]) -> List[Path]:
    root = Path(images_dir)
    if not root.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in extensions])


def run_pipeline(cfg: Config):
    print("=" * 90)
    print("MOROCCAN ALPR — FULL TWO-STAGE PIPELINE (RASPBERRY PI 4, CPU) — FIXED")
    print("=" * 90)
    print(f"Detection model : {cfg.DETECTION_MODEL_PATH}")
    print(f"OCR model       : {cfg.OCR_MODEL_PATH}")
    print(f"Images dir      : {cfg.IMAGES_DIR}")
    print(f"ORT threads     : intra={cfg.ORT_INTRA_THREADS} inter={cfg.ORT_INTER_THREADS}")
    print(f"KEEP_ALL_PLATES : {cfg.KEEP_ALL_PLATES}")
    print("=" * 90)

    out_dir = Path(cfg.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "annotated").mkdir(exist_ok=True)
    (out_dir / "plate_crops").mkdir(exist_ok=True)

    print("\n→ Loading ONNX Runtime sessions...")
    t_load0 = time.perf_counter()
    det_session = build_session(cfg.DETECTION_MODEL_PATH, cfg)
    ocr_session = build_session(cfg.OCR_MODEL_PATH, cfg)
    t_load = time.perf_counter() - t_load0
    print(f"  ✓ Both models loaded in {t_load:.2f}s")
    print(f"  Detection input  : {det_session.get_inputs()[0].name} "
          f"{det_session.get_inputs()[0].shape} {det_session.get_inputs()[0].type}")
    print(f"  OCR input        : {ocr_session.get_inputs()[0].name} "
          f"{ocr_session.get_inputs()[0].shape} {ocr_session.get_inputs()[0].type}")

    det_input_dtype = resolve_input_dtype(det_session, cfg.DET_INPUT_DTYPE, "Detection")
    ocr_input_dtype = resolve_input_dtype(ocr_session, cfg.OCR_INPUT_DTYPE, "OCR")

    images = find_images(cfg.IMAGES_DIR, cfg.IMAGE_EXTENSIONS)
    if not images:
        print(f"\n✗ No images found in {cfg.IMAGES_DIR} with extensions {cfg.IMAGE_EXTENSIONS}")
        sys.exit(1)
    print(f"\n→ Found {len(images)} image(s) to process")

    det_tracker = LatencyTracker(cfg.WARMUP_RUNS)
    ocr_tracker = LatencyTracker(cfg.WARMUP_RUNS)
    e2e_tracker = LatencyTracker(cfg.WARMUP_RUNS)

    results = []
    errors = []
    pipeline_start = time.perf_counter()

    for idx, img_path in enumerate(images, 1):
        print(f"\n[{idx}/{len(images)}] {img_path.name}")
        try:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"  ✗ Failed to read image, skipping")
                errors.append({"image": img_path.name, "error": "cv2.imread returned None"})
                continue

            e2e_t0 = time.perf_counter()

            detections, det_latency_ms = run_detection(det_session, img_bgr, cfg, det_input_dtype)
            det_tracker.add(det_latency_ms)
            print(f"  Stage1 (detect) : {det_latency_ms:8.2f} ms | "
                  f"{len(detections)} object(s) -> {[d.class_name for d in detections]}")

            plates_to_process = select_plates_to_ocr(detections, cfg)
            n_all_plate_boxes = sum(1 for d in detections if d.class_id == cfg.PLATE_CLASS_ID)
            if n_all_plate_boxes > len(plates_to_process):
                print(f"    ({n_all_plate_boxes} plate box(es) found, "
                      f"OCR'ing top {len(plates_to_process)} by confidence)")

            plate_texts = []
            plate_len_flags = []
            ocr_plate_ids = []  # index into `detections` for annotation lookup

            for plate_det in plates_to_process:
                det_idx = detections.index(plate_det)
                crop = crop_plate(img_bgr, plate_det.box, cfg.PLATE_CROP_PAD_RATIO)
                if crop is None or crop.size == 0:
                    plate_texts.append("")
                    plate_len_flags.append(True)
                    ocr_plate_ids.append(det_idx)
                    print(f"    Plate: invalid crop, skipped")
                    continue

                if cfg.SAVE_PLATE_CROPS:
                    crop_path = out_dir / "plate_crops" / f"{img_path.stem}_plate{det_idx}.jpg"
                    cv2.imwrite(str(crop_path), crop)

                chars, ocr_latency_ms = run_ocr(ocr_session, crop, cfg, ocr_input_dtype)
                ocr_tracker.add(ocr_latency_ms)
                plate_text, len_flag = assemble_plate_text(chars, cfg)
                plate_texts.append(plate_text)
                plate_len_flags.append(len_flag)
                ocr_plate_ids.append(det_idx)
                flag_str = "  [LEN-FLAG]" if len_flag else ""
                print(f"    Plate: '{plate_text}'  ({len(chars)} chars, "
                      f"{ocr_latency_ms:7.2f} ms, conf={plate_det.score:.2f}){flag_str}")

            e2e_latency_ms = (time.perf_counter() - e2e_t0) * 1000.0
            e2e_tracker.add(e2e_latency_ms)
            print(f"  End-to-end      : {e2e_latency_ms:8.2f} ms")

            if cfg.SAVE_ANNOTATED_IMAGES:
                annotated = draw_annotations(img_bgr, detections, plate_texts, ocr_plate_ids)
                out_path = out_dir / "annotated" / f"{img_path.stem}_annotated.jpg"
                cv2.imwrite(str(out_path), annotated)

            results.append({
                "image": img_path.name,
                "detections": [
                    {"class": d.class_name, "score": safe_round(d.score, 4),
                     "box": [safe_round(v, 1) for v in d.box]}
                    for d in detections
                ],
                "plates": plate_texts,
                "plate_len_flags": plate_len_flags,
                "num_plate_boxes_detected": n_all_plate_boxes,
                "num_plates_ocred": len(plates_to_process),
                "det_latency_ms": safe_round(det_latency_ms),
                "e2e_latency_ms": safe_round(e2e_latency_ms),
            })

        except Exception as e:
            print(f"  ✗ ERROR processing {img_path.name}: {e}")
            errors.append({"image": img_path.name, "error": str(e)})
            continue

    total_wall_time = time.perf_counter() - pipeline_start

    # ==========================================================================
    # METRICS SUMMARY
    # ==========================================================================
    print("\n" + "=" * 90)
    print("PIPELINE METRICS SUMMARY")
    print("=" * 90)

    det_summary = det_tracker.summary()
    ocr_summary = ocr_tracker.summary()
    e2e_summary = e2e_tracker.summary()

    def print_summary(name, s):
        if not s:
            print(f"\n{name}: no data")
            return
        print(f"\n{name} (n={s['count']}, warmup={cfg.WARMUP_RUNS} discarded):")
        print(f"  Mean latency   : {s['mean_ms']:.2f} ms")
        print(f"  Median latency : {s['median_ms']:.2f} ms")
        print(f"  P95 latency    : {s['p95_ms']:.2f} ms")
        print(f"  Min / Max      : {s['min_ms']:.2f} / {s['max_ms']:.2f} ms")
        print(f"  FPS            : {s['fps']:.3f}")

    print_summary("Stage 1 — Detection (YOLOv8n-100ep FP16)", det_summary)
    print_summary("Stage 2 — OCR (YOLO11-OCR INT8)", ocr_summary)
    print_summary("End-to-end (per image, includes both stages + cropping)", e2e_summary)

    n_flagged = sum(1 for r in results for f in r["plate_len_flags"] if f)
    n_empty = sum(1 for r in results for p in r["plates"] if p == "")
    n_total_plates = sum(len(r["plates"]) for r in results)

    print(f"\nTotal images processed : {len(results)}")
    print(f"Errors/skipped          : {len(errors)}")
    print(f"Total plates OCR'd      : {n_total_plates}")
    print(f"  Empty-string results  : {n_empty} ({100*n_empty/max(1,n_total_plates):.1f}%)")
    print(f"  Length-flagged (audit): {n_flagged} ({100*n_flagged/max(1,n_total_plates):.1f}%)")
    print(f"Total wall time         : {total_wall_time:.2f} s")
    if total_wall_time > 0:
        print(f"Overall throughput      : {len(results) / total_wall_time:.3f} images/s")

    # ==========================================================================
    # SAVE OUTPUTS (numpy-safe)
    # ==========================================================================
    metrics_payload = {
        "config": {
            "detection_model": cfg.DETECTION_MODEL_PATH,
            "ocr_model": cfg.OCR_MODEL_PATH,
            "det_input_size": cfg.DET_INPUT_SIZE,
            "ocr_input_size": cfg.OCR_INPUT_SIZE,
            "det_conf_thres": cfg.DET_CONF_THRES,
            "ocr_conf_thres": cfg.OCR_CONF_THRES,
            "ort_intra_threads": cfg.ORT_INTRA_THREADS,
            "keep_all_plates": cfg.KEEP_ALL_PLATES,
        },
        "num_images": len(results),
        "num_errors": len(errors),
        "total_wall_time_s": safe_round(total_wall_time),
        "stage1_detection": det_summary,
        "stage2_ocr": ocr_summary,
        "end_to_end": e2e_summary,
        "model_load_time_s": safe_round(t_load),
        "n_empty_plates": n_empty,
        "n_len_flagged_plates": n_flagged,
        "n_total_plates_ocred": n_total_plates,
    }

    metrics_path = out_dir / "metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(to_py(metrics_payload), f, indent=2)

    results_path = out_dir / "per_image_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(to_py(results), f, indent=2, ensure_ascii=False)

    if errors:
        errors_path = out_dir / "errors.json"
        with open(errors_path, "w", encoding="utf-8") as f:
            json.dump(to_py(errors), f, indent=2, ensure_ascii=False)
        print(f"✓ Errors logged to   : {errors_path}")

    print(f"\n✓ Metrics saved to   : {metrics_path}")
    print(f"✓ Results saved to   : {results_path}")
    if cfg.SAVE_ANNOTATED_IMAGES:
        print(f"✓ Annotated images in: {out_dir / 'annotated'}")
    if cfg.SAVE_PLATE_CROPS:
        print(f"✓ Plate crops in     : {out_dir / 'plate_crops'}")
    print("\n" + "=" * 90)
    print("PIPELINE COMPLETE")
    print("=" * 90)


# ==============================================================================
# CLI OVERRIDES
# ==============================================================================

def parse_args(cfg: Config):
    ap = argparse.ArgumentParser(description="Moroccan ALPR two-stage pipeline (fixed)")
    ap.add_argument("--det-model", default=cfg.DETECTION_MODEL_PATH)
    ap.add_argument("--ocr-model", default=cfg.OCR_MODEL_PATH)
    ap.add_argument("--images-dir", default=cfg.IMAGES_DIR)
    ap.add_argument("--output-dir", default=cfg.OUTPUT_DIR)
    ap.add_argument("--det-conf", type=float, default=cfg.DET_CONF_THRES)
    ap.add_argument("--ocr-conf", type=float, default=cfg.OCR_CONF_THRES)
    ap.add_argument("--threads", type=int, default=cfg.ORT_INTRA_THREADS)
    ap.add_argument("--keep-all-plates", action="store_true")
    ap.add_argument("--no-annotate", action="store_true")
    ap.add_argument("--no-crops", action="store_true")
    args = ap.parse_args()

    cfg.DETECTION_MODEL_PATH = args.det_model
    cfg.OCR_MODEL_PATH = args.ocr_model
    cfg.IMAGES_DIR = args.images_dir
    cfg.OUTPUT_DIR = args.output_dir
    cfg.DET_CONF_THRES = args.det_conf
    cfg.OCR_CONF_THRES = args.ocr_conf
    cfg.ORT_INTRA_THREADS = args.threads
    cfg.KEEP_ALL_PLATES = args.keep_all_plates
    cfg.SAVE_ANNOTATED_IMAGES = not args.no_annotate
    cfg.SAVE_PLATE_CROPS = not args.no_crops
    return cfg


if __name__ == "__main__":
    cfg = Config()
    cfg = parse_args(cfg)
    run_pipeline(cfg)
