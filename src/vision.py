"""
vision.py

The full vision stack:

  1. YOLO models + detection helpers  -> detect_objects / detect_spectacles
  2. JSON persistence                  -> presence + last-seen files
  3. Headless camera loop              -> ties detection + persistence together

config is imported first so QT_QPA_PLATFORM=offscreen is set before cv2 / YOLO.
"""

import os
import json
import time
import traceback

from . import config  # noqa: F401  (imported for its QT_QPA_PLATFORM side effect)
from . import state
from .config import (
    MODEL_KITCHEN, MODEL_SPEC,
    LAST_SEEN_JSON_PATH, PRESENCE_JSON_PATH,
    SPEC_EVERY_N, PRESENCE_PUSH_INTERVAL, SPECTACLE_LABELS, SPEC_THRESHOLD,
)
from .voice import speak_text_async

try:
    import cv2
except Exception as e:
    print("[ERROR] cv2 import failed:", e)
    raise

try:
    from ultralytics import YOLO
except Exception as e:
    print("[ERROR] ultralytics YOLO import failed:", e)
    raise


# =========================================================
# YOLO model loading + detection helpers
# =========================================================
if not os.path.exists(MODEL_KITCHEN):
    raise FileNotFoundError(f"Kitchen model missing: {MODEL_KITCHEN}")
if not os.path.exists(MODEL_SPEC):
    raise FileNotFoundError(f"Spec model missing: {MODEL_SPEC}")

print("[MODEL] loading YOLO models ...")
model = YOLO(MODEL_KITCHEN, task='detect')
spec_model = YOLO(MODEL_SPEC, task='detect')
print("[MODEL] models loaded")


def safe_conf_from_box(box):
    try:
        conf_attr = getattr(box, "conf", None)
        if conf_attr is None:
            return 0.0
        try:
            if len(conf_attr) > 0:
                return float(conf_attr[0])
        except Exception:
            return float(conf_attr)
    except Exception:
        return 0.0


def safe_xyxy_from_box(box):
    try:
        if hasattr(box, "xyxy"):
            pts = box.xyxy[0]
            return tuple(map(int, pts))
    except Exception:
        pass
    return None


def detect_objects(frame):
    res = model(frame, verbose=False)[0]
    best = {}
    names = res.names
    for box in res.boxes:
        try:
            cls_id = int(box.cls[0])
            conf = safe_conf_from_box(box)
            label = names[cls_id]
            if label not in best or conf > best[label]:
                best[label] = conf
        except Exception:
            continue
    return res.boxes, best, names


def detect_spectacles(frame):
    res = spec_model(frame, verbose=False)[0]
    return res.boxes, res.names


# =========================================================
# Presence + last-seen persistence (JSON files)
# =========================================================
def load_last_seen():
    if os.path.exists(LAST_SEEN_JSON_PATH):
        try:
            with open(LAST_SEEN_JSON_PATH, 'r') as f:
                data = json.load(f)
            with state.state_lock:
                state.last_spec_seen.update({
                    'place': data.get('place'),
                    'time': data.get('time'),
                    'conf': data.get('conf'),
                    'bbox': tuple(data['bbox']) if isinstance(data.get('bbox'), list) else data.get('bbox'),
                    'label': data.get('label')
                })
            print("[STATE] loaded last_spec_seen")
        except Exception as e:
            print("[STATE] load failed:", e)


def save_last_seen():
    try:
        data = dict(state.last_spec_seen)
        if isinstance(data.get('bbox'), tuple):
            data['bbox'] = list(data['bbox'])
        if data.get('time'):
            data['time_iso'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data['time']))
        with open(LAST_SEEN_JSON_PATH, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print("[STATE] save failed:", e)


def push_presence_update(location, reason, is_kitchen, score=None, speak=False):
    try:
        ts = time.time()
        payload = {
            'location': location,
            'is_kitchen': bool(is_kitchen),
            'reason': reason or '',
            'score': float(score) if score is not None else None,
            'time': ts,
            'time_iso': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        }
        with state.state_lock:
            state.last_presence = payload
        with open(PRESENCE_JSON_PATH, 'w') as f:
            json.dump(payload, f, indent=2)
        if speak:
            speak_text_async(f"Presence: {payload['location']}. Reason: {reason}", 'en')
    except Exception as e:
        print("[PRESENCE] failed:", e)


# =========================================================
# Headless camera loop
# =========================================================
def open_camera(preferred_size=(640, 480)):
    trials = [
        (0, cv2.CAP_V4L2),
        (0, cv2.CAP_ANY),
        (1, cv2.CAP_V4L2),
        (1, cv2.CAP_ANY),
    ]
    for idx, backend in trials:
        try:
            cap = cv2.VideoCapture(idx, backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, preferred_size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, preferred_size[1])
            if cap.isOpened():
                ok, frame = cap.read()
                if ok:
                    print(f"[CAM] using video index {idx}, backend {backend}")
                    return cap
            try:
                cap.release()
            except Exception:
                pass
        except Exception as e:
            print("[CAM] open attempt failed:", e)
    print("[CAM] no usable camera found")
    return None


def camera_loop():
    load_last_seen()
    cap = open_camera((640, 480))
    if cap is None:
        print("[CAM] aborting camera loop")
        return

    frame_idx = 0
    prev_kitchen_now = None
    last_presence_push = 0.0

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            # presence detection (lightweight)
            boxes, best_conf_map, names = detect_objects(frame)
            kitchen_now = False  # placeholder (put real logic if you have kitchen label)
            now = time.time()
            if (prev_kitchen_now is None) or (kitchen_now != prev_kitchen_now) or (now - last_presence_push >= PRESENCE_PUSH_INTERVAL):
                loc = 'Kitchen' if kitchen_now else 'Unknown'
                loc = 'Level 3 EE department'  # hardcoded for demo
                push_presence_update(loc, "", kitchen_now, None, speak=False)
                last_presence_push = now
                prev_kitchen_now = kitchen_now

            # spectacles detection every N frames
            if frame_idx % SPEC_EVERY_N == 0:
                spec_boxes, spec_names = detect_spectacles(frame)
                only_one_class = len(spec_names) == 1
                best_conf = None
                best_bbox = None
                best_label = None
                for b in spec_boxes:
                    conf = safe_conf_from_box(b)
                    try:
                        cls_id = int(b.cls[0])
                        label = spec_names[cls_id]
                    except Exception:
                        label = None
                    label_norm = str(label).strip().lower() if label else ""
                    if (not only_one_class) and (label_norm not in SPECTACLE_LABELS):
                        continue
                    if best_conf is None or conf > best_conf:
                        best_conf = conf
                        best_label = label
                        best_bbox = safe_xyxy_from_box(b)

                if best_conf is not None and best_conf >= SPEC_THRESHOLD:
                    place = 'Kitchen' if kitchen_now else 'Unknown'
                    place = 'Level 3 EE department'  # hardcoded for demo
                    ts = time.time()
                    with state.state_lock:
                        state.last_spec_seen.update({'place': place, 'time': ts, 'conf': best_conf, 'bbox': best_bbox, 'label': best_label})
                    save_last_seen()
                    if state.find_spec_mode:
                        time_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
                        msg = f"Your spectacles were last seen at {place}, at {time_iso}."
                        print("[ANNOUNCE]", msg)
                        speak_text_async(msg, 'en')
                        state.find_spec_mode = False

            # headless heartbeat logging
            if frame_idx % 150 == 0:
                print(f"[CAM] running... frame {frame_idx}")

            frame_idx += 1
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("[CAM] loop error:", e)
            traceback.print_exc()
            time.sleep(0.5)

    try:
        cap.release()
    except Exception:
        pass
    print("[CAM] camera loop exiting")
