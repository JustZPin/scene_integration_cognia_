#!/usr/bin/env python3
"""
scene_detector_headless.py previous version

Headless (no Qt) version:
 - YOLO camera (kitchen + spectacles)
 - Porcupine wake-word ("Hey Pico")
 - AssemblyAI streaming ASR
 - gTTS playback via pygame
 - Presence + last-seen persistence (JSON files)

Config via environment variables:
 - MODEL_KITCHEN, MODEL_SPEC: paths to your YOLO models
 - PV_DEVICE_INDEX: integer device index for PvRecorder (optional)
 - ASSEMBLYAI_API_KEY, PICOVOICE_ACCESS_KEY
"""

import os
import sys
import time
import json
import threading
import traceback
from collections import deque

# Ensure no GUI backend is requested
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# --- Imports that may raise errors; handle gracefully ---
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

# voice libs
try:
    import pygame
    from gtts import gTTS
except Exception as e:
    print("[ERROR] TTS/playback imports failed:", e)
    raise

try:
    import assemblyai as aai
    from assemblyai.streaming.v3 import (
        BeginEvent, StreamingClient, StreamingClientOptions, StreamingParameters,
        StreamingEvents, TerminationEvent, TurnEvent, StreamingError
    )
except Exception as e:
    print("[ERROR] AssemblyAI import failed:", e)
    raise

try:
    from pvrecorder import PvRecorder
    import pvporcupine
except Exception as e:
    print("[ERROR] Porcupine/PvRecorder import failed:", e)
    raise

import requests

# -------------------------
# CONFIG (edit / override via env)
# -------------------------
MODEL_KITCHEN = os.environ.get("MODEL_KITCHEN", "/home/zhipin/Documents/scene_integration/my_model.pt")
MODEL_SPEC = os.environ.get("MODEL_SPEC", "/home/zhipin/Documents/scene_integration/my_model_spec.pt")
LAST_SEEN_JSON_PATH = os.environ.get("LAST_SEEN_JSON_PATH", "last_spec_seen.json")
PRESENCE_JSON_PATH = os.environ.get("PRESENCE_JSON_PATH", "presence.json")

SPEC_EVERY_N = int(os.environ.get("SPEC_EVERY_N", "3"))
PRESENCE_PUSH_INTERVAL = float(os.environ.get("PRESENCE_PUSH_INTERVAL", "15"))

ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "a5da8071d9a04261bb43d7c414664ff7")
PICOVOICE_ACCESS_KEY = os.environ.get("PICOVOICE_ACCESS_KEY", "FSo7LADnWrT20JU6nUyBrhypXy+U/1AikrLYeaPmwNHE1yQTDJ/aog==")
KEYWORD_PATH = os.environ.get("KEYWORD_PATH", "/home/zhipin/Documents/scene_integration/hey-pico_en_raspberry-pi_v3_0_0.ppn")

# optional: set PV device index via env; if unset, we'll try a fallback approach
PV_DEVICE_INDEX = os.environ.get("PV_DEVICE_INDEX", None)
if PV_DEVICE_INDEX is not None:
    try:
        PV_DEVICE_INDEX = int(PV_DEVICE_INDEX)
    except Exception:
        PV_DEVICE_INDEX = None

# Other settings
SPECTACLE_LABELS = {'spectacle','glasses','spectacles','eyeglasses','sunglasses'}
SPEC_THRESHOLD = float(os.environ.get("SPEC_THRESHOLD", "0.60"))

# -------------------------
# STATE
# -------------------------
state_lock = threading.Lock()
last_presence = None
last_spec_seen = {'place': None, 'time': None, 'conf': None, 'bbox': None, 'label': None}
wake_active = False
find_spec_mode = False
voiceflow_mode = False  # not used now, kept for compatibility
last_transcript = ""
recent_decisions = deque(maxlen=5)

# -------------------------
# Utilities: TTS (gTTS + pygame)
# -------------------------
def speak_text(text, lang='en'):
    """Blocking tts then playback (non-fatal)."""
    if not text:
        return
    try:
        fname = f"/tmp/tts_{int(time.time()*1000)}.mp3"
        tts = gTTS(text=text, lang=lang)
        tts.save(fname)
        try:
            pygame.mixer.init(frequency=44100)
            pygame.mixer.music.load(fname)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(50)
        except Exception as e:
            print("[TTS] playback error:", e)
        try:
            os.remove(fname)
        except Exception:
            pass
    except Exception as e:
        print("[TTS] failed to create/play TTS:", e)

def speak_text_async(text, lang='en'):
    threading.Thread(target=speak_text, args=(text, lang), daemon=True).start()

# -------------------------
# Persistence helpers
# -------------------------
def load_last_seen():
    global last_spec_seen
    if os.path.exists(LAST_SEEN_JSON_PATH):
        try:
            with open(LAST_SEEN_JSON_PATH, 'r') as f:
                data = json.load(f)
            with state_lock:
                last_spec_seen.update({
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
        data = dict(last_spec_seen)
        if isinstance(data.get('bbox'), tuple):
            data['bbox'] = list(data['bbox'])
        if data.get('time'):
            data['time_iso'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data['time']))
        with open(LAST_SEEN_JSON_PATH, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print("[STATE] save failed:", e)

def push_presence_update(location, reason, is_kitchen, score=None, speak=False):
    global last_presence
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
        with state_lock:
            last_presence = payload
        with open(PRESENCE_JSON_PATH, 'w') as f:
            json.dump(payload, f, indent=2)
        if speak:
            speak_text_async(f"Presence: {payload['location']}. Reason: {reason}", 'en')
    except Exception as e:
        print("[PRESENCE] failed:", e)

# -------------------------
# YOLO model loading
# -------------------------
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

# -------------------------
# AssemblyAI handlers
# -------------------------
def on_begin(client, event: BeginEvent):
    global wake_active, last_transcript
    wake_active = True
    last_transcript = ""
    print("[ASR] session started")
    speak_text_async("I'm listening.", 'en')

def parse_reminder(text):
    import re
    t = text.lower()
    if "remind me" not in t:
        return None
    m = re.search(r"(\d+)\s*(second|seconds|sec|min|minute|minutes|hour|hours)?", t)
    if not m:
        return None
    v = int(m.group(1))
    u = m.group(2) or "second"
    if "min" in u:
        return v * 60
    if "hour" in u:
        return v * 3600
    return v

def countdown_timer(seconds):
    for i in range(seconds, 0, -1):
        time.sleep(1)
    speak_text_async(f"{seconds} seconds reached!", 'en')

def on_turn(client, event: TurnEvent):
    global wake_active, find_spec_mode, last_transcript, voiceflow_mode

    # Ignore partial empty transcripts
    if not event.end_of_turn:
        return

    text = event.transcript or ""
    text = text.strip()

    # Ignore empty noise responses
    if text == "":
        return

    # Ignore duplicates
    if text.lower() == last_transcript.lower():
        return

    last_transcript = text

    print("[ASR]", text)
    low = text.lower()

    # --- Control voiceflow ---
    if low == "start":
        voiceflow_mode = True
        speak_text_async("Voiceflow activated.", 'en')
        return
    
    if low == "exit voiceflow":
        voiceflow_mode = False
        speak_text_async("Voiceflow deactivated.", 'en')
        return

    if voiceflow_mode:
        vf = call_voiceflow(text)
        speak_voiceflow_response(vf)
        return

    # --- Reminder logic ---
    delay = parse_reminder(text)
    if delay:
        threading.Thread(target=countdown_timer, args=(delay,), daemon=True).start()
        speak_text_async("Reminder set.", 'en')
        return

    # --- Location inquiry ---
    if 'where am i' in low or 'what is my location' in low:
        try:
            with open(PRESENCE_JSON_PATH, 'r') as f:
                p = json.load(f)
            msg = f"You are at {p.get('location', 'Unknown')}."
            speak_text_async(msg, 'en')
        except:
            speak_text_async("I cannot read presence right now.", 'en')
        return

    # --- Glasses detection ---
    if 'where' in low and ('spec' in low or 'spectacle' in low or 'glasses' in low):
        find_spec_mode = True
        speak_text_async("Turn around to check.", 'en')
        return

    # --- Fallback translation ---
    try:
        translated = translator.translate(text)
        print("[TRANSLATION]", translated)
    except:
        pass

def on_terminated(client, event: TerminationEvent):
    global wake_active
    wake_active = False
    print("[ASR] session ended")
    speak_text_async("Session ended.", 'en')

def on_error(client, error: StreamingError):
    print("[ASR] error:", error)

def start_assembly_ai(sample_rate=44100):
    """Blocking: starts streaming client and returns after the session ends."""
    client = StreamingClient(StreamingClientOptions(api_key=ASSEMBLYAI_API_KEY, api_host="streaming.assemblyai.com"))
    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Termination, on_terminated)
    client.on(StreamingEvents.Error, on_error)

    try:
        client.connect(StreamingParameters(sample_rate=sample_rate, format_turns=True))
        mic = aai.extras.MicrophoneStream(sample_rate=sample_rate)
        client.stream(mic)
    finally:
        try:
            client.disconnect(terminate=True)
        except Exception:
            pass

# -------------------------
# Wake-word (Porcupine) loop
# -------------------------
def choose_pv_device_index():
    """Return integer device index for PvRecorder, try PV_DEVICE_INDEX, else try reasonable fallbacks."""
    if PV_DEVICE_INDEX is not None:
        print("[PV] using PV_DEVICE_INDEX from env:", PV_DEVICE_INDEX)
        return PV_DEVICE_INDEX

    # try to list devices
    try:
        devices = PvRecorder.get_audio_devices()
        print("[PV] available audio devices:", devices)
        # Heuristic: choose first device that mentions 'usb' or 'mic' else default 0
        for i, d in enumerate(devices):
            dd = d.lower() if isinstance(d, str) else ""
            if "usb" in dd or "mic" in dd or "input" in dd:
                print("[PV] heuristic selected device index", i, "->", d)
                return i
        print("[PV] defaulting to device index 0")
        return 0
    except Exception as e:
        print("[PV] could not list devices, defaulting to 0:", e)
        return 0

def wake_word_listener_loop():
    global find_spec_mode
    print("[WAKE] initializing Porcupine ...")
    device_index = choose_pv_device_index()
    try:
        porcupine = pvporcupine.create(access_key=PICOVOICE_ACCESS_KEY, keyword_paths=[KEYWORD_PATH])
    except Exception as e:
        print("[WAKE] Porcupine create failed:", e)
        return

    try:
        recorder = PvRecorder(device_index=device_index, frame_length=porcupine.frame_length)
    except Exception as e:
        print(f"[WAKE] PvRecorder init failed with device {device_index}:", e)
        # try fallback without specifying device index
        try:
            recorder = PvRecorder(frame_length=porcupine.frame_length)
        except Exception as e2:
            print("[WAKE] PvRecorder fallback failed:", e2)
            return

    try:
        recorder.start()
    except Exception as e:
        print("[WAKE] recorder.start() failed:", e)
        return

    print("[WAKE] Ready - say the wake-word")
    try:
        while True:
            try:
                pcm = recorder.read()
                res = porcupine.process(pcm)
                if res >= 0:
                    print("[WAKE] detected")
                    # stop recorder while handling
                    try:
                        recorder.stop()
                    except Exception:
                        pass
                    try:
                        recorder.delete()
                    except Exception:
                        pass
                    try:
                        porcupine.delete()
                    except Exception:
                        pass

                    # Start AssemblyAI session (blocks until finished)
                    try:
                        start_assembly_ai(sample_rate=44100)
                    except Exception as e:
                        print("[WAKE] assembly ai session failed:", e)

                    # Recreate porcupine & recorder after session
                    try:
                        porcupine = pvporcupine.create(access_key=PICOVOICE_ACCESS_KEY, keyword_paths=[KEYWORD_PATH])
                        recorder = PvRecorder(device_index=device_index, frame_length=porcupine.frame_length)
                        recorder.start()
                        print("[WAKE] resumed listening")
                    except Exception as e:
                        print("[WAKE] recreate failed:", e)
                        break
            except KeyboardInterrupt:
                break
            except Exception as e:
                print("[WAKE] loop error:", e)
                time.sleep(0.5)
    finally:
        try:
            recorder.stop()
            recorder.delete()
        except Exception:
            pass
        try:
            porcupine.delete()
        except Exception:
            pass
        print("[WAKE] listener exiting")

# -------------------------
# Camera loop (headless)
# -------------------------
def open_camera(preferred_size=(640,480)):
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
    cap = open_camera((640,480))
    if cap is None:
        print("[CAM] aborting camera loop")
        return

    frame_idx = 0
    prev_kitchen_now = None
    last_presence_push = 0.0
    global find_spec_mode

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
                loc = 'Level 3 EE department' # hardcoded for demo
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
                    with state_lock:
                        last_spec_seen.update({'place': place, 'time': ts, 'conf': best_conf, 'bbox': best_bbox, 'label': best_label})
                    save_last_seen()
                    if find_spec_mode:
                        time_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
                        msg = f"Your spectacles were last seen at {place}, at {time_iso}."
                        print("[ANNOUNCE]", msg)
                        speak_text_async(msg, 'en')
                        find_spec_mode = False

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

# -------------------------
# MAIN
# -------------------------
def main():
    # sanity check: API keys & models
    if not os.path.exists(MODEL_KITCHEN):
        print("[MAIN] Kitchen model missing:", MODEL_KITCHEN)
        return
    if not os.path.exists(MODEL_SPEC):
        print("[MAIN] Spec model missing:", MODEL_SPEC)
        return
    if not os.path.exists(KEYWORD_PATH):
        print("[MAIN] keyword file missing:", KEYWORD_PATH)
        # not fatal; porcupine will fail at runtime

    # start wakeword listener thread
    t_wake = threading.Thread(target=wake_word_listener_loop, daemon=True)
    t_wake.start()

    # start camera loop in main thread (so KeyboardInterrupt works)
    try:
        camera_loop()
    except KeyboardInterrupt:
        print("[MAIN] KeyboardInterrupt received, exiting")
    finally:
        print("[MAIN] exiting")

if _name_ == "_main_":
    main()
