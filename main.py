#!/usr/bin/env python3
"""
scene_detector_headless.py

Headless (no Qt) entry point. Wires together the split modules under ``src/``:
 - YOLO camera (kitchen + spectacles)      -> src/vision.py
 - Porcupine wake-word ("Hey Pico")         -> src/voice.py
 - AssemblyAI streaming ASR                 -> src/voice.py
 - gTTS playback via pygame                 -> src/voice.py
 - Presence + last-seen persistence (JSON)  -> src/vision.py
 - Shared configuration / state             -> src/config.py / src/state.py

Config via environment variables:
 - MODEL_KITCHEN, MODEL_SPEC: paths to your YOLO models
 - PV_DEVICE_INDEX: integer device index for PvRecorder (optional)
 - ASSEMBLYAI_API_KEY, PICOVOICE_ACCESS_KEY
"""

import os
import threading

from src.config import MODEL_KITCHEN, MODEL_SPEC, KEYWORD_PATH
from src.api import start_api_server
from src.voice import wake_word_listener_loop
from src.vision import camera_loop


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

    # start Flask API server (background thread; dashboard polls it)
    start_api_server()

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


if __name__ == "__main__":
    main()
