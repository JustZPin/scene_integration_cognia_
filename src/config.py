"""
config.py

Central configuration for the headless scene-integration app.

All values can be overridden via environment variables:
 - MODEL_KITCHEN, MODEL_SPEC: paths to your YOLO models
 - PV_DEVICE_INDEX: integer device index for PvRecorder (optional)
 - ASSEMBLYAI_API_KEY, PICOVOICE_ACCESS_KEY
"""

import os

# Load variables from a local ".env" file if present (keeps secrets out of git).
# Optional: if python-dotenv isn't installed, env vars still work normally.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Ensure no GUI backend is requested (must happen before cv2 / YOLO import)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# -------------------------
# CONFIG (edit / override via env)
# -------------------------
MODEL_KITCHEN = os.environ.get("MODEL_KITCHEN", "/home/zhipin/Documents/scene_integration/my_model.pt")
MODEL_SPEC = os.environ.get("MODEL_SPEC", "/home/zhipin/Documents/scene_integration/my_model_spec.pt")
LAST_SEEN_JSON_PATH = os.environ.get("LAST_SEEN_JSON_PATH", "last_spec_seen.json")
PRESENCE_JSON_PATH = os.environ.get("PRESENCE_JSON_PATH", "presence.json")

SPEC_EVERY_N = int(os.environ.get("SPEC_EVERY_N", "3"))
PRESENCE_PUSH_INTERVAL = float(os.environ.get("PRESENCE_PUSH_INTERVAL", "15"))

# Secrets: provide these via a local .env file or environment variables.
# Do NOT hardcode real keys here — this file is committed to git.
ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")
PICOVOICE_ACCESS_KEY = os.environ.get("PICOVOICE_ACCESS_KEY", "")
KEYWORD_PATH = os.environ.get("KEYWORD_PATH", "/home/zhipin/Documents/scene_integration/hey-pico_en_raspberry-pi_v3_0_0.ppn")

# optional: set PV device index via env; if unset, we'll try a fallback approach
PV_DEVICE_INDEX = os.environ.get("PV_DEVICE_INDEX", None)
if PV_DEVICE_INDEX is not None:
    try:
        PV_DEVICE_INDEX = int(PV_DEVICE_INDEX)
    except Exception:
        PV_DEVICE_INDEX = None

# Other settings
SPECTACLE_LABELS = {'spectacle', 'glasses', 'spectacles', 'eyeglasses', 'sunglasses'}
SPEC_THRESHOLD = float(os.environ.get("SPEC_THRESHOLD", "0.60"))

# -------------------------
# Flask API (caregiver dashboard reads these endpoints)
# -------------------------
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "5000"))

# -------------------------
# Voiceflow (optional conversational agent) — secret via env/.env
# -------------------------
VOICEFLOW_API_KEY = os.environ.get("VOICEFLOW_API_KEY", "")
VOICEFLOW_USER_ID = os.environ.get("VOICEFLOW_USER_ID", "raspi")
VOICEFLOW_VERSION_ID = os.environ.get("VOICEFLOW_VERSION_ID", "production")
VOICEFLOW_URL = f"https://general-runtime.voiceflow.com/state/user/{VOICEFLOW_USER_ID}/interact"

# Translation target language for the fallback translator (EN -> this)
TRANSLATE_TARGET = os.environ.get("TRANSLATE_TARGET", "ms")
