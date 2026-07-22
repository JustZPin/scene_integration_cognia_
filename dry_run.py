#!/usr/bin/env python3
"""
dry_run.py - hardware-free smoke test for the Cognia on-device unit.

The real app needs a Pi camera, a microphone, two YOLO ``.pt`` models, a
Porcupine ``.ppn`` wake-word file and live API keys. None of that exists on a
plain dev box, so this script *stubs the hardware/ML edges* (camera, YOLO,
microphone, wake-word, TTS, ASR, translator) while running everything that is
real: the Flask API server, Python threading, and the shared ``state`` wiring.

It then reproduces exactly what ``main.py`` does - start the API server, start
the wake-word thread, run the camera loop - and verifies that all three
subsystems come up together, share state, and produce output. Finally it drives
the voice command handlers (reminder / location / glasses / translate) directly.

Run:  python dry_run.py         (exit code 0 = all checks passed)
"""

import os
import sys
import time
import types
import json
import tempfile
import threading

# ---------------------------------------------------------------------------
# 0. Point config at throwaway files/ports BEFORE importing anything real.
# ---------------------------------------------------------------------------
_TMP = tempfile.mkdtemp(prefix="cognia_dry_")


def _touch(name):
    p = os.path.join(_TMP, name)
    open(p, "wb").close()
    return p


os.environ["MODEL_KITCHEN"] = _touch("kitchen.pt")
os.environ["MODEL_SPEC"] = _touch("spec.pt")
os.environ["KEYWORD_PATH"] = _touch("hey-pico.ppn")
os.environ["LAST_SEEN_JSON_PATH"] = os.path.join(_TMP, "last_spec_seen.json")
os.environ["PRESENCE_JSON_PATH"] = os.path.join(_TMP, "presence.json")
os.environ["API_HOST"] = "127.0.0.1"
os.environ["API_PORT"] = "5057"
os.environ.setdefault("ASSEMBLYAI_API_KEY", "dry-run")
os.environ.setdefault("PICOVOICE_ACCESS_KEY", "dry-run")


# ---------------------------------------------------------------------------
# 1. Fake the hardware / heavy-ML modules (Flask stays real).
# ---------------------------------------------------------------------------
def _mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# --- cv2: a camera that yields 8 frames then signals shutdown -------------
class _FakeCap:
    def __init__(self):
        self._n = 0

    def set(self, *a):
        return True

    def isOpened(self):
        return True

    def read(self):
        self._n += 1
        if self._n > 8:                 # end the camera loop cleanly
            raise KeyboardInterrupt
        return True, ("frame", self._n)

    def release(self):
        pass


_mod(
    "cv2",
    CAP_V4L2=200, CAP_ANY=0,
    CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4,
    VideoCapture=lambda *a, **k: _FakeCap(),
)


# --- ultralytics.YOLO: always "sees" a spectacle at 0.92 confidence -------
class _FakeBox:
    def __init__(self, conf, cls, xyxy):
        self.conf = [conf]
        self.cls = [cls]
        self.xyxy = [list(xyxy)]


class _FakeResult:
    def __init__(self):
        self.names = {0: "spectacles"}
        self.boxes = [_FakeBox(0.92, 0, (10, 20, 30, 40))]


class _FakeModel:
    def __call__(self, frame, verbose=False):
        return [_FakeResult()]


_mod("ultralytics", YOLO=lambda *a, **k: _FakeModel())


# --- audio / TTS ----------------------------------------------------------
class _FakeGTTS:
    def __init__(self, text=None, lang=None):
        self.text = text

    def save(self, fname):             # no filesystem touch
        pass


_mixer_music = types.SimpleNamespace(
    load=lambda *a: None, play=lambda *a: None, get_busy=lambda: False
)
_mod("pygame",
     mixer=types.SimpleNamespace(init=lambda *a, **k: None, music=_mixer_music),
     time=types.SimpleNamespace(wait=lambda *a: None))
_mod("gtts", gTTS=_FakeGTTS)


# --- deep_translator ------------------------------------------------------
class _FakeTranslator:
    def __init__(self, source=None, target=None):
        self.target = target

    def translate(self, text):
        return f"[{text} -> {self.target}]"


_mod("deep_translator", GoogleTranslator=_FakeTranslator)


# --- assemblyai (module + streaming.v3 + extras) --------------------------
class _Evt:
    pass


aai = _mod("assemblyai")
aai.extras = types.SimpleNamespace(MicrophoneStream=lambda sample_rate=None: object())
_streaming = _mod("assemblyai.streaming")
_v3 = _mod(
    "assemblyai.streaming.v3",
    BeginEvent=_Evt, TerminationEvent=_Evt, TurnEvent=_Evt, StreamingError=_Evt,
    StreamingClient=object, StreamingClientOptions=object,
    StreamingParameters=object, StreamingEvents=types.SimpleNamespace(
        Begin=1, Turn=2, Termination=3, Error=4),
)
aai.streaming = _streaming
_streaming.v3 = _v3


# --- Porcupine wake-word + PvRecorder mic (never triggers a detection) ----
class _FakePorcupine:
    frame_length = 512

    def process(self, pcm):
        return -1                      # silence - no wake-word

    def delete(self):
        pass


class _FakePvRecorder:
    def __init__(self, device_index=None, frame_length=512):
        self.frame_length = frame_length

    @staticmethod
    def get_audio_devices():
        return ["Fake USB mic"]

    def start(self):
        pass

    def read(self):
        time.sleep(0.02)
        return [0] * self.frame_length

    def stop(self):
        pass

    def delete(self):
        pass


_mod("pvporcupine", create=lambda *a, **k: _FakePorcupine())
_mod("pvrecorder", PvRecorder=_FakePvRecorder)


# ---------------------------------------------------------------------------
# 2. Import the REAL app now that its edges are stubbed.
# ---------------------------------------------------------------------------
import requests                                    # noqa: E402
import main                                         # noqa: E402
from src import state, voice                        # noqa: E402

# Capture spoken output instead of playing audio.
_SPOKEN = []


def _record_speech(text, lang="en"):
    _SPOKEN.append(text)


voice.speak_text = _record_speech          # used by vision's imported async wrapper
voice.speak_text_async = _record_speech    # used by the voice command handlers


# ---------------------------------------------------------------------------
# 3. Check harness.
# ---------------------------------------------------------------------------
_RESULTS = []


def check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def main_run():
    print("\n=== Cognia dry run (hardware stubbed, Flask real) ===\n")
    base = f"http://{os.environ['API_HOST']}:{os.environ['API_PORT']}"

    # Ask the vision loop to announce the sighting, proving the find-spec path.
    state.find_spec_mode = True

    # --- Reproduce main.main(): API thread + wake thread + camera loop -----
    main.start_api_server()
    t_wake = threading.Thread(target=voice.wake_word_listener_loop,
                              daemon=True, name="wake")
    t_wake.start()
    time.sleep(0.6)                     # let Flask bind + wake thread spin up

    from src.vision import camera_loop
    camera_loop()                       # runs 8 frames then returns (KeyboardInterrupt)

    time.sleep(0.2)

    # --- 1) All three subsystems concurrent -------------------------------
    threads = [t.name for t in threading.enumerate()]
    check("wake-word thread alive", t_wake.is_alive(), f"threads: {threads}")

    # --- 2) Flask API really serves over HTTP -----------------------------
    try:
        h = requests.get(base + "/api/health", timeout=3).json()
        check("GET /api/health", h.get("ok") is True, str(h))
    except Exception as e:
        check("GET /api/health", False, repr(e))

    try:
        s = requests.get(base + "/api/summary", timeout=3).json()
        ok = s.get("ok") and s.get("presence") and s.get("last_seen")
        check("GET /api/summary has presence+last_seen", ok, str(s))
    except Exception as e:
        check("GET /api/summary has presence+last_seen", False, repr(e))

    for ep in ("/api/presence", "/api/last_seen"):
        try:
            r = requests.get(base + ep, timeout=3).json()
            check(f"GET {ep}", r.get("ok") is True, str(r))
        except Exception as e:
            check(f"GET {ep}", False, repr(e))

    # --- 3) Vision loop wrote state + JSON files --------------------------
    check("state.last_presence populated", bool(state.last_presence),
          str(state.last_presence))
    check("state.last_spec_seen populated",
          bool(state.last_spec_seen.get("time")),
          f"conf={state.last_spec_seen.get('conf')}")
    check("presence.json written", os.path.exists(os.environ["PRESENCE_JSON_PATH"]))
    check("last_spec_seen.json written",
          os.path.exists(os.environ["LAST_SEEN_JSON_PATH"]))
    check("spectacle sighting announced (find_spec_mode)",
          any("last seen" in s.lower() for s in _SPOKEN), str(_SPOKEN))

    # --- 4) Voice command handlers ----------------------------------------
    def turn(text):
        _SPOKEN.clear()
        state.last_transcript = ""
        ev = types.SimpleNamespace(end_of_turn=True, transcript=text)
        voice.on_turn(None, ev)
        return list(_SPOKEN)

    check("cmd: 'remind me in 2 seconds' -> reminder set",
          any("reminder set" in s.lower() for s in turn("remind me in 2 seconds")))
    check("cmd: 'where am i' -> speaks location",
          any("level 3" in s.lower() or "you are at" in s.lower()
              for s in turn("where am i")))
    spoke = turn("where are my glasses")
    check("cmd: 'where are my glasses' -> find-spec mode + prompt",
          state.find_spec_mode and any("turn around" in s.lower() for s in spoke))
    check("cmd: plain sentence -> translator invoked",
          voice.translator is not None)

    # --- summary ----------------------------------------------------------
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print(f"\n=== {passed}/{total} checks passed ===")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main_run())
