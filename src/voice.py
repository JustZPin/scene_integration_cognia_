"""
voice.py

The full voice stack, in the order audio flows through it:

  1. TTS output          -> speak_text / speak_text_async  (gTTS + pygame)
  2. Streaming ASR       -> AssemblyAI handlers + start_assembly_ai
  3. Wake-word listener  -> Porcupine loop (hands off to the ASR session)

NOTE: ``on_turn`` references ``call_voiceflow``, ``speak_voiceflow_response`` and
``translator``, which are not defined in this project (pre-existing in the
original code). Those code paths will raise if reached; behavior is preserved
here intentionally.
"""

import os
import re
import json
import time
import threading

from . import state
from .config import (
    ASSEMBLYAI_API_KEY, PRESENCE_JSON_PATH,
    PICOVOICE_ACCESS_KEY, KEYWORD_PATH, PV_DEVICE_INDEX,
)

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


# =========================================================
# TTS (gTTS + pygame)
# =========================================================
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


# =========================================================
# AssemblyAI streaming ASR
# =========================================================
def on_begin(client, event: BeginEvent):
    state.wake_active = True
    state.last_transcript = ""
    print("[ASR] session started")
    speak_text_async("I'm listening.", 'en')


def parse_reminder(text):
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
    # Ignore partial empty transcripts
    if not event.end_of_turn:
        return

    text = event.transcript or ""
    text = text.strip()

    # Ignore empty noise responses
    if text == "":
        return

    # Ignore duplicates
    if text.lower() == state.last_transcript.lower():
        return

    state.last_transcript = text

    print("[ASR]", text)
    low = text.lower()

    # --- Control voiceflow ---
    if low == "start":
        state.voiceflow_mode = True
        speak_text_async("Voiceflow activated.", 'en')
        return

    if low == "exit voiceflow":
        state.voiceflow_mode = False
        speak_text_async("Voiceflow deactivated.", 'en')
        return

    if state.voiceflow_mode:
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
        state.find_spec_mode = True
        speak_text_async("Turn around to check.", 'en')
        return

    # --- Fallback translation ---
    try:
        translated = translator.translate(text)
        print("[TRANSLATION]", translated)
    except:
        pass


def on_terminated(client, event: TerminationEvent):
    state.wake_active = False
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


# =========================================================
# Wake-word (Porcupine) loop
# =========================================================
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
