"""
api.py

Small Flask API the caregiver dashboard polls for live status.
Runs in a background thread; reads the shared state written by the vision loop.

Endpoints:
  GET /api/health    -> liveness + whether the JSON files exist
  GET /api/last_seen -> last spectacle sighting (place / time / conf)
  GET /api/presence  -> current presence (location / reason)
  GET /api/summary   -> both of the above in one call
"""

import os
import time
import threading

from . import state
from .config import API_HOST, API_PORT, LAST_SEEN_JSON_PATH, PRESENCE_JSON_PATH

try:
    from flask import Flask, jsonify
    from flask_cors import CORS
except Exception as e:
    print("[ERROR] Flask import failed:", e)
    raise

api_app = Flask(__name__)
CORS(api_app, resources={r"/api/*": {"origins": "*"}})


def _with_time_iso(data):
    if data and data.get('time') and not data.get('time_iso'):
        data['time_iso'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data['time']))
    return data


@api_app.get('/api/health')
def api_health():
    return jsonify({
        "ok": True,
        "last_seen_file": os.path.exists(LAST_SEEN_JSON_PATH),
        "presence_file": os.path.exists(PRESENCE_JSON_PATH),
    })


@api_app.get('/api/last_seen')
def api_last_seen():
    with state.state_lock:
        data = dict(state.last_spec_seen) if state.last_spec_seen else None
    return jsonify({"ok": data is not None, "last_seen": _with_time_iso(data)})


@api_app.get('/api/presence')
def api_presence():
    with state.state_lock:
        data = dict(state.last_presence) if state.last_presence else None
    return jsonify({"ok": data is not None, "presence": data})


@api_app.get('/api/summary')
def api_summary():
    with state.state_lock:
        ls = dict(state.last_spec_seen) if state.last_spec_seen else None
        pr = dict(state.last_presence) if state.last_presence else None
    return jsonify({"ok": bool(ls or pr), "last_seen": _with_time_iso(ls), "presence": pr})


def start_api_server():
    """Launch the Flask app in a daemon thread."""
    threading.Thread(
        target=lambda: api_app.run(host=API_HOST, port=API_PORT, debug=False,
                                   use_reloader=False, threaded=True),
        daemon=True,
    ).start()
    print(f"[API] listening at http://{API_HOST}:{API_PORT}")
