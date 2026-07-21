"""
state.py

Shared, mutable application state.

NOTE: import this module as a whole (``from . import state``) and read/write
attributes as ``state.find_spec_mode`` etc. Do NOT do
``from .state import find_spec_mode``, because rebinding a value imported by name
would not be visible to other modules.
"""

import threading
from collections import deque

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
