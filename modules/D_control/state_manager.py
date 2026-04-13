"""
Module 9 — State Manager
==========================
Central memory and source of truth for the entire AutoML assistant.

This is the most important module in the backend.  It answers:
"What is the current condition of the system right now?"

All modules read from and write to this shared state object, ensuring
consistency between the backend controller, Streamlit UI, and event logs.

Public API
----------
StateManager  (class — singleton via get_state_manager())
    .get_state()            -> dict     — full state snapshot
    .get_ui_state()         -> dict     — UI-friendly subset
    .set_dataset(name)      -> None
    .set_model(name)        -> None
    .set_learning_rate(lr)  -> None
    .set_batch_size(bs)     -> None
    .set_epochs(n)          -> None
    .set_training_status(s) -> None
    .set_epoch_current(n)   -> None
    .append_loss(v)         -> None
    .append_accuracy(v)     -> None
    .append_log(msg)        -> None
    .reset_experiment()     -> None
    .reset_all()            -> None

get_state_manager() -> StateManager
    Returns the singleton instance.

Design notes
------------
- Thread-safe via threading.Lock (training simulation runs in a thread).
- Default values from config.py for consistency.
- Event log entries are timestamped automatically.
- reset_experiment() preserves verification/wake state but clears
  experiment config, useful when loading a new dataset.
"""

import threading
from datetime import datetime
from copy import deepcopy

from config import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
)


class StateManager:
    """
    Centralised, thread-safe state store for the AutoML assistant.

    All experiment configuration, training metrics, and event logs
    are managed through this single object.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state: dict = {}
        self.reset_all()

    # ─────────────────────────────────────────────────────
    # FULL RESET
    # ─────────────────────────────────────────────────────

    def reset_all(self) -> None:
        """Reset the entire state to initial defaults."""
        with self._lock:
            self._state = {
                # ── Session / pipeline state ──────────────
                "verified":           False,
                "wake_detected":      False,
                "listening":          False,
                "transcript":         "",
                "assistant_response": "",

                # ── Experiment configuration ──────────────
                "dataset":            None,
                "model":              None,
                "learning_rate":      DEFAULT_LEARNING_RATE,
                "batch_size":         DEFAULT_BATCH_SIZE,
                "epochs_total":       DEFAULT_EPOCHS,
                "epoch_current":      0,

                # ── Training runtime ─────────────────────
                "training_status":    "idle",   # idle | training | paused | stopped | completed
                "loss_history":       [],
                "accuracy_history":   [],

                # ── Logs ──────────────────────────────────
                "event_log":          [],
            }

    def reset_experiment(self) -> None:
        """
        Reset experiment-specific state (dataset, model, hyperparams,
        metrics) but preserve session/pipeline state (verified, wake, etc.).
        """
        with self._lock:
            self._state["dataset"]         = None
            self._state["model"]           = None
            self._state["learning_rate"]   = DEFAULT_LEARNING_RATE
            self._state["batch_size"]      = DEFAULT_BATCH_SIZE
            self._state["epochs_total"]    = DEFAULT_EPOCHS
            self._state["epoch_current"]   = 0
            self._state["training_status"] = "idle"
            self._state["loss_history"]    = []
            self._state["accuracy_history"] = []

    # ─────────────────────────────────────────────────────
    # GETTERS
    # ─────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """Return a deep copy of the full state."""
        with self._lock:
            return deepcopy(self._state)

    def get_ui_state(self) -> dict:
        """Return a UI-friendly state snapshot (no deep internal lists)."""
        with self._lock:
            return {
                "dataset":         self._state["dataset"],
                "model":           self._state["model"],
                "learning_rate":   self._state["learning_rate"],
                "batch_size":      self._state["batch_size"],
                "epochs_total":    self._state["epochs_total"],
                "epoch_current":   self._state["epoch_current"],
                "training_status": self._state["training_status"],
                "loss_latest":     self._state["loss_history"][-1] if self._state["loss_history"] else None,
                "accuracy_latest": self._state["accuracy_history"][-1] if self._state["accuracy_history"] else None,
                "n_events":        len(self._state["event_log"]),
            }

    def get(self, key: str, default=None):
        """Get a single state value by key."""
        with self._lock:
            return deepcopy(self._state.get(key, default))

    # ─────────────────────────────────────────────────────
    # SETTERS — Session / Pipeline
    # ─────────────────────────────────────────────────────

    def set_verified(self, value: bool) -> None:
        with self._lock:
            self._state["verified"] = value

    def set_wake_detected(self, value: bool) -> None:
        with self._lock:
            self._state["wake_detected"] = value

    def set_listening(self, value: bool) -> None:
        with self._lock:
            self._state["listening"] = value

    def set_transcript(self, text: str) -> None:
        with self._lock:
            self._state["transcript"] = text

    def set_assistant_response(self, text: str) -> None:
        with self._lock:
            self._state["assistant_response"] = text

    # ─────────────────────────────────────────────────────
    # SETTERS — Experiment Configuration
    # ─────────────────────────────────────────────────────

    def set_dataset(self, name: str) -> None:
        """Set the active dataset name."""
        with self._lock:
            self._state["dataset"] = name

    def set_model(self, name: str) -> None:
        """Set the active model name."""
        with self._lock:
            self._state["model"] = name

    def set_learning_rate(self, lr: float) -> None:
        """Set the learning rate."""
        with self._lock:
            self._state["learning_rate"] = lr

    def set_batch_size(self, bs: int) -> None:
        """Set the batch size."""
        with self._lock:
            self._state["batch_size"] = bs

    def set_epochs(self, n: int) -> None:
        """Set the total number of epochs."""
        with self._lock:
            self._state["epochs_total"] = n

    # ─────────────────────────────────────────────────────
    # SETTERS — Training Runtime
    # ─────────────────────────────────────────────────────

    def set_training_status(self, status: str) -> None:
        """Set training status (idle, training, paused, stopped, completed)."""
        with self._lock:
            self._state["training_status"] = status

    def set_epoch_current(self, n: int) -> None:
        """Set the current epoch counter."""
        with self._lock:
            self._state["epoch_current"] = n

    def append_loss(self, value: float) -> None:
        """Append a loss value to history."""
        with self._lock:
            self._state["loss_history"].append(value)

    def append_accuracy(self, value: float) -> None:
        """Append an accuracy value to history."""
        with self._lock:
            self._state["accuracy_history"].append(value)

    def reset_metrics(self) -> None:
        """Clear loss/accuracy history and reset epoch counter."""
        with self._lock:
            self._state["loss_history"] = []
            self._state["accuracy_history"] = []
            self._state["epoch_current"] = 0

    # ─────────────────────────────────────────────────────
    # EVENT LOG
    # ─────────────────────────────────────────────────────

    def append_log(self, message: str) -> None:
        """Append a timestamped event to the log."""
        with self._lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self._state["event_log"].append(f"[{timestamp}] {message}")

    def get_event_log(self, last_n: int | None = None) -> list[str]:
        """Return the event log (optionally only the last N entries)."""
        with self._lock:
            if last_n is not None:
                return list(self._state["event_log"][-last_n:])
            return list(self._state["event_log"])

    def clear_event_log(self) -> None:
        """Clear all event log entries."""
        with self._lock:
            self._state["event_log"] = []


# ─────────────────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────────────────

_instance: StateManager | None = None
_instance_lock = threading.Lock()


def get_state_manager() -> StateManager:
    """Return the singleton StateManager instance."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = StateManager()
    return _instance
