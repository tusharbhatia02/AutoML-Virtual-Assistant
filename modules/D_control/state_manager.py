from __future__ import annotations

import threading
from copy import deepcopy
from datetime import datetime

try:
    from config import DEFAULT_LEARNING_RATE, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS
except Exception:
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_EPOCHS = 10


class StateManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = {}
        self.reset_all()

    def reset_all(self) -> None:
        with self._lock:
            self._state = {
                "verified": False,
                "wake_detected": False,
                "listening": False,
                "transcript": "",
                "assistant_response": "",

                "dataset": None,
                "dataset_info": {},
                "dataset_preview": [],
                "dataset_files": [],
                "dataset_profile": {},

                "model": None,
                "learning_rate": DEFAULT_LEARNING_RATE,
                "batch_size": DEFAULT_BATCH_SIZE,
                "epochs_total": DEFAULT_EPOCHS,
                "epoch_current": 0,
                "layers": 3,

                "training_status": "idle",
                "loss_history": [],
                "accuracy_history": [],

                "generated_code_py": "",
                "generated_code_ipynb": "",
                "reference_code": "",
                "reference_code_format": "",
                "code_title": "",
                "code_output_text": "",
                "outputs": [],

                "stateless_results": [],
                "event_log": [],
            }

    def reset_experiment(self) -> None:
        with self._lock:
            self._state["dataset"] = None
            self._state["dataset_info"] = {}
            self._state["dataset_preview"] = []
            self._state["dataset_files"] = []
            self._state["dataset_profile"] = {}
            self._state["model"] = None
            self._state["learning_rate"] = DEFAULT_LEARNING_RATE
            self._state["batch_size"] = DEFAULT_BATCH_SIZE
            self._state["epochs_total"] = DEFAULT_EPOCHS
            self._state["epoch_current"] = 0
            self._state["layers"] = 3
            self._state["training_status"] = "idle"
            self._state["loss_history"] = []
            self._state["accuracy_history"] = []
            self._state["generated_code_py"] = ""
            self._state["generated_code_ipynb"] = ""
            self._state["reference_code"] = ""
            self._state["reference_code_format"] = ""
            self._state["code_title"] = ""
            self._state["code_output_text"] = ""
            self._state["outputs"] = []
            self._state["stateless_results"] = []

    def get_state(self) -> dict:
        with self._lock:
            return deepcopy(self._state)

    def get(self, key: str, default=None):
        with self._lock:
            return deepcopy(self._state.get(key, default))

    def get_ui_state(self) -> dict:
        """Return a compact dict suitable for UI display."""
        with self._lock:
            loss = list(self._state["loss_history"])
            acc = list(self._state["accuracy_history"])
            return {
                "dataset": self._state["dataset"],
                "model": self._state["model"],
                "learning_rate": self._state["learning_rate"],
                "batch_size": self._state["batch_size"],
                "epochs_total": self._state["epochs_total"],
                "epoch_current": self._state["epoch_current"],
                "layers": self._state["layers"],
                "training_status": self._state["training_status"],
                "loss_latest": loss[-1] if loss else None,
                "accuracy_latest": acc[-1] if acc else None,
                "n_events": len(self._state["event_log"]),
            }

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

    def set_dataset(self, name: str) -> None:
        with self._lock:
            self._state["dataset"] = name

    def set_dataset_info(self, info: dict) -> None:
        with self._lock:
            self._state["dataset_info"] = info

    def set_dataset_preview(self, preview) -> None:
        with self._lock:
            self._state["dataset_preview"] = preview

    def set_dataset_files(self, files) -> None:
        with self._lock:
            self._state["dataset_files"] = files

    def set_dataset_profile(self, profile: dict) -> None:
        with self._lock:
            self._state["dataset_profile"] = profile

    def set_model(self, name: str) -> None:
        with self._lock:
            self._state["model"] = name

    def set_learning_rate(self, lr: float) -> None:
        with self._lock:
            self._state["learning_rate"] = lr

    def set_batch_size(self, bs: int) -> None:
        with self._lock:
            self._state["batch_size"] = bs

    def set_epochs(self, n: int) -> None:
        with self._lock:
            self._state["epochs_total"] = n

    def set_layers(self, n: int) -> None:
        with self._lock:
            self._state["layers"] = n

    def set_training_status(self, status: str) -> None:
        with self._lock:
            self._state["training_status"] = status

    def set_epoch_current(self, n: int) -> None:
        with self._lock:
            self._state["epoch_current"] = n

    def append_loss(self, value: float) -> None:
        with self._lock:
            self._state["loss_history"].append(value)

    def append_accuracy(self, value: float) -> None:
        with self._lock:
            self._state["accuracy_history"].append(value)

    def reset_metrics(self) -> None:
        with self._lock:
            self._state["loss_history"] = []
            self._state["accuracy_history"] = []
            self._state["epoch_current"] = 0

    def set_generated_code_py(self, code: str) -> None:
        with self._lock:
            self._state["generated_code_py"] = code

    def set_generated_code_ipynb(self, code: str) -> None:
        with self._lock:
            self._state["generated_code_ipynb"] = code

    def set_reference_code(self, code: str, fmt: str = "", title: str = "") -> None:
        with self._lock:
            self._state["reference_code"] = code
            self._state["reference_code_format"] = fmt
            self._state["code_title"] = title

    def set_code_output_text(self, text: str) -> None:
        with self._lock:
            self._state["code_output_text"] = text

    def set_outputs(self, outputs) -> None:
        with self._lock:
            self._state["outputs"] = outputs

    def set_stateless_results(self, results) -> None:
        with self._lock:
            self._state["stateless_results"] = results

    def append_log(self, message: str) -> None:
        with self._lock:
            ts = datetime.now().strftime("%H:%M:%S")
            self._state["event_log"].append(f"[{ts}] {message}")

    def get_event_log(self, last_n: int | None = None):
        with self._lock:
            logs = list(self._state["event_log"])
        return logs[-last_n:] if last_n else logs

    def clear_event_log(self) -> None:
        with self._lock:
            self._state["event_log"] = []


_instance = None
_instance_lock = threading.Lock()

def get_state_manager() -> StateManager:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = StateManager()
    return _instance