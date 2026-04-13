"""
Module 10 — Experiment Controller
====================================
Executes control commands on the ML experiment.  This is the fulfilment
layer for stateful experiment commands — it takes the {intent, slots}
command from Module C, validates prerequisites, applies changes through
the State Manager (Module 9), and returns structured result metadata.

Public API
----------
ExperimentController  (class)
    .execute(command: dict) -> dict
        Main entry — dispatches the command to the correct handler.

Each handler returns:
    {"success": bool, "message": str}

Design notes
------------
- Business-logic rules are enforced here (not in State Manager or Router).
  Example: cannot start training if no dataset is loaded.
- Training simulation runs in a background thread, producing fake
  loss/accuracy curves that decrease/increase over epochs.
- The controller never writes to state directly — it always goes
  through State Manager's setter methods to ensure thread safety.
"""

import math
import random
import threading
import time

from modules.D_control.state_manager import StateManager, get_state_manager
from config import SUPPORTED_MODELS

# Simulation speed: seconds between simulated epochs
_SIMULATION_EPOCH_INTERVAL = 0.5


class ExperimentController:
    """
    Handles execution of stateful experiment commands.

    Usage
    -----
    controller = ExperimentController()
    result = controller.execute({
        "intent": "set_learning_rate",
        "slots": {"learning_rate": 0.01}
    })
    # result == {"success": True, "message": "Learning rate updated to 0.01"}
    """

    def __init__(self, state_manager: StateManager | None = None):
        self._sm = state_manager or get_state_manager()
        self._training_thread: threading.Thread | None = None
        self._stop_training_event = threading.Event()
        self._pause_training_event = threading.Event()

        # Intent → handler dispatch table
        self._handlers: dict[str, callable] = {
            "load_dataset":      self._handle_load_dataset,
            "select_model":      self._handle_select_model,
            "set_learning_rate": self._handle_set_learning_rate,
            "set_batch_size":    self._handle_set_batch_size,
            "set_epochs":        self._handle_set_epochs,
            "start_training":    self._handle_start_training,
            "pause_training":    self._handle_pause_training,
            "stop_training":     self._handle_stop_training,
            "resume_training":   self._handle_resume_training,
            "show_status":       self._handle_show_status,
            "show_accuracy":     self._handle_show_accuracy,
            "show_loss_curve":   self._handle_show_loss_curve,
        }

    # ─────────────────────────────────────────────────────
    # MAIN DISPATCHER
    # ─────────────────────────────────────────────────────

    def execute(self, command: dict) -> dict:
        """
        Execute a structured command.

        Parameters
        ----------
        command : dict — must contain "intent" and "slots" keys

        Returns
        -------
        dict with "success" (bool) and "message" (str)
        """
        intent = command.get("intent", "unknown_intent")
        slots = command.get("slots", {})
        missing = command.get("missing_slots", [])
        invalid = command.get("invalid_slots", {})

        # Check for missing required slots
        if missing:
            return {
                "success": False,
                "message": f"Missing required parameter(s): {', '.join(missing)}. "
                           f"Please specify: {', '.join(missing)}.",
            }

        # Check for invalid slot values
        if invalid:
            errors = "; ".join(f"{k}: {v}" for k, v in invalid.items())
            return {
                "success": False,
                "message": f"Invalid parameter(s): {errors}",
            }

        handler = self._handlers.get(intent)
        if handler is None:
            return {
                "success": False,
                "message": f"No handler for intent '{intent}'.",
            }

        return handler(slots)

    # ─────────────────────────────────────────────────────
    # HANDLERS — Data
    # ─────────────────────────────────────────────────────

    def _handle_load_dataset(self, slots: dict) -> dict:
        dataset = slots.get("dataset")
        if not dataset:
            return {"success": False, "message": "No dataset name provided."}

        # Reset metrics when loading a new dataset
        state = self._sm.get_state()
        if state["training_status"] == "training":
            return {
                "success": False,
                "message": "Cannot load a new dataset while training is in progress. "
                           "Stop training first.",
            }

        self._sm.set_dataset(dataset)
        self._sm.reset_metrics()
        self._sm.set_training_status("idle")
        self._sm.append_log(f"📂 Dataset loaded: {dataset}")
        return {"success": True, "message": f"Dataset '{dataset}' loaded successfully."}

    # ─────────────────────────────────────────────────────
    # HANDLERS — Model
    # ─────────────────────────────────────────────────────

    def _handle_select_model(self, slots: dict) -> dict:
        model = slots.get("model")
        if not model:
            return {"success": False, "message": "No model name provided."}

        if model not in SUPPORTED_MODELS:
            return {
                "success": False,
                "message": f"Model '{model}' is not supported. "
                           f"Choose from: {', '.join(SUPPORTED_MODELS)}.",
            }

        state = self._sm.get_state()
        if state["training_status"] == "training":
            return {
                "success": False,
                "message": "Cannot change model while training is in progress. "
                           "Stop training first.",
            }

        self._sm.set_model(model)
        self._sm.append_log(f"🤖 Model selected: {model}")
        return {"success": True, "message": f"Model set to '{model}'."}

    # ─────────────────────────────────────────────────────
    # HANDLERS — Hyperparameters
    # ─────────────────────────────────────────────────────

    def _handle_set_learning_rate(self, slots: dict) -> dict:
        lr = slots.get("learning_rate")
        if lr is None:
            return {"success": False, "message": "No learning rate value provided."}

        self._sm.set_learning_rate(lr)
        self._sm.append_log(f"📐 Learning rate set to {lr}")
        return {"success": True, "message": f"Learning rate updated to {lr}."}

    def _handle_set_batch_size(self, slots: dict) -> dict:
        bs = slots.get("batch_size")
        if bs is None:
            return {"success": False, "message": "No batch size value provided."}

        self._sm.set_batch_size(bs)
        self._sm.append_log(f"📦 Batch size set to {bs}")
        return {"success": True, "message": f"Batch size updated to {bs}."}

    def _handle_set_epochs(self, slots: dict) -> dict:
        epochs = slots.get("epochs")
        if epochs is None:
            return {"success": False, "message": "No epoch count provided."}

        self._sm.set_epochs(epochs)
        self._sm.append_log(f"🔄 Total epochs set to {epochs}")
        return {"success": True, "message": f"Total epochs set to {epochs}."}

    # ─────────────────────────────────────────────────────
    # HANDLERS — Training Control
    # ─────────────────────────────────────────────────────

    def _handle_start_training(self, slots: dict) -> dict:
        state = self._sm.get_state()

        # Prerequisite checks
        if not state["dataset"]:
            return {
                "success": False,
                "message": "Cannot start training: no dataset loaded. "
                           "Use 'load dataset <name>' first.",
            }
        if not state["model"]:
            return {
                "success": False,
                "message": "Cannot start training: no model selected. "
                           "Use 'select model <name>' first.",
            }
        if state["training_status"] == "training":
            return {
                "success": False,
                "message": "Training is already in progress.",
            }

        # Reset metrics for a fresh training run
        self._sm.reset_metrics()
        self._sm.set_training_status("training")
        self._sm.append_log(
            f"🚀 Training started: model={state['model']}, "
            f"dataset={state['dataset']}, lr={state['learning_rate']}, "
            f"batch_size={state['batch_size']}, epochs={state['epochs_total']}"
        )

        # Start simulation in background thread
        self._stop_training_event.clear()
        self._pause_training_event.clear()
        self._training_thread = threading.Thread(
            target=self._training_simulation_loop,
            daemon=True,
        )
        self._training_thread.start()

        return {
            "success": True,
            "message": f"Training started! "
                       f"Model: {state['model']}, Dataset: {state['dataset']}, "
                       f"Epochs: {state['epochs_total']}.",
        }

    def _handle_pause_training(self, slots: dict) -> dict:
        state = self._sm.get_state()
        if state["training_status"] != "training":
            return {
                "success": False,
                "message": f"Cannot pause: training is not running "
                           f"(current status: {state['training_status']}).",
            }

        self._pause_training_event.set()
        self._sm.set_training_status("paused")
        self._sm.append_log(
            f"⏸️ Training paused at epoch {state['epoch_current']}"
        )
        return {
            "success": True,
            "message": f"Training paused at epoch {state['epoch_current']}.",
        }

    def _handle_stop_training(self, slots: dict) -> dict:
        state = self._sm.get_state()
        if state["training_status"] not in ("training", "paused"):
            return {
                "success": False,
                "message": f"Cannot stop: training is not active "
                           f"(current status: {state['training_status']}).",
            }

        self._stop_training_event.set()
        self._pause_training_event.set()   # unblock if paused
        self._sm.set_training_status("stopped")
        self._sm.append_log(
            f"⏹️ Training stopped at epoch {state['epoch_current']}"
        )
        return {
            "success": True,
            "message": f"Training stopped at epoch {state['epoch_current']}.",
        }

    def _handle_resume_training(self, slots: dict) -> dict:
        state = self._sm.get_state()
        if state["training_status"] != "paused":
            return {
                "success": False,
                "message": f"Cannot resume: training is not paused "
                           f"(current status: {state['training_status']}).",
            }

        self._pause_training_event.clear()
        self._sm.set_training_status("training")
        self._sm.append_log(
            f"▶️ Training resumed from epoch {state['epoch_current']}"
        )
        return {
            "success": True,
            "message": f"Training resumed from epoch {state['epoch_current']}.",
        }

    # ─────────────────────────────────────────────────────
    # HANDLERS — Status / Metrics
    # ─────────────────────────────────────────────────────

    def _handle_show_status(self, slots: dict) -> dict:
        state = self._sm.get_state()
        lines = [
            f"Dataset:         {state['dataset'] or '(none)'}",
            f"Model:           {state['model'] or '(none)'}",
            f"Learning Rate:   {state['learning_rate']}",
            f"Batch Size:      {state['batch_size']}",
            f"Epochs:          {state['epoch_current']} / {state['epochs_total']}",
            f"Status:          {state['training_status']}",
        ]
        if state["loss_history"]:
            lines.append(f"Latest Loss:     {state['loss_history'][-1]:.4f}")
        if state["accuracy_history"]:
            lines.append(f"Latest Accuracy: {state['accuracy_history'][-1]:.4f}")
        summary = "\n".join(lines)
        return {"success": True, "message": summary}

    def _handle_show_accuracy(self, slots: dict) -> dict:
        state = self._sm.get_state()
        if not state["accuracy_history"]:
            return {
                "success": True,
                "message": "No accuracy data available yet. Start training first.",
            }
        latest = state["accuracy_history"][-1]
        return {
            "success": True,
            "message": f"Current accuracy: {latest:.4f} "
                       f"(epoch {state['epoch_current']}/{state['epochs_total']}).",
        }

    def _handle_show_loss_curve(self, slots: dict) -> dict:
        state = self._sm.get_state()
        if not state["loss_history"]:
            return {
                "success": True,
                "message": "No loss data available yet. Start training first.",
            }
        latest = state["loss_history"][-1]
        n_points = len(state["loss_history"])
        return {
            "success": True,
            "message": f"Loss curve has {n_points} data points. "
                       f"Latest loss: {latest:.4f}.",
        }

    # ─────────────────────────────────────────────────────
    # TRAINING SIMULATION
    # ─────────────────────────────────────────────────────

    def _training_simulation_loop(self) -> None:
        """
        Background thread that simulates a training loop.

        Produces realistic-looking loss (decreasing) and accuracy
        (increasing) curves using exponential decay + noise.
        """
        state = self._sm.get_state()
        total_epochs = state["epochs_total"]
        lr = state["learning_rate"]

        for epoch in range(1, total_epochs + 1):
            # Check stop signal
            if self._stop_training_event.is_set():
                return

            # Check pause signal — block until unpaused or stopped
            while self._pause_training_event.is_set():
                if self._stop_training_event.is_set():
                    return
                time.sleep(0.1)

            # Re-check after potential pause
            if self._stop_training_event.is_set():
                return

            # Simulate epoch computation time
            time.sleep(_SIMULATION_EPOCH_INTERVAL)

            # Generate realistic metrics
            progress = epoch / total_epochs
            # Loss: starts ~2.0, decays exponentially with noise
            base_loss = 2.0 * math.exp(-3.0 * progress)
            noise = random.gauss(0, 0.02 * (1 - progress + 0.1))
            loss = max(0.001, base_loss + noise)

            # Accuracy: starts ~0.3, increases with diminishing returns
            base_acc = 0.3 + 0.65 * (1 - math.exp(-4.0 * progress))
            acc_noise = random.gauss(0, 0.015 * (1 - progress + 0.1))
            accuracy = min(0.999, max(0.05, base_acc + acc_noise))

            # Update state
            self._sm.set_epoch_current(epoch)
            self._sm.append_loss(round(loss, 4))
            self._sm.append_accuracy(round(accuracy, 4))

        # Training completed successfully
        if not self._stop_training_event.is_set():
            self._sm.set_training_status("completed")
            state = self._sm.get_state()
            final_loss = state["loss_history"][-1] if state["loss_history"] else "N/A"
            final_acc = state["accuracy_history"][-1] if state["accuracy_history"] else "N/A"
            self._sm.append_log(
                f"✅ Training completed! "
                f"Final loss: {final_loss}, Final accuracy: {final_acc}"
            )
