from __future__ import annotations

import threading
import time

from modules.E_ml_automl.dataset_service import load_dataset_by_query
from modules.E_ml_automl.code_generator import generate_code_bundle
from modules.E_ml_automl.experiment_runner import run_generated_experiment
from modules.F_stateless_info.kaggle_kernel_service import get_kernel_code


_SIM_INTERVAL = 0.4

# Models that the controller considers valid
SUPPORTED_MODELS = {"xgboost", "random_forest", "logistic_regression", "cnn", "mlp", "resnet"}


class ExperimentController:
    def __init__(self, state_manager):
        self._sm = state_manager
        self._worker = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

    def execute(self, command: dict) -> dict:
        # ── Reject commands with missing or invalid slots ────
        missing = command.get("missing_slots", [])
        invalid = command.get("invalid_slots", {})
        if missing:
            return {"success": False, "message": f"Missing required slot(s): {', '.join(missing)}."}
        if invalid:
            details = "; ".join(f"{k}: {v}" for k, v in invalid.items())
            return {"success": False, "message": f"Invalid slot(s): {details}."}

        intent = command.get("intent")
        slots = command.get("slots", {})

        if intent == "load_dataset":
            return self._handle_load_dataset(slots)
        if intent == "select_model":
            return self._handle_select_model(slots)
        if intent == "set_learning_rate":
            return self._handle_set_learning_rate(slots)
        if intent == "set_batch_size":
            return self._handle_set_batch_size(slots)
        if intent == "set_epochs":
            return self._handle_set_epochs(slots)
        if intent == "set_layers":
            return self._handle_set_layers(slots)
        if intent == "load_code":
            return self._handle_load_code(slots)
        if intent == "run_code":
            return self._handle_run_code()
        if intent == "show_output":
            return self._handle_show_output()
        if intent == "start_training":
            return self._handle_start_training()
        if intent == "pause_training":
            return self._handle_pause_training()
        if intent == "resume_training":
            return self._handle_resume_training()
        if intent == "stop_training":
            return self._handle_stop_training()
        if intent == "show_status":
            return self._handle_show_status()
        if intent == "show_accuracy":
            return self._handle_show_accuracy()
        if intent == "show_loss_curve":
            return self._handle_show_loss_curve()

        return {"success": False, "message": f"No handler for stateful intent '{intent}'."}

    # ── helpers ──────────────────────────────────────────────

    def _is_training(self) -> bool:
        return self._sm.get("training_status") == "training"

    # ── Load Dataset ─────────────────────────────────────────

    def _handle_load_dataset(self, slots: dict) -> dict:
        dataset_query = slots.get("dataset")
        if not dataset_query:
            return {"success": False, "message": "No dataset query provided."}

        if self._is_training():
            return {"success": False, "message": "Stop training before loading a new dataset."}

        result = load_dataset_by_query(dataset_query)
        if not result["success"]:
            return {"success": False, "message": result.get("error", "Failed to load dataset.")}

        self._sm.set_dataset(dataset_query.strip().lower())
        self._sm.set_dataset_info(result["dataset_info"])
        self._sm.set_dataset_preview(result["dataset_preview"])
        self._sm.set_dataset_files(result.get("dataset_files", []))
        self._sm.set_dataset_profile(result.get("dataset_profile", {}))
        suggested = result.get("dataset_profile", {}).get("suggested_model")
        if suggested:
            self._sm.set_model(suggested)
        self._sm.reset_metrics()
        self._sm.set_training_status("idle")
        self._sm.append_log(f"📂 Dataset loaded: {dataset_query}")

        return {
            "success": True,
            "message": f"Loaded dataset '{dataset_query}' into the workspace.",
        }

    # ── Select Model ─────────────────────────────────────────

    def _handle_select_model(self, slots: dict) -> dict:
        model = slots.get("model")
        if not model:
            return {"success": False, "message": "No model provided."}

        if model not in SUPPORTED_MODELS:
            return {"success": False, "message": f"Model '{model}' is not supported. Choose from: {', '.join(sorted(SUPPORTED_MODELS))}."}

        if self._is_training():
            return {"success": False, "message": "Stop training before changing the model."}

        self._sm.set_model(model)
        self._sm.append_log(f"🧠 Model set to {model}")
        return {"success": True, "message": f"Model updated to {model}."}

    # ── Hyperparameters ──────────────────────────────────────

    def _handle_set_learning_rate(self, slots: dict) -> dict:
        lr = slots.get("learning_rate")
        if lr is None or lr <= 0:
            return {"success": False, "message": "Learning rate must be > 0."}
        self._sm.set_learning_rate(float(lr))
        self._sm.append_log(f"⚙️ Learning rate set to {lr}")
        return {"success": True, "message": f"Learning rate updated to {lr}."}

    def _handle_set_batch_size(self, slots: dict) -> dict:
        bs = slots.get("batch_size")
        if bs is None or bs <= 0:
            return {"success": False, "message": "Batch size must be > 0."}
        self._sm.set_batch_size(int(bs))
        self._sm.append_log(f"⚙️ Batch size set to {bs}")
        return {"success": True, "message": f"Batch size updated to {bs}."}

    def _handle_set_epochs(self, slots: dict) -> dict:
        n = slots.get("epochs")
        if n is None or n <= 0:
            return {"success": False, "message": "Epochs must be > 0."}
        self._sm.set_epochs(int(n))
        self._sm.append_log(f"⚙️ Epochs set to {n}")
        return {"success": True, "message": f"Epoch count updated to {n}."}

    def _handle_set_layers(self, slots: dict) -> dict:
        n = slots.get("layers")
        if n is None or n <= 0:
            return {"success": False, "message": "Layers must be > 0."}
        self._sm.set_layers(int(n))
        self._sm.append_log(f"🏗️ Layers set to {n}")
        return {"success": True, "message": f"Layers updated to {n}."}

    # ── Code Generation / Execution ──────────────────────────

    def _handle_load_code(self, slots: dict) -> dict:
        state = self._sm.get_state()
        dataset_query = slots.get("dataset") or state.get("dataset")
        if not dataset_query:
            return {"success": False, "message": "Load a dataset first."}

        bundle = generate_code_bundle(state)
        self._sm.set_generated_code_py(bundle["py_source"])
        self._sm.set_generated_code_ipynb(bundle["ipynb_source"])
        self._sm.append_log("💻 Runnable code generated from current state")

        kernel_result = get_kernel_code(dataset_query)
        if kernel_result.get("success"):
            title = kernel_result["top_result"].get("title") or kernel_result["kernel_ref"]
            self._sm.set_reference_code(
                kernel_result["code_text"],
                fmt=kernel_result["code_format"],
                title=title,
            )
            self._sm.append_log(f"📘 Kaggle reference code pulled: {kernel_result['kernel_ref']}")
            return {
                "success": True,
                "message": f"Generated runnable code and pulled Kaggle reference code for '{dataset_query}'.",
            }

        self._sm.set_reference_code("", "", "")
        return {
            "success": True,
            "message": f"Generated runnable code for '{dataset_query}'. No Kaggle reference notebook was found.",
        }

    def _handle_run_code(self) -> dict:
        state = self._sm.get_state()
        if not state.get("generated_code_py"):
            bundle = generate_code_bundle(state)
            self._sm.set_generated_code_py(bundle["py_source"])
            self._sm.set_generated_code_ipynb(bundle["ipynb_source"])

        result = run_generated_experiment(self._sm.get_state())
        if not result["success"]:
            return {"success": False, "message": "Code execution failed."}

        self._sm.set_code_output_text(result["text_output"])
        self._sm.set_outputs(result["outputs"])
        self._sm.append_log("▶️ Code executed")
        return {"success": True, "message": "Code executed successfully. Output panel updated."}

    def _handle_show_output(self) -> dict:
        outputs = self._sm.get("outputs", [])
        return {
            "success": True,
            "message": f"Output panel has {len(outputs)} item(s).",
        }

    # ── Training ─────────────────────────────────────────────

    def _train_loop(self):
        state = self._sm.get_state()
        total = int(state.get("epochs_total", 10))
        start_epoch = int(state.get("epoch_current", 0))
        for epoch in range(start_epoch + 1, total + 1):
            if self._stop_event.is_set():
                return

            while self._pause_event.is_set():
                if self._stop_event.is_set():
                    return
                time.sleep(0.1)

            time.sleep(_SIM_INTERVAL)

            if self._stop_event.is_set():
                return

            while self._pause_event.is_set():
                if self._stop_event.is_set():
                    return
                time.sleep(0.1)

            progress = epoch / max(total, 1)
            loss = round(max(0.05, 1.0 - 0.85 * progress), 4)
            acc = round(min(0.99, 0.55 + 0.42 * progress), 4)

            self._sm.set_epoch_current(epoch)
            self._sm.append_loss(loss)
            self._sm.append_accuracy(acc)

            if epoch >= total:
                self._sm.set_training_status("completed")
                self._sm.append_log("✅ Training completed")
                return

    def _handle_start_training(self) -> dict:
        state = self._sm.get_state()
        if not state.get("dataset"):
            return {"success": False, "message": "Load a dataset first."}
        if not state.get("model"):
            return {"success": False, "message": "Select a model first."}
        if state.get("training_status") == "training":
            return {"success": False, "message": "Training is already running."}

        self._stop_event.clear()
        self._pause_event.clear()
        self._sm.set_training_status("training")
        self._sm.append_log("🚀 Training started")

        self._worker = threading.Thread(target=self._train_loop, daemon=True)
        self._worker.start()
        return {"success": True, "message": "Training started."}

    def _handle_pause_training(self) -> dict:
        if self._sm.get("training_status") != "training":
            return {"success": False, "message": "Training is not running."}
        self._pause_event.set()
        self._sm.set_training_status("paused")
        self._sm.append_log("⏸️ Training paused")
        return {"success": True, "message": "Training paused."}

    def _handle_resume_training(self) -> dict:
        if self._sm.get("training_status") != "paused":
            return {"success": False, "message": "Training is not paused."}
        self._pause_event.clear()
        self._sm.set_training_status("training")
        self._sm.append_log("▶️ Training resumed")
        return {"success": True, "message": "Training resumed."}

    def _handle_stop_training(self) -> dict:
        status = self._sm.get("training_status")
        if status not in ("training", "paused"):
            return {"success": False, "message": "Training is not running."}
        self._stop_event.set()
        self._pause_event.clear()
        self._sm.set_training_status("stopped")
        self._sm.append_log("🛑 Training stopped")
        return {"success": True, "message": "Training stopped."}

    # ── Status / Metrics ─────────────────────────────────────

    def _handle_show_status(self) -> dict:
        state = self._sm.get_state()
        dataset = state.get("dataset") or "none"
        model = state.get("model") or "none"
        return {
            "success": True,
            "message": (
                f"Status: {state['training_status']}. "
                f"Dataset: {dataset}. Model: {model}. "
                f"Epoch {state['epoch_current']} / {state['epochs_total']}."
            ),
        }

    def _handle_show_accuracy(self) -> dict:
        acc = self._sm.get("accuracy_history", [])
        latest = acc[-1] if acc else None
        if latest is not None:
            return {"success": True, "message": f"Current accuracy is {latest}."}
        return {"success": True, "message": "No accuracy data yet."}

    def _handle_show_loss_curve(self) -> dict:
        loss = self._sm.get("loss_history", [])
        if not loss:
            return {"success": True, "message": "No loss data yet."}
        return {"success": True, "message": f"Loss curve has {len(loss)} point(s)."}
