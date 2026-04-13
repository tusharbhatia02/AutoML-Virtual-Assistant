"""
Module 8 — Command Router
===========================
Central dispatcher that takes the structured command JSON from
Module C (NLU Pipeline) and routes it to the correct backend handler.

This is the single unified entry point for all actions.  Instead of
Streamlit or ASR calling random backend functions directly, everything
passes through:

    route_command(command_json) -> dict

The router determines:
1. Whether the command is stateful (experiment control) or stateless
   (information query) or utility (help/repeat).
2. Dispatches to the correct handler module.
3. Returns a structured result for the UI.

Public API
----------
CommandRouter  (class)
    .route(command: dict) -> dict

route_command(command: dict) -> dict
    Module-level convenience wrapper using singleton router.

Design notes
------------
- Stateful commands go to ExperimentController (Module 10).
- Stateless commands go to lightweight info handlers (stubs for now,
  ready for Kaggle API integration in future modules).
- Utility commands are handled inline.
- Unknown intents produce a helpful error message.
"""

from modules.C_nlu.intent_detection import (
    is_stateful_intent,
    is_stateless_intent,
    is_utility_intent,
    get_supported_intents,
)
from modules.D_control.state_manager import get_state_manager
from modules.D_control.experiment_controller import ExperimentController


class CommandRouter:
    """
    Dispatches structured NLU commands to the appropriate handler.

    Usage
    -----
    router = CommandRouter()
    result = router.route({
        "intent": "set_learning_rate",
        "slots": {"learning_rate": 0.01},
        "missing_slots": [],
        "invalid_slots": {},
    })
    """

    def __init__(self):
        self._sm = get_state_manager()
        self._controller = ExperimentController(self._sm)

    def route(self, command: dict) -> dict:
        """
        Route a structured command to the correct handler.

        Parameters
        ----------
        command : dict — output from Module C's understand() function.
                  Must contain at minimum: "intent", "slots"

        Returns
        -------
        dict with keys:
            success   (bool)  — whether the command succeeded
            message   (str)   — human-readable result description
            intent    (str)   — the intent that was processed
            category  (str)   — "stateful", "stateless", "utility", or "error"
        """
        intent = command.get("intent", "unknown_intent")

        # ── Stateful experiment commands ──────────────────
        if is_stateful_intent(intent):
            result = self._controller.execute(command)
            return {
                **result,
                "intent": intent,
                "category": "stateful",
            }

        # ── Stateless information queries ─────────────────
        if is_stateless_intent(intent):
            result = self._handle_stateless(command)
            return {
                **result,
                "intent": intent,
                "category": "stateless",
            }

        # ── Utility commands ──────────────────────────────
        if is_utility_intent(intent):
            result = self._handle_utility(command)
            return {
                **result,
                "intent": intent,
                "category": "utility",
            }

        # ── Unknown intent ────────────────────────────────
        self._sm.append_log(f"❓ Unknown command: {command.get('raw_text', '')}")
        return {
            "success": False,
            "message": "I didn't understand that command. "
                       "Say 'help' to see available commands.",
            "intent": intent,
            "category": "error",
        }

    # ─────────────────────────────────────────────────────
    # STATELESS HANDLERS
    # ─────────────────────────────────────────────────────

    def _handle_stateless(self, command: dict) -> dict:
        """Handle stateless information queries."""
        intent = command.get("intent")
        slots = command.get("slots", {})

        if intent == "search_dataset":
            query = slots.get("query", "")
            self._sm.append_log(f"🔍 Dataset search: '{query}'")
            return {
                "success": True,
                "message": f"Searching for datasets matching '{query}'... "
                           f"(Kaggle API integration coming in future modules).",
            }

        if intent == "get_dataset_info":
            dataset = slots.get("dataset") or self._sm.get("dataset")
            if dataset:
                self._sm.append_log(f"ℹ️ Dataset info requested: {dataset}")
                return {
                    "success": True,
                    "message": f"Dataset '{dataset}' info: "
                               f"(detailed stats coming in future modules).",
                }
            return {
                "success": False,
                "message": "No dataset specified or loaded. "
                           "Load a dataset first or specify one.",
            }

        if intent == "show_competition":
            self._sm.append_log("🏆 Competition listing requested")
            return {
                "success": True,
                "message": "Kaggle competitions: "
                           "(Kaggle API integration coming in future modules).",
            }

        if intent == "show_leaderboard":
            self._sm.append_log("📊 Leaderboard requested")
            return {
                "success": True,
                "message": "Leaderboard data: "
                           "(Kaggle API integration coming in future modules).",
            }

        return {
            "success": False,
            "message": f"No handler for stateless intent '{intent}'.",
        }

    # ─────────────────────────────────────────────────────
    # UTILITY HANDLERS
    # ─────────────────────────────────────────────────────

    def _handle_utility(self, command: dict) -> dict:
        """Handle utility commands (help, repeat, etc.)."""
        intent = command.get("intent")

        if intent == "help":
            help_text = (
                "Available commands:\n"
                "• Load dataset <name>  — Load a dataset (titanic, iris, mnist, etc.)\n"
                "• Select model <name>  — Choose a model (xgboost, random_forest, logistic_regression)\n"
                "• Set learning rate <value>  — Set the learning rate\n"
                "• Set batch size <value>  — Set the batch size\n"
                "• Set epochs <value>  — Set total training epochs\n"
                "• Start training  — Begin training the model\n"
                "• Pause training  — Pause current training\n"
                "• Resume training  — Resume paused training\n"
                "• Stop training  — Stop training entirely\n"
                "• Show status  — Show current experiment status\n"
                "• Show accuracy  — Show current accuracy\n"
                "• Show loss curve  — Show loss history\n"
                "• Search dataset <query>  — Search for datasets\n"
                "• Help  — Show this help message"
            )
            self._sm.append_log("❓ Help requested")
            return {"success": True, "message": help_text}

        if intent == "repeat":
            last_response = self._sm.get("assistant_response")
            if last_response:
                return {"success": True, "message": f"(Repeating) {last_response}"}
            return {"success": True, "message": "Nothing to repeat yet."}

        if intent == "unknown_intent":
            return {
                "success": False,
                "message": "I didn't understand that command. "
                           "Say 'help' to see available commands.",
            }

        return {
            "success": False,
            "message": f"No handler for utility intent '{intent}'.",
        }


# ─────────────────────────────────────────────────────────
# MODULE-LEVEL CONVENIENCE
# ─────────────────────────────────────────────────────────

_router: CommandRouter | None = None


def route_command(command: dict) -> dict:
    """
    Route a structured NLU command to the correct handler.

    Module-level convenience wrapper using a singleton CommandRouter.

    Parameters
    ----------
    command : dict — output from Module C's understand() function

    Returns
    -------
    dict with success, message, intent, category
    """
    global _router
    if _router is None:
        _router = CommandRouter()
    return _router.route(command)
