from __future__ import annotations

from modules.C_nlu.intent_detection import (
    is_stateful_intent,
    is_stateless_intent,
    is_utility_intent,
)
from modules.D_control.state_manager import get_state_manager
from modules.D_control.experiment_controller import ExperimentController
from modules.F_stateless_info.kaggle_service import (
    search_datasets,
    get_dataset_info,
    show_competitions,
    show_leaderboard,
)
from modules.F_stateless_info.kaggle_kernel_service import search_kernels


class CommandRouter:
    def __init__(self):
        self._sm = get_state_manager()
        self._controller = ExperimentController(self._sm)

    def route(self, command: dict) -> dict:
        intent = command.get("intent", "unknown_intent")

        if is_stateful_intent(intent):
            result = self._controller.execute(command)
            return {**result, "intent": intent, "category": "stateful"}

        if is_stateless_intent(intent):
            result = self._handle_stateless(command)
            return {**result, "intent": intent, "category": "stateless"}

        if is_utility_intent(intent):
            result = self._handle_utility(command)
            return {**result, "intent": intent, "category": "utility"}

        self._sm.append_log(f"❓ Unknown command: {command.get('raw_text', '')}")
        return {
            "success": False,
            "message": "I didn't understand that command. Say 'help' to see examples.",
            "intent": intent,
            "category": "error",
        }

    def _handle_stateless(self, command: dict) -> dict:
        intent = command.get("intent")
        slots = command.get("slots", {})

        if intent == "search_dataset":
            query = slots.get("query", "")
            result = search_datasets(query)
            self._sm.set_stateless_results(result.get("results", []))
            self._sm.append_log(f"🔍 Dataset search: {query}")
            return {"success": True, "message": f"Found {len(result.get('results', []))} dataset result(s).", "data": result.get("results", [])}

        if intent == "get_dataset_info":
            dataset_query = slots.get("dataset") or self._sm.get("dataset") or ""
            result = get_dataset_info(dataset_query)
            payload = {
                "top_result": result.get("top_result", {}),
                "files": result.get("files", []),
                "search_results": result.get("search_results", []),
            } if result.get("success") else {}
            self._sm.set_stateless_results(payload)
            self._sm.append_log(f"ℹ️ Dataset info: {dataset_query}")
            return {"success": result.get("success", False), "message": "Dataset info retrieved." if result.get("success") else result.get("error", "Failed."), "data": payload}

        if intent == "show_competition":
            result = show_competitions()
            self._sm.set_stateless_results(result.get("results", []))
            self._sm.append_log("🏆 Competition list requested")
            return {"success": True, "message": f"Retrieved {len(result.get('results', []))} competition result(s).", "data": result.get("results", [])}

        if intent == "show_leaderboard":
            competition = slots.get("query", "titanic")
            result = show_leaderboard(competition)
            self._sm.set_stateless_results(result.get("results", []))
            self._sm.append_log(f"📊 Leaderboard requested: {competition}")
            return {"success": True, "message": f"Leaderboard retrieved for {competition}.", "data": result.get("results", [])}

        if intent == "search_code":
            query = slots.get("query", "")
            result = search_kernels(query)
            self._sm.set_stateless_results(result.get("results", []))
            self._sm.append_log(f"📘 Code search: {query}")
            success = result.get("success", False)
            if success:
                msg = f"Found {len(result.get('results', []))} notebook/script result(s)."
            else:
                msg = f"No Kaggle reference notebooks found for '{query}'."
                success = True # Soft fallback

            return {
                "success": success,
                "message": msg,
                "data": result.get("results", []),
            }

        return {"success": False, "message": f"No handler for stateless intent '{intent}'.", "data": []}

    def _handle_utility(self, command: dict) -> dict:
        intent = command.get("intent")

        if intent == "help":
            help_text = (
                "Here are the available commands:\n"
                "• hey mello load mnist dataset\n"
                "• hey mello load corresponding code\n"
                "• hey mello set learning rate to 0.01\n"
                "• hey mello set layers to 4\n"
                "• hey mello run code\n"
                "• hey mello search dataset fraud detection\n"
                "• hey mello show leaderboard titanic"
            )
            return {"success": True, "message": help_text}

        if intent == "repeat":
            last = self._sm.get("assistant_response", "")
            return {"success": True, "message": last or "Nothing to repeat yet."}

        return {"success": False, "message": f"No handler for utility intent '{intent}'."}


_router = None

def route_command(command: dict) -> dict:
    global _router
    if _router is None:
        _router = CommandRouter()
    return _router.route(command)