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
from modules.F_stateless_info.weather_service import get_weather, weather_to_text
from modules.E_ml_automl.qwen_llm import QwenAssistant


def _format_duration(seconds: int) -> str:
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)

    parts = []
    if hours:
        parts.append(f"{hours} hour(s)")
    if mins:
        parts.append(f"{mins} minute(s)")
    if secs or not parts:
        parts.append(f"{secs} second(s)")
    return ", ".join(parts)


class CommandRouter:
    def __init__(self):
        self._sm = get_state_manager()
        self._controller = ExperimentController(self._sm)

    def route(self, command: dict) -> dict:
        intent = command.get("intent", "out_of_scope")

        if intent in {
            "set_timer",
            "check_timer",
            "pause_timer",
            "resume_timer",
            "stop_timer",
            "restart_timer",
            "reset_timer",
            "add_time_to_timer",
            "cancel_timer",
        }:
            result = self._handle_timer(command)
            return {**result, "intent": intent, "category": "stateful"}

        if is_stateful_intent(intent):
            result = self._controller.execute(command)
            return {**result, "intent": intent, "category": "stateful"}

        if is_stateless_intent(intent):
            result = self._handle_stateless(command)
            return {**result, "intent": intent, "category": "stateless"}

        if is_utility_intent(intent):
            result = self._handle_utility(command)
            return {**result, "intent": intent, "category": "utility"}

        self._sm.append_log(f"❓ Out-of-scope command: {command.get('raw_text', '')}")
        return {
            "success": False,
            "message": "I am unable to perform that operation, it is out of the scope I was build upon.",
            "intent": "out_of_scope",
            "category": "utility",
        }

    def _handle_timer(self, command: dict) -> dict:
        intent = command.get("intent")
        slots = command.get("slots", {})

        if intent == "set_timer":
            duration = int(slots.get("duration_seconds", 0) or 0)
            if duration <= 0:
                return {
                    "success": False,
                    "message": "Please specify a valid timer duration, for example: set a timer for 2 minutes.",
                }
            label = slots.get("label", "timer")
            self._sm.start_timer(duration, label)
            self._sm.append_log(f"⏲️ Timer started: {label} ({duration}s)")
            return {
                "success": True,
                "message": f"Timer started for {_format_duration(duration)}.",
            }

        if intent == "check_timer":
            timer = self._sm.get_timer_info()
            if not timer.get("exists"):
                return {"success": True, "message": "There is no active timer."}

            if timer["status"] == "completed":
                return {"success": True, "message": f"Your {timer.get('label', 'timer')} is done."}

            return {
                "success": True,
                "message": (
                    f"Your {timer.get('label', 'timer')} is {timer['status']} "
                    f"with {_format_duration(timer['remaining_seconds'])} remaining."
                ),
            }

        if intent == "pause_timer":
            if self._sm.pause_timer():
                timer = self._sm.get_timer_info()
                return {
                    "success": True,
                    "message": f"Paused the {timer.get('label', 'timer')} with {_format_duration(timer['remaining_seconds'])} remaining.",
                }
            return {"success": False, "message": "There is no running timer to pause."}

        if intent == "resume_timer":
            if self._sm.resume_timer():
                timer = self._sm.get_timer_info()
                return {
                    "success": True,
                    "message": f"Resumed the {timer.get('label', 'timer')}.",
                }
            return {"success": False, "message": "There is no paused timer to resume."}

        if intent == "stop_timer":
            if self._sm.stop_timer():
                return {"success": True, "message": "Stopped the timer."}
            return {"success": False, "message": "There is no timer to stop."}

        if intent == "restart_timer":
            if self._sm.restart_timer():
                timer = self._sm.get_timer_info()
                return {
                    "success": True,
                    "message": f"Restarted the {timer.get('label', 'timer')} for {_format_duration(timer['remaining_seconds'])}.",
                }
            return {"success": False, "message": "There is no timer to restart."}

        if intent == "reset_timer":
            if self._sm.reset_timer():
                timer = self._sm.get_timer_info()
                return {
                    "success": True,
                    "message": f"Reset the {timer.get('label', 'timer')} back to {_format_duration(timer['remaining_seconds'])}. It is now paused.",
                }
            return {"success": False, "message": "There is no timer to reset."}

        if intent == "add_time_to_timer":
            extra = int(slots.get("duration_seconds", 0) or 0)
            if extra <= 0:
                return {"success": False, "message": "Please specify how much time to add."}
            if self._sm.add_time_to_timer(extra):
                timer = self._sm.get_timer_info()
                return {
                    "success": True,
                    "message": f"Added {_format_duration(extra)} to the {timer.get('label', 'timer')}. It now has {_format_duration(timer['remaining_seconds'])} remaining.",
                }
            return {"success": False, "message": "There is no timer to extend."}

        if intent == "cancel_timer":
            timer = self._sm.get_timer_info()
            if not timer.get("exists"):
                return {"success": True, "message": "There was no active timer to cancel."}
            self._sm.cancel_timer()
            return {"success": True, "message": f"Cancelled the {timer.get('label', 'timer')}."}

        return {"success": False, "message": f"No timer handler for '{intent}'."}

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
            msg = f"Found {len(result.get('results', []))} notebook/script result(s)." if success else f"No Kaggle reference notebooks found for '{query}'."
            return {"success": True, "message": msg, "data": result.get("results", [])}

        if intent == "suggest_model":
            profile = self._sm.get("dataset_info", {}).get("profile", {})
            is_tabular = profile.get("modality", "") == "tabular"
            msg = QwenAssistant.suggest_model(profile, is_tabular)
            self._sm.append_log("🤖 Qwen suggested model")
            return {"success": True, "message": msg, "data": []}

        if intent == "suggest_hyperparameters":
            profile = self._sm.get("dataset_info", {}).get("profile", {})
            model_name = self._sm.get("model", "Unknown")
            msg = QwenAssistant.suggest_hyperparameters(model_name, profile)
            self._sm.append_log("🤖 Qwen suggested hyperparameters")
            return {"success": True, "message": msg, "data": []}

        if intent == "get_weather":
            city = slots.get("city", "")
            day = slots.get("day", "today")
            try:
                result = get_weather(city, day)
            except Exception as e:
                return {"success": False, "message": f"Weather lookup failed: {e}", "data": {}}

            if not result.get("success"):
                return {"success": False, "message": result.get("error", "Weather lookup failed."), "data": result}

            self._sm.set_weather_result(result)
            self._sm.set_stateless_results(result)
            self._sm.append_log(f"🌦️ Weather requested: {city} ({day})")
            return {"success": True, "message": weather_to_text(result), "data": result}

        return {"success": False, "message": f"No handler for stateless intent '{intent}'.", "data": []}

    def _handle_utility(self, command: dict) -> dict:
        intent = command.get("intent")

        if intent == "help":
            help_text = (
                "Here are the available commands:\n"
                "• hey mycroft set a timer for 2 minutes\n"
                "• hey mycroft pause timer\n"
                "• hey mycroft resume timer\n"
                "• hey mycroft restart timer\n"
                "• hey mycroft reset timer\n"
                "• hey mycroft add 2 minutes to timer\n"
                "• hey mycroft what is the weather in Ottawa today\n"
                "• hey mycroft load mnist dataset\n"
                "• hey mycroft load corresponding code\n"
                "• hey mycroft set learning rate to 0.01\n"
                "• hey mycroft search dataset fraud detection\n"
                "• hey mycroft show leaderboard titanic"
            )
            return {"success": True, "message": help_text}

        if intent == "repeat":
            last = self._sm.get("assistant_response", "")
            return {"success": True, "message": last or "Nothing to repeat yet."}

        if intent == "greetings":
            import random
            greetings = [
                "Hello! I am Mycroft, your machine learning assistant. How can I help you today?",
                "Hi there! Ready to build some models?",
                "Greetings! What dataset are we working with today?",
                "Good to see you! You can also ask me for weather updates or timer controls.",
                "Hey! I'm Mycroft, here to help you train models, check weather, and manage timers.",
            ]
            self._sm.append_log("👋 Greeted the user")
            return {"success": True, "message": random.choice(greetings)}

        if intent == "farewell":
            import random
            farewells = [
                "Goodbye! Let me know when you're ready to train more models.",
                "See you later! Your workspace will be waiting.",
                "Farewell! Have a great rest of your day.",
                "Bye! Let me know if you need any more AI support later.",
                "Good night! Mycroft shutting down.",
            ]
            self._sm.append_log("👋 Said goodbye")
            return {"success": True, "message": random.choice(farewells)}

        if intent == "out_of_scope":
            self._sm.append_log("🚫 Out-of-scope request received")
            return {
                "success": False,
                "message": "I am unable to perform that operation, it is out of the scope I was build upon.",
            }

        return {"success": False, "message": f"No handler for utility intent '{intent}'."}


_router = None


def route_command(command: dict) -> dict:
    global _router
    if _router is None:
        _router = CommandRouter()
    return _router.route(command)