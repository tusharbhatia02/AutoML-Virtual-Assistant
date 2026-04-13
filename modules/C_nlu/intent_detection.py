from __future__ import annotations

import re

# ── Intent Sets ─────────────────────────────────────────────

STATEFUL_INTENTS = {
    "load_dataset",
    "select_model",
    "set_learning_rate",
    "set_batch_size",
    "set_epochs",
    "set_layers",
    "load_code",
    "run_code",
    "show_output",
    "start_training",
    "pause_training",
    "resume_training",
    "stop_training",
    "show_status",
    "show_accuracy",
    "show_loss_curve",
}

STATELESS_INTENTS = {
    "search_dataset",
    "get_dataset_info",
    "show_competition",
    "show_leaderboard",
    "search_code",
}

UTILITY_INTENTS = {
    "help",
    "repeat",
    "unknown_intent",
}

MODEL_KEYWORDS = {
    "xgboost": "xgboost",
    "xgb": "xgboost",
    "random forest": "random_forest",
    "random_forest": "random_forest",
    "rf": "random_forest",
    "logistic regression": "logistic_regression",
    "logistic_regression": "logistic_regression",
    "cnn": "cnn",
    "mlp": "mlp",
    "resnet": "resnet",
}

# ── ASR correction map ──────────────────────────────────────

_ASR_CORRECTIONS = [
    (r"\bx\s*g\s*boost\b", "xgboost"),
    (r"\blearning\s+great\b", "learning rate"),
    (r"\bbatch\s+eyes\b", "batch size"),
    (r"\bepics?\b", "epochs"),
    (r"\bep[io]cks?\b", "epochs"),
]


def normalize_text(text: str | None) -> str:
    """Lowercase, strip punctuation, collapse whitespace, apply ASR corrections."""
    if text is None:
        return ""
    t = text.strip()
    if not t:
        return ""
    t = t.lower()
    # Apply ASR corrections
    for pattern, replacement in _ASR_CORRECTIONS:
        t = re.sub(pattern, replacement, t)
    # Strip punctuation (keep periods in numbers like 0.01)
    t = re.sub(r"(?<!\d)\.(?!\d)", " ", t)   # dots not between digits
    t = re.sub(r"[^\w\s.]", " ", t)           # other punctuation
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ── Category helpers ────────────────────────────────────────

def is_stateful_intent(intent: str) -> bool:
    return intent in STATEFUL_INTENTS

def is_stateless_intent(intent: str) -> bool:
    return intent in STATELESS_INTENTS

def is_utility_intent(intent: str) -> bool:
    return intent in UTILITY_INTENTS

def get_intent_category(intent: str) -> str:
    if intent in STATEFUL_INTENTS:
        return "stateful"
    if intent in STATELESS_INTENTS:
        return "stateless"
    if intent in UTILITY_INTENTS:
        return "utility"
    return "unknown"

def get_supported_intents() -> list[str]:
    return sorted(STATEFUL_INTENTS | STATELESS_INTENTS | UTILITY_INTENTS)


# ── Intent Detection ────────────────────────────────────────

def detect_intent(text: str) -> str:
    t = normalize_text(text)

    if not t:
        return "unknown_intent"

    # ── Utility ──────────────────────────────────────────
    if t in {"help", "what can you do", "show help"}:
        return "help"
    if t in {"repeat", "say that again", "pardon", "come again"}:
        return "repeat"

    # ── Stateless (check early — "show leaderboard" before "show") ──
    if "leaderboard" in t:
        return "show_leaderboard"
    if re.search(r"\bcompetition", t):
        return "show_competition"

    if any(p in t for p in ["search code", "find code", "search notebook", "find notebook", "search kernel"]):
        return "search_code"

    if any(p in t for p in ["dataset info", "about dataset", "files for dataset",
                            "tell me about", "describe dataset", "describe the dataset",
                            "info on dataset", "info about"]):
        return "get_dataset_info"

    if any(p in t for p in ["search dataset", "find dataset", "search data", "find data"]):
        return "search_dataset"

    # ── Code workspace commands ──────────────────────────
    if any(v in t for v in ["load", "retrieve", "fetch"]) and any(k in t for k in ["code", "notebook", "script"]):
        return "load_code"

    if any(v in t for v in ["run", "execute"]) and any(k in t for k in ["code", "experiment"]):
        return "run_code"

    if "show output" in t or "display output" in t:
        return "show_output"

    # ── Hyperparameters ──────────────────────────────────
    if "learning rate" in t or re.search(r"\blr\b", t):
        return "set_learning_rate"

    if "batch size" in t:
        return "set_batch_size"

    if re.search(r"\bepochs?\b", t) and any(w in t for w in ["set", "change", "to", "for", "train"]):
        return "set_epochs"

    if "set layers" in t or "change layers" in t or "layers to" in t:
        return "set_layers"

    # ── Training control ─────────────────────────────────
    if any(p in t for p in ["start training", "begin training", "begin the training",
                            "run training", "launch training", "launch the training",
                            "train the model", "start the training"]):
        return "start_training"

    if any(p in t for p in ["resume training", "resume the training",
                            "continue training", "continue the training",
                            "unpause training", "unpause the training"]):
        return "resume_training"

    if any(p in t for p in ["pause training", "pause the training",
                            "hold training", "hold the training"]):
        return "pause_training"

    if any(p in t for p in ["stop training", "stop the training",
                            "cancel training", "cancel the training",
                            "abort training", "abort the training"]):
        return "stop_training"

    # ── Status / Metrics ─────────────────────────────────
    if any(p in t for p in ["show status", "check status", "current status",
                            "training status", "what is the status"]):
        return "show_status"

    if any(p in t for p in ["show accuracy", "current accuracy",
                            "what is the accuracy", "how accurate",
                            "accuracy of the model"]):
        return "show_accuracy"

    if any(p in t for p in ["show loss curve", "plot loss", "loss curve",
                            "loss history", "show loss history",
                            "loss graph", "plot loss graph"]):
        return "show_loss_curve"

    # ── Model selection (keyword match) ──────────────────
    if any(m in t for m in MODEL_KEYWORDS):
        return "select_model"

    # ── Dataset loading (regex fallback) ─────────────────
    if re.search(r"\b(load|import|open|retrieve|fetch|use)\b", t) and \
       not any(k in t for k in ["code", "notebook", "script", "model"]):
        return "load_dataset"

    return "unknown_intent"