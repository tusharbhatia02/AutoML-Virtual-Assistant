from __future__ import annotations

import re

from modules.C_nlu.intent_detection import MODEL_KEYWORDS, normalize_text

# ── Known datasets (for normalisation) ──────────────────────

_KNOWN_DATASETS = {
    "titanic": "titanic",
    "iris": "iris",
    "mnist": "mnist",
    "digits": "mnist",
    "cifar10": "cifar10",
    "cifar 10": "cifar10",
    "cifar-10": "cifar10",
    "boston": "boston",
    "wine": "wine",
    "diabetes": "diabetes",
    "breast cancer": "breast_cancer",
    "breast_cancer": "breast_cancer",
}

# ── Slot schemas ────────────────────────────────────────────

_SLOT_SCHEMAS: dict[str, list[dict] | None] = {
    "load_dataset":       [{"name": "dataset",       "type": "str",   "required": True}],
    "select_model":       [{"name": "model",         "type": "str",   "required": True}],
    "set_learning_rate":  [{"name": "learning_rate", "type": "float", "required": True}],
    "set_batch_size":     [{"name": "batch_size",    "type": "int",   "required": True}],
    "set_epochs":         [{"name": "epochs",        "type": "int",   "required": True}],
    "set_layers":         [{"name": "layers",        "type": "int",   "required": True}],
    "search_dataset":     [{"name": "query",         "type": "str",   "required": True}],
    "get_dataset_info":   [{"name": "dataset",       "type": "str",   "required": False}],
    "load_code":          [{"name": "dataset",       "type": "str",   "required": False}],
    "search_code":        [{"name": "query",         "type": "str",   "required": True}],
    "show_leaderboard":   [{"name": "query",         "type": "str",   "required": False}],
    # No-slot intents
    "start_training":     [],
    "pause_training":     [],
    "resume_training":    [],
    "stop_training":      [],
    "show_status":        [],
    "show_accuracy":      [],
    "show_loss_curve":    [],
    "run_code":           [],
    "show_output":        [],
    "help":               [],
    "repeat":             [],
    "unknown_intent":     [],
}

_SUPPORTED_MODELS = set(MODEL_KEYWORDS.values())


def get_slot_schema(intent: str) -> list[dict] | None:
    """Return the slot schema for an intent, or None if the intent is unknown."""
    return _SLOT_SCHEMAS.get(intent)


def _extract_first_number(text: str):
    m = re.search(r"(-?\d+(?:\.\d+)?)", text)
    return m.group(1) if m else None


def _try_known_dataset(text: str) -> str | None:
    """Check if any known dataset name appears in the text."""
    t = text.lower().strip()
    # Try longest matches first to prefer "breast cancer" over partial
    for name in sorted(_KNOWN_DATASETS.keys(), key=len, reverse=True):
        if name in t:
            return _KNOWN_DATASETS[name]
    return None


def _extract_dataset_phrase(text: str, for_code: bool = False) -> str | None:
    t = text.lower().strip()

    # Try known datasets first
    known = _try_known_dataset(t)
    if known:
        return known

    patterns = [
        r"(?:load|retrieve|fetch|use|import|open)\s+(?:the\s+)?(.+?)(?:\s+dataset|\s+data)?\s*$",
        r"(?:get dataset info|tell me about|about dataset|describe dataset|describe the dataset)\s+(.+)$",
        r"(?:search dataset|find dataset|search data|find data)\s+(.+)$",
    ]

    if for_code:
        patterns = [
            r"(?:load|retrieve|fetch)\s+(.+?)\s+(?:code|notebook|script)$",
            r"(?:load corresponding code(?: for)?\s*)(.*)$",
        ] + patterns

    for pattern in patterns:
        m = re.search(pattern, t)
        if m:
            value = m.group(1).strip(" ,:-")
            
            # Clean up trailing phrases
            value = re.sub(r"\s+from\s+kaggl?e[l]?\s*$", "", value)
            value = re.sub(r"\s+dataset\s*$", "", value)
            value = re.sub(r"\s+data\s*$", "", value)
            value = value.strip(" ,:-")
            
            if value and value != "corresponding":
                # Check again if the extracted phrase is a known dataset
                known2 = _try_known_dataset(value)
                return known2 or value
    return None


def fill_slots(text: str, intent: str) -> dict:
    """Basic slot extraction (returns flat dict). Used by nlu_pipeline."""
    t = normalize_text(text)
    slots: dict = {}

    if intent in {"load_dataset", "search_dataset", "get_dataset_info"}:
        dataset = _extract_dataset_phrase(t)
        if dataset:
            if intent == "search_dataset":
                slots["query"] = dataset
            else:
                slots["dataset"] = dataset

    if intent in {"load_code", "search_code"}:
        dataset = _extract_dataset_phrase(t, for_code=True)
        if dataset:
            if intent == "search_code":
                slots["query"] = dataset
            else:
                slots["dataset"] = dataset

    if intent == "show_leaderboard":
        m = re.search(r"leaderboard\s+(.+)$", t)
        if m:
            slots["query"] = m.group(1).strip(" ,:-")

    if intent == "select_model":
        for k, v in MODEL_KEYWORDS.items():
            if k in t:
                slots["model"] = v
                break

    if intent == "set_learning_rate":
        num = _extract_first_number(t)
        if num is not None:
            slots["learning_rate"] = float(num)

    if intent == "set_batch_size":
        num = _extract_first_number(t)
        if num is not None:
            slots["batch_size"] = int(float(num))

    if intent == "set_epochs":
        num = _extract_first_number(t)
        if num is not None:
            slots["epochs"] = int(float(num))

    if intent == "set_layers":
        num = _extract_first_number(t)
        if num is not None:
            slots["layers"] = int(float(num))

    return slots


def extract_slots(text: str, intent: str) -> dict:
    """
    Full slot extraction with validation.
    Returns: {"slots": {...}, "missing_slots": [...], "invalid_slots": {...}}
    """
    slots = fill_slots(text, intent)
    missing_slots: list[str] = []
    invalid_slots: dict[str, str] = {}

    schema = _SLOT_SCHEMAS.get(intent, [])
    if schema is None:
        schema = []

    for field in schema:
        name = field["name"]
        if field.get("required") and name not in slots:
            missing_slots.append(name)

    # ── Validation ──────────────────────────────────────────
    if "learning_rate" in slots:
        lr = slots["learning_rate"]
        if lr <= 0:
            invalid_slots["learning_rate"] = "must be > 0"
        elif lr > 10:
            invalid_slots["learning_rate"] = "must be <= 10"

    if "batch_size" in slots:
        bs = slots["batch_size"]
        if bs <= 0:
            invalid_slots["batch_size"] = "must be > 0"
        elif bs > 4096:
            invalid_slots["batch_size"] = "must be <= 4096"

    if "epochs" in slots:
        ep = slots["epochs"]
        if ep <= 0:
            invalid_slots["epochs"] = "must be > 0"
        elif ep > 10000:
            invalid_slots["epochs"] = "must be <= 10000"

    if "layers" in slots:
        ly = slots["layers"]
        if ly <= 0:
            invalid_slots["layers"] = "must be > 0"

    return {
        "slots": slots,
        "missing_slots": missing_slots,
        "invalid_slots": invalid_slots,
    }