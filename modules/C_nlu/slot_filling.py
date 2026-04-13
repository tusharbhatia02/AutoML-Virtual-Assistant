"""
Module 7 — Slot Filling / Parameter Extraction
================================================
Once Intent Detection (Module 6) identifies the action, Slot Filling
extracts the specific parameter values from the user's command.

Intent tells you the action.  Slots tell you the details.

Example:
    text   = "set learning rate to 0.01"
    intent = "set_learning_rate"
    output = {"learning_rate": 0.01}

Public API
----------
extract_slots(text: str, intent: str) -> dict
    Extract, validate, and return slots for the given intent.
    Returns dict with keys: "slots", "missing_slots", "invalid_slots"

get_slot_schema(intent: str) -> dict | None
    Return the expected slot schema for an intent.

Design notes
------------
- Regex-based extraction with intent-aware patterns (e.g., the number
  appearing after "to" or "of" in "set learning rate to 0.01").
- Vocabulary lists for model names and dataset names.
- Synonym resolution for common ASR variations (e.g., "x g boost" → "xgboost",
  "random forest classifier" → "random_forest", "lr" → "learning_rate").
- Type conversion: string → float / int / str with proper error handling.
- Slot validation: numeric ranges, supported model membership, non-empty strings.
- Missing slot detection: reports which required slots were not found.
- Invalid slot detection: reports which slots had bad values (e.g., "banana").
"""

import re
from modules.C_nlu.intent_detection import normalize_text


# ─────────────────────────────────────────────────────────
# SYNONYM DICTIONARIES
# ─────────────────────────────────────────────────────────

# Model synonyms — all spoken/written forms mapped to canonical ID.
# Sorted longest-first during lookup so "random forest" matches before "rf".
_MODEL_SYNONYMS: dict[str, str] = {
    # XGBoost
    "xgboost":                  "xgboost",
    "xg boost":                 "xgboost",
    "x g boost":                "xgboost",
    "ex g boost":               "xgboost",
    "extreme gradient boosting":"xgboost",
    "extreme gradient":         "xgboost",
    "xgb":                      "xgboost",
    # Random Forest
    "random forest":            "random_forest",
    "random forests":           "random_forest",
    "randomforest":             "random_forest",
    "rf":                       "random_forest",
    # Logistic Regression
    "logistic regression":      "logistic_regression",
    "logistic":                 "logistic_regression",
    "log reg":                  "logistic_regression",
    "lr model":                 "logistic_regression",
    "logreg":                   "logistic_regression",
}

# Dataset synonyms — spoken/written forms → canonical dataset name.
_DATASET_SYNONYMS: dict[str, str] = {
    "titanic":              "titanic",
    "the titanic":          "titanic",
    "titanic dataset":      "titanic",
    "iris":                 "iris",
    "the iris":             "iris",
    "iris dataset":         "iris",
    "iris data":            "iris",
    "mnist":                "mnist",
    "the mnist":            "mnist",
    "cifar":                "cifar10",
    "cifar10":              "cifar10",
    "cifar 10":             "cifar10",
    "cifar-10":             "cifar10",
    "boston":                "boston",
    "boston housing":        "boston",
    "the boston":            "boston",
    "wine":                 "wine",
    "wine quality":         "wine",
    "the wine":             "wine",
    "diabetes":             "diabetes",
    "the diabetes":         "diabetes",
    "breast cancer":        "breast_cancer",
    "breast_cancer":        "breast_cancer",
    "the breast cancer":    "breast_cancer",
}

_SUPPORTED_MODELS = frozenset({"random_forest", "xgboost", "logistic_regression"})

_SUPPORTED_DATASETS = frozenset({
    "titanic", "iris", "mnist", "cifar10",
    "boston", "wine", "diabetes", "breast_cancer",
})


# ─────────────────────────────────────────────────────────
# SLOT SCHEMAS — defines what each intent expects
# ─────────────────────────────────────────────────────────

_SLOT_SCHEMAS: dict[str, list[dict]] = {
    "load_dataset":      [{"name": "dataset",       "type": "string", "required": True}],
    "select_model":      [{"name": "model",         "type": "string", "required": True}],
    "set_learning_rate": [{"name": "learning_rate",  "type": "float",  "required": True}],
    "set_batch_size":    [{"name": "batch_size",     "type": "int",    "required": True}],
    "set_epochs":        [{"name": "epochs",         "type": "int",    "required": True}],
    "search_dataset":    [{"name": "query",          "type": "string", "required": True}],
    "get_dataset_info":  [{"name": "dataset",        "type": "string", "required": False}],
    # Intents with no slots
    "start_training":   [],
    "pause_training":   [],
    "stop_training":    [],
    "resume_training":  [],
    "show_status":      [],
    "show_accuracy":    [],
    "show_loss_curve":  [],
    "show_competition": [],
    "show_leaderboard": [],
    "help":             [],
    "repeat":           [],
    "unknown_intent":   [],
}


# ─────────────────────────────────────────────────────────
# EXTRACTION HELPERS
# ─────────────────────────────────────────────────────────

def _extract_learning_rate(text: str) -> float | None:
    """
    Extract a learning rate value.  Prefers the number appearing after
    "to", "of", "=", or "rate" keywords.  Falls back to any float.
    Examples:
        "set learning rate to 0.01"   → 0.01
        "learning rate 0.001"         → 0.001
        "set lr to 0.05"              → 0.05
    """
    # Try: number after "to" / "=" / "rate" / "lr"
    m = re.search(r"(?:to|=|rate|lr)\s+(\d+\.?\d*)", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # Fallback: any decimal number
    m = re.search(r"(\d+\.\d+)", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # Last resort: any number
    m = re.search(r"(\d+)", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _extract_integer(text: str, context_keywords: list[str]) -> int | None:
    """
    Extract an integer value, preferring the number near context keywords.
    E.g., for batch_size the keywords are ["batch", "size", "to", "="].
    """
    # Try: number after a context keyword
    for kw in context_keywords:
        m = re.search(kw + r"\s+(\d+)", text)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    # Fallback: any integer
    m = re.search(r"(\d+)", text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None


def _extract_model(text: str) -> str | None:
    """
    Extract and normalise a model name using the synonym dictionary.
    Checks longest synonyms first (greedy matching avoids partial hits).
    """
    normalised = normalize_text(text)
    for synonym in sorted(_MODEL_SYNONYMS.keys(), key=len, reverse=True):
        if synonym in normalised:
            return _MODEL_SYNONYMS[synonym]
    return None


def _extract_dataset(text: str) -> str | None:
    """
    Extract and normalise a dataset name using the synonym dictionary.
    Checks longest synonyms first (greedy matching).
    """
    normalised = normalize_text(text)
    for synonym in sorted(_DATASET_SYNONYMS.keys(), key=len, reverse=True):
        if synonym in normalised:
            return _DATASET_SYNONYMS[synonym]
    return None


def _extract_search_query(text: str) -> str | None:
    """
    Extract a meaningful search query by stripping command verbs/filler.
    Example: "search for nlp datasets on kaggle" → "nlp datasets"
    """
    normalised = normalize_text(text)
    # Try capturing text after common search prefixes
    for prefix_re in [
        r"search\s+(?:for\s+)?(?:dataset\s+)?(?:on\s+)?(?:kaggle\s+)?",
        r"find\s+(?:dataset\s+)?(?:on\s+)?(?:kaggle\s+)?",
        r"look\s+for\s+(?:dataset\s+)?",
        r"browse\s+(?:dataset\s+)?",
        r"kaggle\s+dataset\s+",
    ]:
        m = re.search(prefix_re + r"(.+)", normalised)
        if m:
            q = m.group(1).strip()
            if q:
                return q
    # Fallback: strip common keywords and return the rest
    cleaned = re.sub(
        r"\b(search|find|look|for|browse|kaggle|dataset|on|data)\b",
        "", normalised,
    ).strip()
    return cleaned if cleaned else None


# ─────────────────────────────────────────────────────────
# SLOT VALIDATION
# ─────────────────────────────────────────────────────────

def _validate_slot(slot_name: str, value) -> tuple[bool, str]:
    """
    Validate a slot value against business rules.

    Returns (is_valid, error_message).
    """
    if slot_name == "learning_rate":
        if not isinstance(value, (int, float)):
            return False, "Learning rate must be a number."
        if value <= 0:
            return False, f"Learning rate must be positive, got {value}."
        if value >= 10:
            return False, f"Learning rate {value} is unreasonably large (expected < 10)."
        return True, ""

    if slot_name == "batch_size":
        if not isinstance(value, int):
            return False, "Batch size must be an integer."
        if value <= 0:
            return False, f"Batch size must be positive, got {value}."
        if value > 4096:
            return False, f"Batch size {value} exceeds maximum (4096)."
        return True, ""

    if slot_name == "epochs":
        if not isinstance(value, int):
            return False, "Epoch count must be an integer."
        if value <= 0:
            return False, f"Epochs must be positive, got {value}."
        if value > 10000:
            return False, f"Epochs {value} exceeds maximum (10000)."
        return True, ""

    if slot_name == "model":
        if not isinstance(value, str) or not value:
            return False, "Model name must be a non-empty string."
        if value not in _SUPPORTED_MODELS:
            return False, (
                f"Model '{value}' is not supported. "
                f"Supported models: {', '.join(sorted(_SUPPORTED_MODELS))}"
            )
        return True, ""

    if slot_name == "dataset":
        if not isinstance(value, str) or not value:
            return False, "Dataset name must be a non-empty string."
        return True, ""

    if slot_name == "query":
        if not isinstance(value, str) or not value.strip():
            return False, "Search query must be a non-empty string."
        return True, ""

    # Unknown slot — allow by default
    return True, ""


# ─────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────

def get_slot_schema(intent: str) -> list[dict] | None:
    """
    Return the expected slot schema for an intent.
    Returns None if the intent is not recognised.
    """
    return _SLOT_SCHEMAS.get(intent)


def extract_slots(text: str, intent: str) -> dict:
    """
    Extract, validate, and return slots for the given intent.

    This is the main entry point of Module 7.

    Parameters
    ----------
    text   : str — raw or normalised user text
    intent : str — detected intent label from Module 6

    Returns
    -------
    dict with keys:
        slots          (dict)  — extracted & validated key-value pairs
        missing_slots  (list)  — names of required slots not found
        invalid_slots  (dict)  — {slot_name: error_message} for bad values
    """
    schema = _SLOT_SCHEMAS.get(intent, [])
    slots: dict = {}
    missing_slots: list[str] = []
    invalid_slots: dict[str, str] = {}

    normalised = normalize_text(text)

    for slot_def in schema:
        name     = slot_def["name"]
        stype    = slot_def["type"]
        required = slot_def.get("required", False)
        value    = None

        # ── Extract value using slot-specific logic ───────
        if name == "learning_rate":
            value = _extract_learning_rate(normalised)

        elif name == "batch_size":
            value = _extract_integer(normalised, ["to", "size", "batch", "="])

        elif name == "epochs":
            value = _extract_integer(normalised, ["to", "epochs", "epoch", "for", "="])

        elif name == "model":
            value = _extract_model(normalised)

        elif name == "dataset":
            value = _extract_dataset(normalised)

        elif name == "query":
            value = _extract_search_query(normalised)

        # ── Missing value? ────────────────────────────────
        if value is None:
            if required:
                missing_slots.append(name)
            continue

        # ── Type conversion (safety net) ──────────────────
        try:
            if stype == "float" and not isinstance(value, float):
                value = float(value)
            elif stype == "int" and not isinstance(value, int):
                value = int(value)
            elif stype == "string" and not isinstance(value, str):
                value = str(value)
        except (ValueError, TypeError):
            invalid_slots[name] = f"Cannot convert '{value}' to {stype}."
            continue

        # ── Validate ──────────────────────────────────────
        is_valid, error_msg = _validate_slot(name, value)
        if is_valid:
            slots[name] = value
        else:
            invalid_slots[name] = error_msg

    return {
        "slots":         slots,
        "missing_slots": missing_slots,
        "invalid_slots": invalid_slots,
    }
