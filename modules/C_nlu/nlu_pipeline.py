"""
Module C — NLU Pipeline (Top-Level Entry Point)
=================================================
Combines Intent Detection (Module 6) and Slot Filling (Module 7)
into a single unified pipeline.

This is the main entry point for converting natural language into
a structured command object ready for Module D (Core Control Logic).

Public API
----------
understand(text: str) -> dict
    Full NLU pipeline: normalise → detect intent → extract slots.
    Returns a structured command object.

Example
-------
>>> from modules.C_nlu.nlu_pipeline import understand
>>> result = understand("set learning rate to 0.01")
>>> result
{
    "intent": "set_learning_rate",
    "slots": {"learning_rate": 0.01},
    "missing_slots": [],
    "invalid_slots": {},
    "raw_text": "set learning rate to 0.01",
    "normalised_text": "set learning rate to 0.01",
    "intent_category": "stateful"
}
"""

from modules.C_nlu.intent_detection import (
    detect_intent,
    normalize_text,
    get_intent_category,
)
from modules.C_nlu.slot_filling import extract_slots


def understand(text: str) -> dict:
    """
    Full NLU pipeline: normalise text → detect intent → extract slots.

    Parameters
    ----------
    text : str — raw text from ASR (Module 5) or text input (Module 3)

    Returns
    -------
    dict with keys:
        intent           (str)   — detected intent label
        slots            (dict)  — extracted parameter key-value pairs
        missing_slots    (list)  — required slots that were not found
        invalid_slots    (dict)  — {slot_name: error_message}
        raw_text         (str)   — original input text
        normalised_text  (str)   — text after normalisation
        intent_category  (str)   — "stateful", "stateless", "utility", or "unknown"
    """
    # Step 1: Normalise text
    normalised = normalize_text(text)

    # Step 2: Detect intent
    intent = detect_intent(normalised)

    # Step 3: Extract and validate slots
    slot_result = extract_slots(normalised, intent)

    # Step 4: Build structured command object
    return {
        "intent": intent,
        "slots": slot_result["slots"],
        "missing_slots": slot_result["missing_slots"],
        "invalid_slots": slot_result["invalid_slots"],
        "raw_text": text,
        "normalised_text": normalised,
        "intent_category": get_intent_category(intent),
    }
