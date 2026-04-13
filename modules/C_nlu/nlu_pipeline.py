from __future__ import annotations

from modules.C_nlu.intent_detection import detect_intent, get_intent_category, normalize_text
from modules.C_nlu.slot_filling import fill_slots


_REQUIRED = {
    "load_dataset": ["dataset"],
    "search_dataset": ["query"],
    "get_dataset_info": [],
    "load_code": [],
    "search_code": ["query"],
    "select_model": ["model"],
    "set_learning_rate": ["learning_rate"],
    "set_batch_size": ["batch_size"],
    "set_epochs": ["epochs"],
    "set_layers": ["layers"],
}

def understand(text: str) -> dict:
    normalised = normalize_text(text)
    intent = detect_intent(text)
    slots = fill_slots(text, intent)
    missing_slots = []
    invalid_slots = {}

    for slot_name in _REQUIRED.get(intent, []):
        if slot_name not in slots:
            missing_slots.append(slot_name)

    if "learning_rate" in slots and slots["learning_rate"] <= 0:
        invalid_slots["learning_rate"] = "must be > 0"
    if "batch_size" in slots and slots["batch_size"] <= 0:
        invalid_slots["batch_size"] = "must be > 0"
    if "epochs" in slots and slots["epochs"] <= 0:
        invalid_slots["epochs"] = "must be > 0"
    if "layers" in slots and slots["layers"] <= 0:
        invalid_slots["layers"] = "must be > 0"

    return {
        "intent": intent,
        "intent_category": get_intent_category(intent),
        "slots": slots,
        "missing_slots": missing_slots,
        "invalid_slots": invalid_slots,
        "raw_text": text,
        "normalised_text": normalised,
    }