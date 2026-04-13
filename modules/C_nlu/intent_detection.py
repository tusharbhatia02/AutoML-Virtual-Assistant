"""
Module 6 — Intent Detection
=============================
Identifies the type of request the user is making from natural-language
text (either transcribed speech from Module 5 or typed text from Module 3).

This module does NOT execute anything or change system state.
It only answers: "Which command category does this sentence belong to?"

Public API
----------
detect_intent(text: str) -> str
    Returns one of the supported intent labels.

normalize_text(text: str) -> str
    Cleans and normalises raw text for downstream matching.

is_stateful_intent(intent: str) -> bool
    Returns True if the intent modifies experiment state.

get_supported_intents() -> list[str]
    Returns all supported intent labels.

get_intent_category(intent: str) -> str
    Returns "stateful", "stateless", "utility", or "unknown".

Design notes
------------
- Rule-based / keyword-based approach chosen for reliability, debuggability,
  and suitability for course-project scope.
- Two-pass detection: keyword substring matching (fast path), then regex
  patterns (flexible fallback) for paraphrased commands.
- Patterns are checked in priority order (most specific first) to avoid
  false positives (e.g., "learning rate" before generic "rate").
- Handles ASR transcription mistakes via a correction dictionary applied
  during normalisation.
- Handles incomplete commands (e.g., "start" alone → ambiguous).
- Handles conflicting wording by checking most-specific first.
- Fallback to "unknown_intent" for unrecognised commands.
"""

import re


# ─────────────────────────────────────────────────────────
# TEXT NORMALISATION
# ─────────────────────────────────────────────────────────

# Common ASR transcription mistakes → correct form.
# Sorted longest-first so multi-word corrections are applied before
# shorter ones that might partially overlap.
_ASR_CORRECTIONS: list[tuple[str, str]] = sorted([
    ("x g boost",                       "xgboost"),
    ("ex g boost",                      "xgboost"),
    ("x gee boost",                     "xgboost"),
    ("ex gee boost",                    "xgboost"),
    ("xg boost",                        "xgboost"),
    ("extreme gradient boosting",       "xgboost"),
    ("extreme gradient",                "xgboost"),
    ("random forest classifier",        "random forest"),
    ("logistic regression classifier",  "logistic regression"),
    ("logistic regression model",       "logistic regression"),
    ("learning great",                  "learning rate"),
    ("learning red",                    "learning rate"),
    ("learning right",                  "learning rate"),
    ("batch eyes",                      "batch size"),
    ("bad size",                        "batch size"),
    ("batch sighs",                     "batch size"),
    ("epic",                            "epoch"),
    ("epics",                           "epochs"),
    ("epoch's",                         "epochs"),
    ("epos",                            "epochs"),
    ("e pox",                           "epochs"),
    ("learningrate",                    "learning rate"),
    ("batchsize",                       "batch size"),
    ("data set",                        "dataset"),
    ("data sit",                        "dataset"),
    ("trainning",                       "training"),
    ("traning",                         "training"),
], key=lambda x: len(x[0]), reverse=True)


def normalize_text(text: str) -> str:
    """
    Clean and normalise raw input text for intent matching.

    Steps applied in order:
    1. Convert to lowercase
    2. Remove punctuation except dots attached to numbers (e.g. 0.01)
    3. Collapse internal whitespace to single spaces
    4. Apply ASR correction dictionary (longest replacements first)
    5. Final strip

    Parameters
    ----------
    text : str — raw text from ASR or text input

    Returns
    -------
    str — normalised text ready for intent detection
    """
    if not text:
        return ""
    text = text.lower().strip()
    # Remove punctuation except dots in numbers (e.g. 0.01) and hyphens
    text = re.sub(r"[^\w\s.\-]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Apply ASR corrections (longest first to avoid partial clobber)
    for wrong, right in _ASR_CORRECTIONS:
        text = text.replace(wrong, right)
    return text


# ─────────────────────────────────────────────────────────
# INTENT REGISTRY — KEYWORD SUBSTRING PATTERNS
# ─────────────────────────────────────────────────────────
# Each entry: (intent_label, list_of_keyword_patterns)
# Patterns are checked against normalised text with Python `in` operator.
# ORDER MATTERS — more specific patterns MUST come before generic ones
# to prevent the generic pattern from stealing a match.

_INTENT_PATTERNS: list[tuple[str, list[str]]] = [

    # ── Hyperparameter setting (most specific first) ─────
    ("set_learning_rate", [
        "learning rate",
        "set lr to",
        "set lr ",
        "change lr",
        "update lr",
    ]),
    ("set_batch_size", [
        "batch size",
        "set batch",
        "change batch",
        "update batch",
        "modify batch",
    ]),
    ("set_epochs", [
        "set epoch",
        "set epochs",
        "change epoch",
        "number of epoch",
        "total epoch",
        "update epoch",
        "modify epoch",
        "train for",
    ]),

    # ── Training control (resume before start to avoid "resume" hitting "start") ─
    ("resume_training", [
        "resume training",
        "resume the training",
        "continue training",
        "continue the training",
        "unpause training",
        "unpause the training",
        "unpause",
    ]),
    ("pause_training", [
        "pause training",
        "pause the training",
        "hold training",
        "hold the training",
        "freeze training",
        "freeze the training",
        "pause",
    ]),
    ("stop_training", [
        "stop training",
        "stop the training",
        "cancel training",
        "cancel the training",
        "abort training",
        "abort the training",
        "end training",
        "end the training",
        "terminate training",
        "terminate the training",
        "kill training",
    ]),
    ("start_training", [
        "start training",
        "begin training",
        "start the training",
        "begin the training",
        "train the model",
        "train model",
        "run training",
        "launch training",
        "commence training",
        "start the train",
        "lets train",
        "let us train",
    ]),

    # ── Metrics / status (loss_curve before accuracy to avoid "loss" hitting accuracy) ─
    ("show_loss_curve", [
        "loss curve",
        "show loss",
        "plot loss",
        "display loss",
        "loss graph",
        "loss chart",
        "loss plot",
        "loss history",
        "training loss",
    ]),
    ("show_accuracy", [
        "show accuracy",
        "what is the accuracy",
        "current accuracy",
        "display accuracy",
        "get accuracy",
        "check accuracy",
        "accuracy so far",
        "how accurate",
        "accuracy",
    ]),
    ("show_status", [
        "show status",
        "current status",
        "what is the status",
        "system status",
        "show state",
        "experiment status",
        "whats the status",
        "check status",
        "status",
    ]),

    # ── Data (get_dataset_info before load_dataset so "dataset info" doesn't become load) ─
    ("get_dataset_info", [
        "dataset info",
        "describe dataset",
        "dataset description",
        "info about dataset",
        "about the dataset",
        "dataset details",
        "describe the dataset",
        "tell me about the dataset",
        "what is the dataset",
        "describe data",
    ]),
    ("search_dataset", [
        "search dataset",
        "find dataset",
        "look for dataset",
        "search for dataset",
        "search kaggle",
        "kaggle dataset",
        "browse dataset",
        "search for data",
        "find data on",
        "search for a dataset",
    ]),
    ("load_dataset", [
        "load dataset",
        "load the dataset",
        "use dataset",
        "open dataset",
        "import dataset",
        "load data",
        "load the data",
        "load the titanic",
        "load the iris",
        "load the mnist",
        "load the cifar",
        "load the boston",
        "load the wine",
        "load the diabetes",
        "load the breast cancer",
        "load titanic",
        "load iris",
        "load mnist",
        "load cifar",
        "load boston",
        "load wine",
        "load diabetes",
        "load breast cancer",
    ]),

    # ── Model selection ──────────────────────────────────
    ("select_model", [
        "select model",
        "choose model",
        "use model",
        "pick model",
        "change model",
        "switch model",
        "set model",
        "switch to model",
        "use xgboost",
        "use random forest",
        "use logistic regression",
        "use logistic",
        "use rf",
        "select xgboost",
        "select random forest",
        "select logistic",
        "choose xgboost",
        "choose random forest",
        "choose logistic",
    ]),

    # ── Competition / leaderboard ────────────────────────
    ("show_leaderboard", [
        "show leaderboard",
        "display leaderboard",
        "leaderboard",
        "ranking",
        "standings",
        "leader board",
    ]),
    ("show_competition", [
        "show competition",
        "kaggle competition",
        "competitions",
        "competition",
        "show competitions",
    ]),

    # ── Utility (last — most generic patterns) ───────────
    ("help", [
        "what can you do",
        "show commands",
        "list commands",
        "available commands",
        "how to use",
        "what do you support",
        "help me",
        "help",
    ]),
    ("repeat", [
        "say again",
        "say that again",
        "what did you say",
        "come again",
        "repeat that",
        "pardon",
        "repeat",
    ]),
]


# ─────────────────────────────────────────────────────────
# INTENT REGISTRY — REGEX FALLBACK PATTERNS
# ─────────────────────────────────────────────────────────
# These catch flexible phrasings that simple substring matching misses.
# Example: "load the titanic dataset" has "the titanic" between "load" and "dataset".

_REGEX_PATTERNS: list[tuple[str, re.Pattern]] = [
    # "load/open/import ... dataset/data" with any words in between
    ("load_dataset",   re.compile(r"\b(load|open|import)\b.{0,30}\b(dataset|data)\b")),
    # "load <known_name>" with optional words in between
    ("load_dataset",   re.compile(r"\bload\b.{0,20}\b(titanic|iris|mnist|cifar|boston|wine|diabetes|breast.?cancer)\b")),
    # "use/select/choose/pick ... model/xgboost/random forest/logistic" with words between
    ("select_model",   re.compile(r"\b(select|choose|pick|switch\s+to)\b.{0,20}\b(model|xgboost|random.forest|logistic)\b")),
    # "set/change/update ... rate/lr ... <number>"
    ("set_learning_rate", re.compile(r"\b(set|change|update|modify)\b.{0,15}\b(learning.?rate|lr)\b")),
    # "set/change ... batch ... <number>"
    ("set_batch_size", re.compile(r"\b(set|change|update|modify)\b.{0,15}\bbatch\b")),
    # "set/change ... epoch ... <number>"
    ("set_epochs",     re.compile(r"\b(set|change|update|modify)\b.{0,15}\bepoch")),
    # "start/begin/run ... train"
    ("start_training", re.compile(r"\b(start|begin|run|launch|commence)\b.{0,15}\btrain")),
    # "stop/cancel/abort ... train"
    ("stop_training",  re.compile(r"\b(stop|cancel|abort|end|terminate|kill)\b.{0,15}\btrain")),
    # "pause/hold/freeze ... train"
    ("pause_training", re.compile(r"\b(pause|hold|freeze)\b.{0,15}\btrain")),
    # "resume/continue/unpause ... train"
    ("resume_training", re.compile(r"\b(resume|continue|unpause)\b.{0,15}\btrain")),
]


# ── Intent category classification ───────────────────────

_STATEFUL_INTENTS = frozenset({
    "load_dataset",
    "select_model",
    "set_learning_rate",
    "set_batch_size",
    "set_epochs",
    "start_training",
    "pause_training",
    "stop_training",
    "resume_training",
    "show_status",
    "show_accuracy",
    "show_loss_curve",
})

_STATELESS_INTENTS = frozenset({
    "search_dataset",
    "get_dataset_info",
    "show_competition",
    "show_leaderboard",
})

_UTILITY_INTENTS = frozenset({
    "help",
    "repeat",
    "unknown_intent",
})


# ─────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────

def detect_intent(text: str) -> str:
    """
    Identify the intent of a user command.

    Two-pass detection:
    1. Fast keyword substring matching (covers 90%+ of commands).
    2. Regex fallback for flexible phrasings the substring pass misses.

    Unsupported / unrecognisable commands → "unknown_intent".

    Parameters
    ----------
    text : str — raw or normalised text from user

    Returns
    -------
    str — one of the supported intent labels, or "unknown_intent"
    """
    normalised = normalize_text(text)

    if not normalised:
        return "unknown_intent"

    # ── Pass 1: keyword substring matching ────────────────
    for intent_label, patterns in _INTENT_PATTERNS:
        for pattern in patterns:
            if pattern in normalised:
                return intent_label

    # ── Pass 2: regex fallback ────────────────────────────
    for intent_label, compiled_re in _REGEX_PATTERNS:
        if compiled_re.search(normalised):
            return intent_label

    return "unknown_intent"


def is_stateful_intent(intent: str) -> bool:
    """Return True if the intent modifies or reads experiment state."""
    return intent in _STATEFUL_INTENTS


def is_stateless_intent(intent: str) -> bool:
    """Return True if the intent is an informational/stateless query."""
    return intent in _STATELESS_INTENTS


def is_utility_intent(intent: str) -> bool:
    """Return True if the intent is a utility command (help, repeat, etc.)."""
    return intent in _UTILITY_INTENTS


def get_supported_intents() -> list[str]:
    """Return a sorted list of all supported intent labels."""
    all_intents = _STATEFUL_INTENTS | _STATELESS_INTENTS | _UTILITY_INTENTS
    return sorted(all_intents)


def get_intent_category(intent: str) -> str:
    """
    Return the category of an intent: "stateful", "stateless", "utility",
    or "unknown".
    """
    if intent in _STATEFUL_INTENTS:
        return "stateful"
    elif intent in _STATELESS_INTENTS:
        return "stateless"
    elif intent in _UTILITY_INTENTS:
        return "utility"
    return "unknown"
