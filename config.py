"""
config.py — Module 21: Config / Constants
==========================================
Single source of truth for all fixed values used across the system.
Import this everywhere instead of hardcoding values.
"""

import hashlib
import os

# ─────────────────────────────────────────────────────────
# WAKE WORD
# ─────────────────────────────────────────────────────────
WAKE_WORD = "hey mello"

# The assistant will also accept these natural variants
WAKE_WORD_VARIANTS = [
    "hey mello",
    "hi mello",
    "okay mello",
    "wake up",
    "hello mello",
]

# ─────────────────────────────────────────────────────────
# USER VERIFICATION
# ─────────────────────────────────────────────────────────
# Default passcode is "1234" — change BEFORE running in any shared/demo setting.
# We store a SHA-256 hash so the real passcode is never in plaintext.
_DEFAULT_PASSCODE = "1234"
PASSCODE_HASH = hashlib.sha256(_DEFAULT_PASSCODE.encode()).hexdigest()

# Maximum failed verification attempts before a cooldown is applied
MAX_FAILED_ATTEMPTS = 3
LOCKOUT_SECONDS = 30  # seconds to wait after MAX_FAILED_ATTEMPTS

# Biometric enrollment storage (per-user folders under this root)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENROLLMENT_ROOT = os.path.join(_BASE_DIR, "data", "enrollment")

# Face: FaceNet (InceptionResnetV1) 512-dim deep embeddings — cosine match threshold.
# MTCNN detects/aligns faces, InceptionResnetV1 produces discriminative embeddings.
# Same person typically scores 0.6–0.9; different people score 0.0–0.4.
FACE_CROP_SIZE = 160          # used internally by MTCNN
FACE_COSINE_THRESHOLD = 0.8  # lowered from 0.82 (raw pixels) — deep embeddings are sparser

# Voice: Resemblyzer embedding cosine threshold (same speaker usually >0.85)
VOICE_COSINE_THRESHOLD = 0.9

# Default profile id for single-user demo (extend to multi-user via UI later)
DEFAULT_USER_ID = "default"

# ─────────────────────────────────────────────────────────
# AUDIO CAPTURE
# ─────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000       # Hz — Whisper is trained on 16 kHz audio
CHANNELS = 1               # Mono recording
RECORD_DURATION = 5        # seconds for a fixed-length recording
AUDIO_DTYPE = "float32"    # sounddevice native dtype (range –1.0 to +1.0)
SILENCE_THRESHOLD = 0.015  # RMS amplitude below this = silence
SILENCE_DURATION = 1.5     # seconds of consecutive silence → stop recording
AUDIO_TMP_PATH = "/tmp/automl_captured_audio.wav"

# ─────────────────────────────────────────────────────────
# WHISPER / SPEECH-TO-TEXT
# ─────────────────────────────────────────────────────────
# Model sizes (accuracy vs speed trade-off):
#   tiny  → fastest,  lowest accuracy  (~39 MB)
#   base  → balanced, good accuracy    (~74 MB)  ← recommended for course
#   small → slower,   better accuracy  (~244 MB)
WHISPER_MODEL_SIZE = "base"
WHISPER_LANGUAGE = "en"    # language hint; None = auto-detect
WHISPER_DEVICE = "cpu"     # "cpu" or "cuda" (if GPU available)

# ─────────────────────────────────────────────────────────
# ML EXPERIMENT DEFAULTS  (used in State Manager — Module 9)
# ─────────────────────────────────────────────────────────
SUPPORTED_MODELS = ["xgboost", "random_forest", "logistic_regression", "cnn", "mlp", "resnet"]
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20

# ─────────────────────────────────────────────────────────
# SUPPORTED INTENTS  (used in Intent Detection — Module 6)
# ─────────────────────────────────────────────────────────
SUPPORTED_INTENTS = [
    # Stateful control intents
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
    "stop_training",
    "resume_training",
    "show_status",
    "show_accuracy",
    "show_loss_curve",
    # Stateless info intents
    "search_dataset",
    "get_dataset_info",
    "show_competition",
    "show_leaderboard",
    "search_code",
    # Utility intents
    "help",
    "repeat",
    "unknown_intent",
]

# ─────────────────────────────────────────────────────────
# SUPPORTED DATASETS  (used in Slot Filling — Module 7)
# ─────────────────────────────────────────────────────────
SUPPORTED_DATASETS = [
    "titanic", "iris", "mnist", "cifar10",
    "boston", "wine", "diabetes", "breast_cancer",
]

# ─────────────────────────────────────────────────────────
# TRAINING SIMULATION  (used in Experiment Controller — Module 10)
# ─────────────────────────────────────────────────────────
TRAINING_SIMULATION_INTERVAL = 0.5  # seconds between simulated epochs
