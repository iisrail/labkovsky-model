# src/config.py
from pathlib import Path

# === PATHS ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FINE_TUNING_DATA_DIR = DATA_DIR / "fine_tuning"
MODELS_DIR = PROJECT_ROOT / "models"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# === MODEL ===
MODEL_NAME = "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it"  # Instruction-tuned, follows prompts
LORA_PATH = MODELS_DIR / "labkovsky-vikhr-yandex-lora"

# === RS TOKENS ===
RS_TOKENS = ["[ОБЪЯСНЕНИЕ]", "[ВМЕШАТЕЛЬСТВО]", "[ЭСКАЛАЦИЯ]"]

# === GENERATION ===
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
REPETITION_PENALTY = 1.2
