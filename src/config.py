# src/config.py
from pathlib import Path

# === PATHS ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FINE_TUNING_DATA_DIR = DATA_DIR / "fine_tuning"
MODELS_DIR = PROJECT_ROOT / "models"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# === MODEL ===
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = MODELS_DIR / "labkovsky-qwen7b-lora-rs"

# === RS TOKENS ===
RS_TOKENS = ["<RS=INTERVENTION>", "<RS=EXPLANATION>", "<RS=ESCALATION>"]

# === GENERATION ===
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
REPETITION_PENALTY = 1.1
