from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = Path(r"C:\Projects\projects_py\labkovsky-model\models\labkovsky-qwen7b-lora")
OUTPUT_PATH = Path(r"C:\Projects\projects_py\labkovsky-model\models\labkovsky-qwen7b-merged")

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="cpu",  # CPU to avoid VRAM issues
    trust_remote_code=True,
)

print("Loading LoRA...")
model = PeftModel.from_pretrained(base_model, str(LORA_PATH))

print("Merging...")
merged_model = model.merge_and_unload()

print(f"Saving to {OUTPUT_PATH}...")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
merged_model.save_pretrained(OUTPUT_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_PATH)

print("✅ Done! Merged model saved.")