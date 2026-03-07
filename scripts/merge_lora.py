"""Merge SFT LoRA into base model in full precision."""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
BASE_MODEL_PATH = Path("models/vikhr-book-merged")
LORA_PATH = Path("models/labkovsky-rag-context-lora-v10/checkpoint-130")
OUTPUT_PATH = Path("models/vikhr-v10-merged")

def main():
    print(f"Loading base model from {BASE_MODEL_PATH}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"Loading tokenizer from {BASE_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    print(f"Loading LoRA adapter from {LORA_PATH}...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)

    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {OUTPUT_PATH}...")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(OUTPUT_PATH, safe_serialization=True)

    print(f"Saving tokenizer to {OUTPUT_PATH}...")
    tokenizer.save_pretrained(OUTPUT_PATH)

    print("Done!")

if __name__ == "__main__":
    main()
