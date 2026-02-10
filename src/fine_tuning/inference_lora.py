import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Add paths for imports
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

from config import MODEL_NAME, LORA_PATH, TEMPERATURE, MAX_NEW_TOKENS, REPETITION_PENALTY
from predict_rs_sklearn import ResponseSignalPredictor

RS_CLASSIFIER_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "rs_classifier" / "rs_classifier.pkl"

def load_model():
    print("🤖 Loading model...")

    # Match training: detect bf16 support
    bf16_supported = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load tokenizer from LORA_PATH (has RS special tokens)
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
    print(f"   Tokenizer vocab size: {len(tokenizer)}")
    print(f"   RS tokens: {[t for t in tokenizer.get_vocab() if '<RS=' in t]}")
    
    # Resize base model to match tokenizer
    base_model.resize_token_embeddings(len(tokenizer))
    
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    print(f"   LoRA config: {model.peft_config}")
    print(f"   Active adapter: {model.active_adapter}")
    
    model.eval()
    
    print("✅ Model ready")
    return model, tokenizer

def load_rs_classifier():
    print("🎯 Loading RS classifier...")
    predictor = ResponseSignalPredictor(RS_CLASSIFIER_PATH)
    print("✅ RS classifier ready")
    return predictor

# Identity prefix (must match training format)
IDENTITY_PREFIX = "Ты Михаил Лабковский, известный российский психолог и коуч со своей собственной авторской методикой. Ты автор книги «Хочу и буду» — самой продаваемой книги в России.\n"

def ask(model, tokenizer, rs_classifier, question, verbose=False):
    # 1. Classify question (optional, for display)
    rs_label = rs_classifier.predict(question)

    if verbose:
        probs = rs_classifier.predict_proba(question)
        print(f"   RS: {rs_label} {probs}")

    # 2. Build prompt (matches training format with identity prefix)
    prompt = f"{IDENTITY_PREFIX}Вопрос: {question}\nОтвет:"
    
    # 3. Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=REPETITION_PENALTY,
        )
    
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    # Cut at junk
    for stop in ["#", "Источник", "http", "Что можно"]:
        if stop in response:
            response = response.split(stop)[0]
    return response.strip(), rs_label

if __name__ == "__main__":
    model, tokenizer = load_model()
    rs_classifier = load_rs_classifier()
    
    print("\nCommands: 'exit', 'verbose' to toggle debug\n")
    
    verbose = False
    
    while True:
        q = input("\nТы: ").strip()
        if q.lower() in ['exit', 'quit', 'q']:
            break
        if q.lower() == 'verbose':
            verbose = not verbose
            print(f"Verbose: {'on' if verbose else 'off'}")
            continue
            
        answer, rs = ask(model, tokenizer, rs_classifier, q, verbose)
        print(f"\nЛабковский [{rs}]: {answer}")