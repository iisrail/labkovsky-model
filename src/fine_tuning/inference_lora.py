import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from config import MODELS_DIR

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = MODELS_DIR / "labkovsky-qwen7b-lora"

def load_model():
    print("🤖 Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("✅ Ready")
    return model, tokenizer

def ask(model, tokenizer, question):
    messages = [
        {"role": "system", "content": "Ты — Михаил Лабковский, российский психолог."},
        {"role": "user", "content": question}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,                          # Just pass tensor directly, no **
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    model, tokenizer = load_model()
    
    while True:
        q = input("\nТы: ").strip()
        if q.lower() in ['exit', 'quit', 'q']:
            break
        print(f"\nЛабковский: {ask(model, tokenizer, q)}")