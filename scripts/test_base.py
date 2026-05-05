"""Test base Vikhr model without RAG or LoRA."""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
    device_map="auto",
)
tok = AutoTokenizer.from_pretrained("Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it")

question = (
    "I didn't get a callback after two job interviews, and a colleague just ignored "
    "my proposal at work. After every rejection or being ignored, I start doubting "
    "myself. I wonder — am I even a good specialist? Is this normal? "
    "Please answer in English."
)

messages = [{"role": "user", "content": question}]
inp = tok.apply_chat_template(messages, tokenize=True, return_tensors="pt", add_generation_prompt=True).to("cuda")
out = model.generate(inp, max_new_tokens=500, temperature=0.5, do_sample=True)
print(tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True))
