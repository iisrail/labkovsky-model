#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Pipeline with Response Signal Routing

Flow:
1. Classify question → RS (INTERVENTION | EXPLANATION | ESCALATION)
2. Retrieve relevant chunks from ChromaDB
3. Generate response with RS-specific system prompt

Usage:
    python query_rag_with_rs.py --rs-model path/to/rs_classifier.pkl
"""

import argparse
import pickle
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import chromadb

# ============================================================
# PATHS (adjust to your setup)
# ============================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent  # Go up to project root
CHROMA_DIR = PROJECT_DIR / "chroma_db"
MODELS_DIR = PROJECT_DIR / "models"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = MODELS_DIR / "labkovsky-qwen7b-lora"

TOP_K = 2

# ============================================================
# RS-SPECIFIC SYSTEM PROMPTS
# ============================================================

RS_SYSTEM_PROMPTS = {
    "INTERVENTION": """Ты — Михаил Лабковский.

Режим: INTERVENTION.

Твоя задача — вмешаться и изменить поведение человека.

Правила:
- Можно кратко объяснять, если это помогает.
- Не сочувствуй и не утешай.
- Выбери одну ключевую причину проблемы и воздействуй на неё.
- В конце всегда дай конкретную инструкцию: что делать или что прекратить делать.""",

    "EXPLANATION": """Ты — Михаил Лабковский.

Режим: EXPLANATION.

Твоя задача — дать человеку понимание, почему с ним происходит эта ситуация.

Правила:
- Объясняй спокойно и логично.
- Используй примеры из жизни.
- Не дави и не командуй.
- В конце обозначь, какое поведение является здоровым, даже если без жёсткой инструкции.""",

    "ESCALATION": """Ты — Михаил Лабковский.

Режим: ESCALATION.

Твоя задача — прекратить попытки самопомощи в рамках этого ответа.

Правила:
- Не анализируй глубоко.
- Не давай поведенческих инструкций.
- Чётко скажи, что без специалиста дальше идти нельзя, и почему."""
}

# ============================================================
# RS CLASSIFIER
# ============================================================

class RSClassifier:
    def __init__(self, model_path: Path):
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data["classifier"]
        self.id2rs = model_data["id2rs"]
        self.embedding_model_name = model_data["embedding_model_name"]
        self.embed_model = None  # lazy load, share with RAG
    
    def set_embed_model(self, embed_model):
        """Share embedding model with RAG to avoid loading twice"""
        self.embed_model = embed_model
    
    def predict(self, question: str) -> str:
        question_prefixed = f"query: {question}"
        embedding = self.embed_model.encode([question_prefixed])
        pred_id = self.classifier.predict(embedding)[0]
        return self.id2rs[pred_id]
    
    def predict_with_proba(self, question: str) -> tuple:
        question_prefixed = f"query: {question}"
        embedding = self.embed_model.encode([question_prefixed])
        pred_id = self.classifier.predict(embedding)[0]
        rs = self.id2rs[pred_id]
        
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(embedding)[0]
            prob_dict = {self.id2rs[i]: float(p) for i, p in enumerate(probs)}
        else:
            prob_dict = {rs: 1.0}
        
        return rs, prob_dict

# ============================================================
# GLOBAL STATE
# ============================================================

_embed_model = None
_collection = None
_llm = None
_tokenizer = None
_rs_classifier = None

def init_embedding():
    global _embed_model
    if _embed_model is None:
        print("[+] Loading embedding model...")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model

def init_retrieval(chroma_dir: Path):
    global _collection
    if _collection is None:
        print("[+] Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=str(chroma_dir))
        _collection = client.get_collection("labkovsky")
        print(f"   Loaded {_collection.count()} documents")
    return _collection

def init_rs_classifier(model_path: Path):
    global _rs_classifier
    if _rs_classifier is None:
        print(f"[+] Loading RS classifier from {model_path}...")
        _rs_classifier = RSClassifier(model_path)
        _rs_classifier.set_embed_model(init_embedding())
    return _rs_classifier

def init_llm(lora_path: Path = None):
    global _llm, _tokenizer
    if _llm is None:
        print(f"[+] Loading LLM: {LLM_MODEL_NAME}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        if lora_path and lora_path.exists():
            print(f"   Loading LoRA: {lora_path}")
            _llm = PeftModel.from_pretrained(base_model, str(lora_path))
        else:
            print("   (base model, no LoRA)")
            _llm = base_model
        
        _llm.eval()
        _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
        print("[OK] LLM loaded")
    
    return _llm, _tokenizer

# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(query: str, top_k: int = TOP_K) -> list:
    embed_model = init_embedding()
    collection = _collection
    
    query_embedding = embed_model.encode(f"query: {query}")
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    documents = []
    for i in range(len(results['ids'][0])):
        documents.append({
            "text": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "distance": results['distances'][0][i]
        })
    
    return documents

# ============================================================
# GENERATION
# ============================================================

def generate(query: str, context_docs: list, rs: str) -> str:
    llm, tokenizer = _llm, _tokenizer
    
    # Build context from retrieved docs
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        source = doc['metadata'].get('source', 'unknown')
        context_parts.append(f"[{i}] {doc['text']}")
    
    context = "\n\n".join(context_parts)
    
    user_message = f"""Контекст из моих материалов:

{context}

---

Вопрос: {query}"""

    # Select RS-specific system prompt
    system_prompt = RS_SYSTEM_PROMPTS.get(rs, RS_SYSTEM_PROMPTS["INTERVENTION"])
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(llm.device)
    
    input_len = inputs.shape[-1]
    
    with torch.inference_mode():
        outputs = llm.generate(
            inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_tokens = outputs[0][input_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    return response

# ============================================================
# MAIN FUNCTION
# ============================================================

def ask_labkovsky(query: str, top_k: int = TOP_K, verbose: bool = False, override_rs: str = None) -> dict:
    """
    Main function: ask a question, get Labkovsky's response
    
    Args:
        query: Question
        top_k: How many docs for context
        verbose: Show retrieved docs and RS info
        override_rs: Force specific RS (for testing)
    
    Returns:
        dict with 'answer', 'rs', 'rs_probs'
    """
    # Step 1: Classify RS
    if override_rs:
        rs = override_rs
        rs_probs = {override_rs: 1.0}
    else:
        rs, rs_probs = _rs_classifier.predict_with_proba(query)
    
    if verbose:
        print(f"\n[RS] Response Signal: {rs}")
        print(f"     Probabilities: {rs_probs}")
    
    # Step 2: Retrieve
    docs = retrieve(query, top_k)
    
    if verbose:
        print(f"\n[*] Retrieved {len(docs)} documents:")
        for i, doc in enumerate(docs, 1):
            preview = doc['text'][:60] + "..."
            print(f"    [{i}] (dist={doc['distance']:.3f}) {preview}")
    
    # Step 3: Generate with RS-specific prompt
    answer = generate(query, docs, rs)
    
    return {
        "answer": answer,
        "rs": rs,
        "rs_probs": rs_probs
    }

# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rs-model", type=str, required=True, help="Path to rs_classifier.pkl")
    parser.add_argument("--chroma-dir", type=str, default=str(CHROMA_DIR), help="ChromaDB directory")
    parser.add_argument("--lora-path", type=str, default=str(LORA_PATH), help="LoRA adapter path")
    parser.add_argument("--no-lora", action="store_true", help="Use base model without LoRA")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Labkovsky with Response Signal Routing")
    print("=" * 60)
    print("\nCommands:")
    print("  'exit' — quit")
    print("  'verbose' — toggle source display")
    print("  'rs:INTERVENTION' / 'rs:EXPLANATION' / 'rs:ESCALATION' — force RS")
    print("  'rs:auto' — return to automatic RS detection")
    print()
    
    # Initialize all components
    print("Loading models...")
    init_embedding()
    init_retrieval(Path(args.chroma_dir))
    init_rs_classifier(Path(args.rs_model))
    
    lora_path = None if args.no_lora else Path(args.lora_path)
    init_llm(lora_path)
    
    print("\n[OK] Ready!\n")
    
    verbose = False
    override_rs = None
    
    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nPoka!")
            break
        
        if not query:
            continue
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("Poka!")
            break
        
        if query.lower() == 'verbose':
            verbose = not verbose
            print(f"Verbose: {'ON' if verbose else 'OFF'}")
            continue
        
        if query.lower().startswith('rs:'):
            rs_cmd = query.split(':')[1].upper()
            if rs_cmd == 'AUTO':
                override_rs = None
                print("RS: automatic detection")
            elif rs_cmd in RS_SYSTEM_PROMPTS:
                override_rs = rs_cmd
                print(f"RS: forced to {override_rs}")
            else:
                print(f"Unknown RS. Options: INTERVENTION, EXPLANATION, ESCALATION, AUTO")
            continue
        
        print("\n...")
        
        try:
            result = ask_labkovsky(query, verbose=verbose, override_rs=override_rs)
            
            if not verbose:
                print(f"[{result['rs']}]")
            print(f"\nLabkovsky: {result['answer']}\n")
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()