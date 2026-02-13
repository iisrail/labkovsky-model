# Labkovsky AI Model

## Project Overview
Fine-tuned LLM that replicates Mikhail Labkovsky's psychological consultation style using RAG + LoRA fine-tuning.

## Current Status: Working RAG + LoRA Pipeline
Small LoRA (r=8) adds Labkovsky's tone while Vikhr's native RAG capability handles document grounding.

## Tech Stack
- Python 3.10+
- PyTorch + CUDA
- Unsloth (for fast LoRA training)
- Transformers, PEFT, TRL (use `max_length` not `max_seq_length` in SFTConfig)
- ChromaDB for RAG
- Sentence-transformers (multilingual-e5-large)
- Base model: Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it (Russian-native, RAG-trained)

## Key Architecture
```
User Question → ChromaDB (top 3 docs) → Vikhr + Small LoRA → Grounded Response
```

**Critical Insight:** Vikhr has native RAG via `documents` role. Small LoRA adds tone without breaking this.

## Vikhr Native RAG Format (CRITICAL)
```python
messages = [
    {"role": "system", "content": "Ответь кратко, используя только информацию из документов."},
    {"role": "documents", "content": json.dumps([
        {"doc_id": 0, "title": "...", "content": "<chunk>"},
        {"doc_id": 1, "title": "...", "content": "<chunk>"}
    ], ensure_ascii=False)},
    {"role": "user", "content": query}
]
```
Do NOT put docs in user message - use the `documents` role!

## Data Files
| File | Purpose |
|------|---------|
| `qa_rs_final.jsonl` | Training & RAG index (Russian RS markers) |
| `anti_generic_clean.jsonl` | Boundary examples (128) |

### RS Markers (Russian)
- `[ОБЪЯСНЕНИЕ]` - Psychology/explanation
- `[ВМЕШАТЕЛЬСТВО]` - Intervention/advice
- `[ЭСКАЛАЦИЯ]` - Professional help needed

## LoRA Training Config
**Script:** `src/fine_tuning/train_lora_unsloth.py`

| Setting | Value | Why |
|---------|-------|-----|
| LoRA r | 8 | Small to preserve RAG |
| LoRA alpha | 16 | 2x ratio |
| Completion-only | Yes | Custom collator with `<s>assistant\n` template |
| Split | By unique answers | No data leakage |

## What Works / Doesn't Work

### Works
- Vikhr native `documents` role
- Small LoRA (r=8)
- Completion-only training (prevents generating user questions)
- Simple system prompt

### Doesn't Work
- Full sequence training (model generates user questions)
- Docs in user message (hallucinations like `[{user_exp}:`)
- Complex prompts with RS instructions (echoes markers)
- Large LoRA (breaks RAG grounding)

## Project Structure
```
src/
├── fine_tuning/
│   ├── train_lora_unsloth.py    # Main training (Unsloth + completion-only)
│   └── inference_unsloth.py     # Standalone inference
├── rag/
│   ├── build_index.py           # Build ChromaDB
│   └── query_rag.py             # RAG pipeline (Vikhr documents role)
data/fine_tuning/
├── qa_rs_final.jsonl            # Training data
└── anti_generic_clean.jsonl     # Boundaries
models/
└── labkovsky-vikhr-lora-unsloth/ # Trained adapter
```

## Commands
```bash
# Build RAG index
python src/rag/build_index.py

# Train (WSL)
source venv_wsl/bin/activate
python src/fine_tuning/train_lora_unsloth.py

# Query
python src/rag/query_rag.py
python src/rag/query_rag.py --no-lora  # baseline
```

## Code Conventions
- pathlib.Path for all paths
- Russian in user-facing strings, English in code
- Type hints preferred
