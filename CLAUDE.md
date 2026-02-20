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
User Question → ChromaDB (top 3 docs + siblings) → Vikhr + Small LoRA → Grounded Response
```

**Critical Insight:** Vikhr has native RAG via `documents` role. Small LoRA adds tone without breaking this.

## Training Experiments Summary

| Run | Modules | Alpha | Data | Samples | Best Eval Loss |
|-----|---------|-------|------|---------|----------------|
| Full | q,k,v,o + MLP | 16 | qa + anti_generic | ~600 | **2.019** |
| qv_proj | q,v only | 16 | qa + anti_generic | ~600 | 2.033 |
| attn | q,k,v,o | 16 | qa + anti_generic | ~600 | 2.023 |
| attn+articles | q,k,v,o | 16 | + articles + interviews | ~891 | 2.238 |
| attn+book | q,k,v,o | 32 | + interviews + book | ~925 | 2.267 |
| r8a32 | q,k,v,o | 32 | qa + anti_generic + interviews + book | 925 | 2.216 |

### Key Findings
- **Attention-only (q,k,v,o)** preserves RAG grounding better than full modules
- **MLP modules** improve loss but may affect content/reasoning
- **Articles/book hurt training** - chunks are connected, not self-contained Q&A
- **Clean Q&A data works best** - qa_rs_final + anti_generic + interviews

### Recommended Config
```python
LORA_R = 8
LORA_ALPHA = 32  # ratio 4 for stronger style
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # attention only
```

## Data Quality Insights

### Good for Training (self-contained Q&A)
- `qa_rs_final.jsonl` - Real Q&A with RS markers
- `anti_generic_clean.jsonl` - Boundary examples
- `interviews.jsonl` - Real interview Q&A

### Problematic for Training (connected chunks)
- `articles_with_questions.jsonl` - Chunks reference previous context
- `Hochu_i_budu_with_questions.jsonl` - Book chapters are sequential

**Solution:** Use articles/book in RAG index only (with sibling retrieval), not for training.

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

## RAG Sibling Retrieval
For connected chunks (articles/book), query_rag.py fetches neighboring chunks:
- Retrieve chunk N → also fetch N-1 and N+1
- Provides context for chunks that reference previous content
- Enable/disable with `INCLUDE_SIBLINGS = True/False`

## Data Files
| File | Purpose |
|------|---------|
| `qa_rs_final.jsonl` | Training & RAG index (Russian RS markers) |
| `anti_generic_clean.jsonl` | Boundary examples (128) |
| `interviews.jsonl` | Interview Q&A (35) |
| `articles_with_questions.jsonl` | RAG only (connected chunks) |
| `Hochu_i_budu_with_questions.jsonl` | RAG only (book chapters) |

### RS Markers (Russian)
- `[ОБЪЯСНЕНИЕ]` - Psychology/explanation
- `[ВМЕШАТЕЛЬСТВО]` - Intervention/advice
- `[ЭСКАЛАЦИЯ]` - Professional help needed

## LoRA Training Config
**Script:** `src/fine_tuning/train_lora_unsloth.py`

| Setting | Value | Why |
|---------|-------|-----|
| LoRA r | 8 | Small to preserve RAG |
| LoRA alpha | 32 | 4x ratio for stronger style |
| Modules | q,k,v,o_proj | Attention only (style, not reasoning) |
| Completion-only | Yes | Custom collator with `<s>assistant\n` template |
| Split | By unique answers | No data leakage |

## What Works / Doesn't Work

### Works
- Vikhr native `documents` role
- Small LoRA (r=8) with attention-only modules
- Completion-only training (prevents generating user questions)
- Simple system prompt
- Clean Q&A data (qa_rs_final, interviews)
- Sibling retrieval for connected chunks

### Doesn't Work
- Full sequence training (model generates user questions)
- Docs in user message (hallucinations like `[{user_exp}:`)
- Complex prompts with RS instructions (echoes markers)
- Large LoRA (breaks RAG grounding)
- Connected chunks for training (articles, book)

## Project Structure
```
src/
├── fine_tuning/
│   ├── train_lora_unsloth.py    # Main training (Unsloth + completion-only)
│   └── inference_unsloth.py     # Standalone inference
├── rag/
│   ├── build_index.py           # Build ChromaDB
│   └── query_rag.py             # RAG pipeline (Vikhr documents role + siblings)
data/
├── fine_tuning/
│   ├── qa_rs_final.jsonl        # Training data
│   └── anti_generic_clean.jsonl # Boundaries
├── processed/
│   ├── interviews.jsonl         # Interview Q&A
│   ├── articles_with_questions.jsonl  # RAG only
│   └── Hochu_i_budu_with_questions.jsonl  # RAG only
models/
└── labkovsky-vikhr-lora-r8a32/  # Current adapter (r=8, alpha=32)
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
