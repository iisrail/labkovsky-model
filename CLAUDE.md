# Labkovsky AI Model

## Project Overview
Fine-tuned LLM that replicates Mikhail Labkovsky's psychological consultation style using RAG + LoRA fine-tuning.

## Current Status: Two-Stage LoRA + RAG Pipeline
1. **Book LoRA** (MLP modules) merged into base → `vikhr-book-merged`
2. **RAG Context LoRA** (attention + gate_proj) trained on top with retrieved docs in system prompt

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
User Question → ChromaDB (top 2 docs + siblings) → System prompt with docs → Vikhr-book-merged + RAG LoRA → Response
```

**Two-stage approach:**
- Stage 1: Book LoRA (MLP: up_proj, down_proj, gate_proj) trained on book chunks, merged into base
- Stage 2: RAG Context LoRA (attention + gate_proj) trained with retrieved docs in system prompt

## Training Pipeline
```bash
# 1. Clean QA data (strip RS markers)
python src/fine_tuning/clean_qa_data.py
# Input: qa_rs_final.jsonl → Output: qa_clean.jsonl

# 2. Build ChromaDB index (1047 docs)
python src/rag/build_index.py
# Sources: qa_clean + articles + interviews + book

# 3. Build training data with RAG context
python src/fine_tuning/build_rag_training_data.py
# Retrieves docs from book/articles/interviews ONLY (not QA — prevents leakage)
# Output: qa_with_rag_context.jsonl

# 4. Train RAG context LoRA
python src/fine_tuning/train_with_rag_context.py
# Output: models/labkovsky-rag-context-lora

# 5. Query
python src/rag/query_rag.py
python src/rag/query_rag.py --no-lora  # baseline
```

## Training Experiments Summary

### Stage 1: Direct LoRA (no RAG context in training)
| Run | Modules | Alpha | Data | Samples | Best Eval Loss |
|-----|---------|-------|------|---------|----------------|
| Full | q,k,v,o + MLP | 16 | qa + anti_generic | ~600 | **2.019** |
| qv_proj | q,v only | 16 | qa + anti_generic | ~600 | 2.033 |
| attn | q,k,v,o | 16 | qa + anti_generic | ~600 | 2.023 |
| attn+articles | q,k,v,o | 16 | + articles + interviews | ~891 | 2.238 |
| attn+book | q,k,v,o | 32 | + interviews + book | ~925 | 2.267 |
| r8a32 | q,k,v,o | 32 | qa + anti_generic + interviews + book | 925 | 2.216 |

### Stage 2: RAG Context LoRA (docs in system prompt during training)
| Run | Modules | Alpha | Training Docs From | Best Eval Loss |
|-----|---------|-------|--------------------|----------------|
| attn-only | q,k,v,o | 32 | book/articles/interviews | 2.567 |
| attn-a24 | q,k,v,o | 24 | book/articles/interviews | 2.567 |
| **attn+gate** | **q,k,v,o,gate_proj** | **24** | **book/articles/interviews** | **2.558** |
| attn+gate-bookonly | q,k,v,o,gate_proj | 24 | book/articles only | 2.565 |

### Key Findings
- **Attention-only (q,k,v,o)** preserves RAG grounding better than full modules
- **gate_proj** adds content control without breaking RAG (slightly better loss)
- **MLP modules (up_proj, down_proj)** stay frozen — already trained in book LoRA
- **Training on book/article docs only** (not QA) prevents leakage — model learns to synthesize from prose
- **Question-only embeddings** dramatically improve retrieval quality
- **RS markers in training data cause hallucination** — model generates wrong markers. Clean at source.
- **Articles/book chunks hurt direct training** — they're connected, not self-contained Q&A. Use in RAG only.

### Current Best Config
```python
LORA_R = 8
LORA_ALPHA = 24
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
# Base: vikhr-book-merged (book LoRA with MLP modules already merged)
```

## Critical Design Decisions

### Train vs Inference Doc Sources
- **Training**: Retrieve docs from book/articles/interviews ONLY (prevents QA leakage)
- **Inference**: Retrieve from ALL sources including QA (best results)
- Why: If QA answers appear as retrieved docs during training, model learns to copy them verbatim

### Question-Only Embeddings
- Index embeds `question` and `potential_questions` fields only (not answers)
- Matches user queries semantically (question→question, not question→answer)
- Dramatic improvement in retrieval relevance

### Sibling Chunk Expansion
- At inference: Retrieve chunk N → also fetch N-1 and N+1
- At training: Fetch up to 2 next chunks within same chapter/article
- Provides context for chunks that reference previous content

### System Prompt Format (must match train/inference)
```python
SYSTEM_PROMPT_TEMPLATE = (
    "Ты психолог Михаил Лабковский. Используй следующие документы для ответа:\n\n"
    "{docs}\n\n"
    "Отвечай в стиле Лабковского: прямо, уверенно, с конкретными рекомендациями. "
    "Сначала объясни причину проблемы, затем дай конкретные шаги для решения. "
    "Если видишь, что кому-то в ситуации нужна профессиональная помощь — скажи об этом прямо."
)
```

## Data Files
| File | Purpose | Count |
|------|---------|-------|
| `qa_rs_final.jsonl` | Raw QA with RS markers (source) | 478 |
| `qa_clean.jsonl` | Cleaned QA (no RS markers) — for training & index | 478 |
| `anti_generic_finance.jsonl` | Boundary examples (finance→psychology) | 7 |
| `anti_generic_clean.jsonl` | Boundary examples (generic) | 128 |
| `interviews.jsonl` | Interview Q&A | 35 |
| `articles_with_questions.jsonl` | RAG index only (connected chunks) | ~250 |
| `Hochu_i_budu_with_questions.jsonl` | RAG index only (book chapters) | ~284 |
| `qa_with_rag_context.jsonl` | Training data with RAG docs in system prompt | ~485 |

### ChromaDB Index: 1047 documents
- 478 qa_corpus + ~250 articles + 35 interviews + ~284 book

### RS Markers (stripped from training, kept in raw data)
- `[ОБЪЯСНЕНИЕ]` - Psychology/explanation
- `[ВМЕШАТЕЛЬСТВО]` - Intervention/advice
- `[ЭСКАЛАЦИЯ]` - Professional help needed
- `Не требуется.` stubs also stripped

## RAG Config
| Setting | Value |
|---------|-------|
| TOP_K | 2 |
| INCLUDE_SIBLINGS | True (N-1, N+1) |
| MAX_DISTANCE (training) | 0.35 |
| Embedding model | intfloat/multilingual-e5-large |

## LoRA Training Config
**Script:** `src/fine_tuning/train_with_rag_context.py`

| Setting | Value | Why |
|---------|-------|-----|
| LoRA r | 8 | Small to preserve RAG |
| LoRA alpha | 24 | 3x ratio for style |
| Modules | q,k,v,o,gate_proj | Attention + content control |
| Completion-only | Yes | Custom collator with `<s>assistant\n` template |
| Split | By unique answers | No data leakage |
| MAX_SEQ_LENGTH | 2048 | Fit docs in system prompt |
| Epochs | 20 (early stop patience=2) | |
| Effective batch | 16 | 1 * 16 gradient accumulation |
| LR | 5e-5, cosine schedule | |

## What Works / Doesn't Work

### Works
- Two-stage LoRA (book merged, then RAG context adapter)
- Small LoRA (r=8) with attention + gate_proj
- Completion-only training (prevents generating user questions)
- Docs in system prompt (matching train/inference format)
- Training retrieval from book/articles only (prevents QA leakage)
- Inference retrieval from all sources
- Question-only embeddings for index
- Sibling chunk expansion for connected content
- Distance threshold (0.35) with no-docs fallback

### Doesn't Work
- Full sequence training (model generates user questions)
- Docs in user message (hallucinations like `[{user_exp}:`)
- Complex prompts with RS instructions (echoes markers)
- Large LoRA (breaks RAG grounding)
- Connected chunks for direct training (articles, book)
- RS markers in training data (model hallucinates wrong markers)
- QA answers in training retrieval (model copies verbatim)
- Vikhr native `documents` role for advice (works for factual, not advice style)

## Project Structure
```
src/
├── fine_tuning/
│   ├── train_with_rag_context.py    # Current training (RAG context LoRA)
│   ├── train_lora_unsloth.py        # Previous training (direct LoRA)
│   ├── build_rag_training_data.py   # Build training data with RAG docs
│   ├── clean_qa_data.py             # Strip RS markers from QA data
│   └── inference_unsloth.py         # Standalone inference
├── rag/
│   ├── build_index.py               # Build ChromaDB (question-only embeddings)
│   └── query_rag.py                 # RAG pipeline (system prompt + siblings)
data/
├── fine_tuning/
│   ├── qa_rs_final.jsonl            # Raw QA (RS markers)
│   ├── qa_clean.jsonl               # Cleaned QA (no markers)
│   ├── qa_with_rag_context.jsonl    # Training data with RAG context
│   ├── anti_generic_finance.jsonl   # Finance boundaries
│   └── anti_generic_clean.jsonl     # Generic boundaries
├── processed/
│   ├── interviews.jsonl             # Interview Q&A
│   ├── articles_with_questions.jsonl  # RAG only
│   └── Hochu_i_budu_with_questions.jsonl  # RAG only
models/
├── vikhr-book-merged/               # Base + book LoRA merged
└── labkovsky-rag-context-lora/      # Current RAG context adapter
```

## Commands
```bash
# Full pipeline
source venv_wsl/bin/activate
python src/fine_tuning/clean_qa_data.py
python src/rag/build_index.py
python src/fine_tuning/build_rag_training_data.py
python src/fine_tuning/train_with_rag_context.py

# Query
python src/rag/query_rag.py
python src/rag/query_rag.py --no-lora  # baseline
```

## Code Conventions
- pathlib.Path for all paths
- Russian in user-facing strings, English in code
- Type hints preferred
