# Labkovsky AI Model

## Project Overview
Fine-tuned LLM that replicates Mikhail Labkovsky's psychological consultation style using RAG + LoRA fine-tuning.

## Current Status: Two-Stage LoRA + RAG Pipeline, deployed on Modal
1. **Book LoRA** (MLP modules) merged into base → `vikhr-book-merged`
2. **RAG Context LoRA** (attention + gate_proj) trained on top with retrieved docs in system prompt
3. **Merged final model** → `iisrail/vikhr-labkovsky-final`
4. **Production**: Modal-hosted (`modal_deploy_v2.py`) — GPU inference + Telegram bot, see *Production Deployment* below.

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
User Question → ChromaDB (2 book/article + 1 QA, with chunk expansion) → System prompt → Vikhr-book-merged + RAG LoRA → Response
```

**Two-stage approach:**
- Stage 1: Book LoRA (r=8, alpha=16, MLP: up_proj, down_proj, gate_proj) trained on book chunks, merged into base
- Stage 2: RAG Context LoRA (r=8, alpha=24, attention + gate_proj) trained with retrieved docs in system prompt

## Training Pipeline
```bash
# 1. Clean QA data (strip RS markers)
python src/fine_tuning/clean_qa_data.py
# Input: qa_rs_final.jsonl → Output: qa_clean.jsonl

# 2. Build ChromaDB index (~1012 docs)
python src/rag/build_index.py
# Sources: qa_rs_segmented (with RS markers) + articles + book

# 3. Build training data with RAG context
python src/fine_tuning/build_rag_training_data.py
# 2 book/article docs + 1 non-self QA (separate queries, with chunk expansion)
# Records with no docs are skipped (every training example must have RAG context)
# Output: qa_with_rag_context.jsonl

# 4. Train RAG context LoRA
python src/fine_tuning/train_with_rag_context.py
# Output: models/labkovsky-rag-context-lora-v5

# 5. Query
python src/rag/query_rag.py --lora-path models/labkovsky-rag-context-lora-v5
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
- **English system prompt** with doc-type differentiation works better than Russian prompt

### Current Best Config

**Stage 1: Book LoRA** (merged into `vikhr-book-merged`)
```python
LORA_R = 8
LORA_ALPHA = 16  # ratio 2
TARGET_MODULES = ["gate_proj", "up_proj", "down_proj"]  # MLP only
```

**Stage 2: RAG Context LoRA** (loaded on top)
```python
LORA_R = 8
LORA_ALPHA = 24  # ratio 3
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]  # attention + gate
# Base: vikhr-book-merged (book LoRA with MLP modules already merged)
```

## Critical Design Decisions

### Split Retrieval (matches train and inference)
Both training and inference use the same retrieval pattern:
1. **2 docs from book/articles** (with chunk expansion)
2. **1 closest QA doc** (separate query, with RS markers for structure)
- Training: skips self-match QA (same qa_id), distance threshold 0.30
- Inference: no filtering — closest QA included (often the exact match)

### Doc Type Labels in System Prompt
Retrieved docs are labeled by type so the model can differentiate their role:
- `[Book/Article] Doc 1: ...` — core principles, foundation for reasoning
- `[Пример QA] Документ 3: ...` — tone and format guide

### RS-Segmented QA in RAG (not in training targets)
- QA docs in the index use `qa_rs_segmented.jsonl` with `[EXPLANATION]`, `[INTERVENTION]`, `[ESCALATION]` markers
- Model sees structured examples showing response organization in retrieved docs
- Training targets remain clean (from `qa_clean.jsonl`) — model learns to produce answers without markers
- This teaches structure implicitly without the model echoing markers

### No-Docs Records Dropped from Training
- Records where retrieval returns zero docs are skipped during training data build
- Every training example has RAG context, matching inference (which always has docs)

### Question-Only Embeddings
- Index embeds `question` and `potential_questions` fields only (not answers)
- Matches user queries semantically (question→question, not question→answer)
- Dramatic improvement in retrieval relevance

### Chunk Expansion (same in train and inference)
- Fetch chunk N → concatenate N+1, N+2 into same doc text
- Only within same chapter/article (checks `chapter_id`/`article_id`)
- Provides context for chunks that continue an idea forward
- Result: `[Book/Article] Doc 1: chunk3\nchunk4\nchunk5` (single doc, not separate)

### System Prompt Format (must match train/inference)
```python
SYSTEM_PROMPT_TEMPLATE = (
    "You are psychologist Mikhail Labkovsky. Below are reference materials.\n\n"
    "Book and article fragments contain core principles — use them as the foundation "
    "for your reasoning, but do not copy verbatim.\n"
    "QA examples show how to structure a response — use them as a guide for tone and format.\n\n"
    "{docs}\n\n"
    "Answer in Labkovsky's style: direct, confident, with specific recommendations. "
    "First explain the root cause, then give concrete steps. "
    "If professional help is needed — say so directly."
)
```

## Data Files
| File | Purpose | Count |
|------|---------|-------|
| `qa_rs_final.jsonl` | Raw QA with RS markers (source) | 478 |
| `qa_rs_segmented.jsonl` | QA with RS markers (`answer` field) — for RAG index | 479 |
| `qa_clean.jsonl` | Cleaned QA (no RS markers) — for training targets | 478 |
| `anti_generic_finance.jsonl` | Boundary examples (finance→psychology) | 7 |
| `anti_generic_clean.jsonl` | Boundary examples (generic) | 128 |
| `articles_with_questions.jsonl` | RAG index only (connected chunks) | ~250 |
| `Hochu_i_budu_with_questions.jsonl` | RAG index only (book chapters) | ~284 |
| `qa_with_rag_context.jsonl` | Training data with RAG docs in system prompt | ~462 |

### ChromaDB Index: ~1012 documents
- 479 qa_corpus (with RS markers) + ~250 articles + ~284 book

### RS Markers (in RAG QA docs, stripped from training targets)
- `[EXPLANATION]` - Psychology/explanation
- `[INTERVENTION]` - Intervention/advice
- `[ESCALATION]` - Professional help needed

## RAG Config
| Setting | Value |
|---------|-------|
| TOP_K | 2 (book/articles) |
| QA docs | 1 (closest, separate query) |
| Chunk expansion | +2 next chunks, same chapter/article |
| MAX_DISTANCE (training, book/articles) | 0.35 |
| QA_MAX_DISTANCE (training) | 0.30 |
| Embedding model | intfloat/multilingual-e5-large |

## LoRA Training Config

### Stage 1: Book LoRA
**Script:** `src/fine_tuning/train_book_lora.py`

| Setting | Value | Why |
|---------|-------|-----|
| LoRA r | 8 | Small adapter |
| LoRA alpha | 16 | 2x ratio for vocabulary |
| Modules | gate_proj, up_proj, down_proj | MLP only (vocabulary/knowledge) |
| MAX_SEQ_LENGTH | 1024 | Raw book text |
| Epochs | 4 | |
| Effective batch | 8 | 4 * 2 gradient accumulation |
| LR | 5e-4, cosine schedule | Higher LR for vocabulary |

### Stage 2: RAG Context LoRA
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
- English system prompt with doc-role differentiation
- Doc type labels (`[Book/Article]` / `[QA Example]`) in retrieved docs
- RS-segmented QA in RAG index (structure guide), clean targets in training
- Training retrieval from book/articles only (prevents QA leakage)
- Inference retrieval from all sources
- Question-only embeddings for index
- Sibling chunk expansion for connected content
- Dropping no-docs records from training

### Doesn't Work
- Full sequence training (model generates user questions)
- Docs in user message (hallucinations like `[{user_exp}:`)
- Complex prompts with RS instructions (echoes markers)
- Large LoRA (breaks RAG grounding)
- Connected chunks for direct training (articles, book)
- RS markers in training data targets (model hallucinates wrong markers)
- QA answers in training retrieval (model copies verbatim)
- Vikhr native `documents` role with two-step prompt (see below)
- Interviews in RAG index (not useful)
- No-docs fallback prompt in training (mismatch with inference)

## Why Vikhr's Native `documents` Role Didn't Work

Vikhr-YandexGPT has a special `documents` role for grounded RAG generation. We tested whether it could handle psychological advice (reasoning from principles) vs just factual extraction.

### Test Setup (`scripts/test_vikhr_grounded_rag.py`)

| Test | Documents | Query | Prompt |
|------|-----------|-------|--------|
| 1 | Factual (climate data) | Topic ("Глобальное потепление") | Vikhr two-step |
| 2 | Psychological principles | Personal question ("Почему я не могу уйти от мужа?") | Vikhr two-step |
| 3 | Psychological principles | Topic ("Отношения и любовь к себе") | Vikhr two-step |
| 4 | Psychological principles | Personal question | Labkovsky-style single-step |

### Results

| Test | Step 1 (doc selection) | Step 2 (answer) | Verdict |
|------|------------------------|-----------------|---------|
| 1 | `[0, 1]` ✓ | Good factual synthesis | WORKS |
| 2 | `[]` empty ✗ | "Can't find info" | FAILS |
| 3 | `[0, 1, 2]` ✓ | Good synthesis | WORKS |
| 4 | N/A (single-step) | Good reasoning from principles | WORKS |

### Key Finding: Two-Step Prompt Fails on Step 1

Vikhr's documented two-step approach:
```
System: "Give two answers: first list relevant doc IDs, then answer using those docs"
```

**Step 1 is designed for factual retrieval, not principle matching:**
- Query: "Почему я не могу уйти от мужа?" (Why can't I leave my husband?)
- Doc 1: "Почему люди остаются в плохих отношениях" (Why people stay in bad relationships)
- Model returns: `{"relevant_doc_ids": []}` — **EMPTY**

The model looks for **direct factual match** (does doc answer this exact question?), not **principle match** (does doc contain principles that APPLY to this question?).

**Test 4 proves the `documents` role itself works** — with a single-step Labkovsky prompt ("apply these principles to user's situation"), the model successfully reasons from the documents.

### Conclusion

| Approach | Reasoning from Principles | Labkovsky Style |
|----------|--------------------------|-----------------|
| Vikhr two-step + `documents` role | ✗ Fails at doc selection | ✗ Generic |
| Single-step + `documents` role | ✓ Works | ✗ Generic |
| System prompt + LoRA training | ✓ Works | ✓ Authentic |

**The problem is not the `documents` role — it's the two-step extractive prompt.** However, even with a working single-step approach, LoRA training is still needed to capture Labkovsky's distinctive voice (blunt, direct, tough love).

## Project Structure
```
src/
├── fine_tuning/
│   ├── train_book_lora.py           # Stage 1: Book LoRA (MLP, r=8, alpha=16)
│   ├── merge_book_lora.py           # Merge book LoRA into base → vikhr-book-merged
│   ├── train_with_rag_context.py    # Stage 2: RAG context LoRA (attn+gate, r=8, alpha=24)
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
│   ├── qa_rs_segmented.jsonl        # QA with RS markers — for RAG index
│   ├── qa_clean.jsonl               # Cleaned QA (no markers) — training targets
│   ├── qa_with_rag_context.jsonl    # Training data with RAG context
│   ├── anti_generic_finance.jsonl   # Finance boundaries
│   └── anti_generic_clean.jsonl     # Generic boundaries
├── processed/
│   ├── articles_with_questions.jsonl  # RAG only
│   └── Hochu_i_budu_with_questions.jsonl  # RAG only
models/
├── vikhr-book-merged/               # Base + book LoRA merged
├── labkovsky-rag-context-lora-v11/  # Current RAG context adapter (r=8, alpha=24)
└── vikhr-labkovsky-final/           # Fully merged model (book + RAG context LoRA)
```

## Commands
```bash
# Full pipeline
source venv_wsl/bin/activate
python src/fine_tuning/clean_qa_data.py
python src/rag/build_index.py
python src/fine_tuning/build_rag_training_data.py
python src/fine_tuning/train_with_rag_context.py

# Query (with LoRA adapter on vikhr-book-merged)
python src/rag/query_rag.py --lora-path models/labkovsky-rag-context-lora-v11
python src/rag/query_rag.py --no-lora  # baseline

# Query (fully merged model - no adapter needed)
python src/rag/query_rag.py --base-model models/vikhr-labkovsky-final --no-lora
```

## Development Environment

### Why WSL is Required for Training

Unsloth requires **Triton** for optimized GPU kernels. Triton only supports Linux.

**Windows limitations:**
- Triton cannot be installed natively on Windows
- Windows torch stuck at 2.6.0+cu124 — Unsloth requires newer torch
- Result: `ModuleNotFoundError: No module named 'triton'`

**Solution:** WSL Ubuntu supports torch 2.10.0 + triton 3.6.0 — works with Unsloth.

**Two venvs in project:**
| Venv | Platform | Unsloth | Use Case |
|------|----------|---------|----------|
| `venv/` | Windows | ❌ No triton | Inference only (with transformers) |
| `venv_wsl/` | WSL Ubuntu | ✅ Works | Training with Unsloth |

```bash
# Training (WSL required)
source venv_wsl/bin/activate
python src/fine_tuning/train_with_rag_context.py

# Inference (works on both)
python src/rag/query_rag.py
```

## Production Deployment (Modal)

The merged model `iisrail/vikhr-labkovsky-final` is hosted on Modal.com. The
deployment lives in `modal_deploy_v2.py` (single file, single app).

### Topology
```
Telegram user
   ↓
@LabkovskyAI_bot
   ↓
run_telegram_bot  (Modal CPU container, aiogram polling)
   ↓ HTTPS POST
ask_labkovsky    (Modal HTTPS endpoint)
   ↓ modal.method
LabkovskyInference  (Modal A10G GPU, fp16, scaledown_window=600s)
```
Nothing runs on the developer's laptop. Both the inference container and the
Telegram polling loop are Modal-hosted.

### Inference flow (`LabkovskyInference.ask`)
Mirrors `src/fine_tuning/test_rag_cards_common.py` exactly:
1. Embed question with `multilingual-e5-large` (cuda).
2. **Violence vector** — cosine against 7 abuse anchors, threshold `0.84`. If
   triggered: force DT to `DEPENDENCY_BOUNDARIES`, restrict nearest-QA search
   to boundary QAs, and inject `VIOLENCE_RESPONSE_INSTRUCTION`.
3. **DT resolution** — nearest QA → its `decision_type` (unless violence).
4. **Card selection** — top 3 reasoning cards filtered by DT, scored by
   `cosine + lexical_overlap_boost` (cap `0.06`).
5. **QA example** — top 1 QA in same DT.
6. **Topic gate** — if `top_card_sim < TOPIC_FLOOR (0.80)` and not violence,
   bypass LLM and return `OUT_OF_SCOPE_ANSWER = "Это не ко мне. Я только
   психолог."`. Prevents off-topic generic answers ("инфляция", "2+2").
7. **Generate** — fp16 model on A10G, `temperature=0.3`, `max_tokens=512`.
8. **Clean tags** — strip both `[BRACKETED]` and `{BRACED}` markers from output:
   `re.sub(r"(\[[A-Za-z_]{2,32}\]|\{[A-Za-z_]{2,32}\})\s*", "", answer)`.

### API response (full RAG trace)
```json
{
  "question": "...",
  "answer": "...",
  "decision_type": "DEPENDENCY_BOUNDARIES",
  "dt_source": "nearest_qa | violence_vector | topic_gate | fallback",
  "nearest_qa_id": "...",
  "nearest_qa_similarity": 0.83,
  "violence_score": 0.21,
  "is_violence": false,
  "cards": [{"id": "card_46", "dt": [...], "sim": 0.82, "score": 0.83,
             "lex_boost": 0.012, "match_type": "dt_exact",
             "principle_preview": "..."}],
  "qa_example": {"id": "...", "dt": "...", "sim": 0.83,
                 "question_preview": "..."},
  "topic_gate": {"passed": true, "top_sim": 0.82, "threshold": 0.80}
}
```

### Modal functions in `modal_deploy_v2.py`
| Function | Type | Purpose |
|----------|------|---------|
| `LabkovskyInference` | `@app.cls(gpu=A10G)` | Model + embeddings + cards + QA index |
| `ask_labkovsky` | `@modal.fastapi_endpoint(POST)` | Public HTTPS, calls `LabkovskyInference.ask` |
| `health` | `@modal.fastapi_endpoint(GET)` | Liveness check |
| `run_telegram_bot` | `@app.function` (CPU) | Long-running aiogram polling, 24h timeout |
| `test_query` | `@app.function` | Manual smoke test |

`bot_image = image.pip_install("aiogram", "aiohttp")` — reuses the main image
(numpy/torch are imported at module top-level, so the bot container needs them
too).

### Telegram bot commands
- Plain text → answer
- `/start`, `/help` → usage
- `/debug <question>` → answer **plus** full RAG trace (DT, nearest QA,
  3 cards with sim/score/lex_boost/preview, QA example, topic_gate)

### Deployment commands
```bash
# One-time secret setup
modal secret create telegram-bot-token TELEGRAM_BOT_TOKEN=<token>

# Deploy/redeploy the app (API + function definitions)
modal deploy modal_deploy_v2.py

# Start the long-running bot poller in detached mode
modal run --detach modal_deploy_v2.py::run_telegram_bot

# Inspect
modal app list
modal app logs labkovsky-bot-v2

# Stop bot (uses ephemeral app id from `modal app list`)
modal app stop ap-<id>
```

### Why two app entries appear
- `deployed` app (`labkovsky-bot-v2`) — owns the public URLs, hosts function
  definitions, contains the warm inference container.
- `ephemeral (detached)` app — the actual long-running `run_telegram_bot`
  invocation. Same app name, separate dashboard entry. The live bot container
  shows up here, not under the deployed app.

### Model config decisions
- **fp16, no quantization** (was 4-bit nf4 BitsAndBytes — answer quality
  dropped vs local). A10G has 24 GB VRAM; 8B fp16 fits comfortably.
- **Embedding model on CUDA** (was CPU — caused 9-minute container startup
  for embedding the QA corpus).
- **`@modal.fastapi_endpoint`** (not `@modal.asgi_app` with a FastAPI wrapper)
  — avoids the 303 redirect from `redirect_slashes`.
- **`scaledown_window=600s`** — keeps inference container warm 10 min after
  last request; cold start otherwise ~30–60s.

### Testing
- `test_endpoint.py` — 3-case battery (relationship, violence, self-esteem).
  Hits `/health` then POST `/ask_labkovsky` with UTF-8 handling. PowerShell
  mangles Cyrillic, so always run via Python or WSL bash.
- In Telegram: use `/debug` to confirm RAG cards/QA are actually retrieved
  (not just the LLM answering from base knowledge).

### Known limits / next steps
- Topic gate blocks all off-topic questions with a single canned response — no
  graceful redirection. Acceptable for now.
- `anti_generic_finance.jsonl` only has 7 examples; LoRA could be retrained on
  a larger boundary corpus instead of relying solely on the runtime gate.
- Bot polls with `max_containers=1` (Telegram allows only one `getUpdates`
  consumer per token). Restarting requires `modal app stop` on the old
  ephemeral run before launching a new one, or `TelegramConflictError` ensues.

## Code Conventions
- pathlib.Path for all paths
- Russian in user-facing strings, English in code
- Type hints preferred
