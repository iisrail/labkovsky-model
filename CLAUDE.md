# Labkovsky AI Model

## Project Overview
Fine-tuned LLM that replicates Mikhail Labkovsky's psychological consultation style using RAG + LoRA fine-tuning.

## Tech Stack
- Python 3.10+
- PyTorch + CUDA
- Transformers, PEFT, TRL (use `max_length` not `max_seq_length` in SFTConfig)
- ChromaDB for RAG
- Sentence-transformers (multilingual-e5-large)
- Base model: Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it (Russian-native, RAG-trained)

## Project Structure
```
labkovsky-model/
├── src/
│   ├── fine_tuning/      # Training & inference scripts
│   ├── rag/              # RAG pipeline
│   ├── data_processing/  # Data preparation scripts
│   └── config.py         # Shared configuration
├── data/
│   └── fine_tuning/      # JSONL training data
└── models/               # Saved LoRA adapters
```

## Key Architecture Decisions
- Style comes from LoRA fine-tuning, not from system prompt
- RS (Response Signal) derived from RAG chunks, not separate classifier
- Completion-only loss (DataCollatorForCompletionOnlyLM with "Ответ:" delimiter)

## Data Files
- **RAG index**: `data/fine_tuning/qa_corpus_rag_optimized.jsonl`
- **Fine-tuning**: `data/fine_tuning/qa_rs_corpus_short.jsonl`

## RAG Data Format
```json
{"chunk_role": "EXPLANATION", "video_id": "...", "question": "...", "answer": "...", "id": "video_01_expl"}
```
- **chunk_role field** (not "rs"): EXPLANATION | INTERVENTION | ESCALATION
- Each Q&A has up to 3 chunks with different chunk_roles
- qa_id extracted from id: "video_01_expl" → "video_01"

## RAG Index Structure (IMPORTANT)
- **Embedding**: question + answer combined (for semantic matching)
- **Document stored**: ANSWER only (what model sees for generation)
- This prevents model from seeing question repeated in context

## RAG Generation (Vikhr)
- Uses Vikhr's `documents` role with JSON format
- Single-step generation (two-step with relevant_doc_ids doesn't work with base model)

## System Prompt (IMPORTANT)
- **Minimal Russian prompt works best**: "Ответь кратко, используя только информацию из документов."
- Complex English instructions make model MORE generic (tries to be "helpful")
- Let LoRA fine-tuning control the style, prompt only grounds on documents
- Labkovsky style: focus on ONE thing, not "bunch of answers"

## Fine-tuning Data Format
```
Вопрос: {question}
Ответ: {answer}
```

## Hardware Constraints
- 12GB VRAM GPU
- Use 4-bit quantization (QLoRA)
- Batch size 1 with gradient accumulation

## Common Commands
```bash
# Train
python src/fine_tuning/train_lora.py

# Inference
python src/inference/inference_lora_rs.py

# RAG query
python src/rag/query_rag.py

# Run with WSL venv (for unsloth/CUDA training)
wsl -e bash -c "cd /mnt/c/Projects/projects_py/labkovsky-model && source venv_wsl/bin/activate && python <script>"
```

## Code Conventions
- Use pathlib.Path for all paths
- Russian language in user-facing strings, English in code/comments
- Type hints preferred
