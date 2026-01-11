# Labkovsky AI Model

## Project Overview
Fine-tuned LLM that replicates Mikhail Labkovsky's psychological consultation style using RAG + LoRA fine-tuning.

## Tech Stack
- Python 3.10+
- PyTorch + CUDA
- Transformers, PEFT, TRL (use `max_length` not `max_seq_length` in SFTConfig)
- ChromaDB for RAG
- Sentence-transformers (multilingual-e5-large)
- Base model: Qwen/Qwen2.5-7B-Instruct

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
- No system prompt in training/inference
- RS (Response Signal) prefix as behavioral control: <RS=INTERVENTION>, <RS=EXPLANATION>, <RS=ESCALATION>
- RS tokens registered as special tokens (not subword-split)
- Completion-only loss (DataCollatorForCompletionOnlyLM with "Ответ:" delimiter)
- RAG is supportive at inference only, not in training

## Training Data Format
```
<RS=INTERVENTION>
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
```

## Code Conventions
- Use pathlib.Path for all paths
- Russian language in user-facing strings, English in code/comments
- Type hints preferred
