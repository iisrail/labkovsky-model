#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the RAG LoRA trained on top of the merged book model.

Usage:
    python src/fine_tuning/test_rag_cards_merged_model.py --id srJvn19GKNA_02
    python src/fine_tuning/test_rag_cards_merged_model.py --interactive
"""

from test_rag_cards_common import main


if __name__ == "__main__":
    main(
        base_model="./models/vikhr-book-merged",
        lora_dir_name="labkovsky-rag-context-lora-v12-v4-no-mmr",
        label="merged book-base attention-only RAG LoRA",
    )
