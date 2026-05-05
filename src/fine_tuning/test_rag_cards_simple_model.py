#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the simple reasoning-card LoRA trained on the fresh Vikhr base.

Usage:
    python src/fine_tuning/test_rag_cards_simple_model.py --id srJvn19GKNA_02
    python src/fine_tuning/test_rag_cards_simple_model.py --interactive
"""

from test_rag_cards_common import main


if __name__ == "__main__":
    main(
        base_model="Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it",
        lora_dir_name="labkovsky-reasoning-cards-v1",
        label="simple fresh-base reasoning-card LoRA",
    )
