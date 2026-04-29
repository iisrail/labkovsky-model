#!/usr/bin/env python3
"""
Telegram bot for Labkovsky AI psychologist.

Uses aiogram 3.x + existing RAG pipeline + AWQ model.

Usage:
    set TELEGRAM_BOT_TOKEN=your_token_here
    python src/bot/telegram_bot.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "rag"))

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart
from aiogram.enums import ParseMode

from query_rag import init_embedding, init_retrieval, init_llm, ask_labkovsky

# =============================================================================
# CONFIG
# =============================================================================

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    print("ERROR: Set TELEGRAM_BOT_TOKEN environment variable")
    print("  Windows: set TELEGRAM_BOT_TOKEN=your_token")
    print("  Linux:   export TELEGRAM_BOT_TOKEN=your_token")
    sys.exit(1)

CHROMA_DIR = PROJECT_ROOT / "chroma_db"
AWQ_MODEL = PROJECT_ROOT / "models" / "vikhr-labkovsky-awq"

# Generation parameters
TEMPERATURE = 0.5
REPETITION_PENALTY = 1.15
TOP_P = 0.85
MAX_TOKENS = 500

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# =============================================================================
# BOT SETUP
# =============================================================================

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def handle_start(message: Message):
    """Handle /start command."""
    await message.answer(
        "Здравствуйте! Я AI-психолог в стиле Михаила Лабковского.\n\n"
        "Задайте мне вопрос о психологии, отношениях или личных проблемах, "
        "и я постараюсь помочь.\n\n"
        "Просто напишите свой вопрос."
    )


@dp.message(F.text)
async def handle_question(message: Message):
    """Handle user questions."""
    question = message.text.strip()

    if not question:
        return

    log.info(f"Question from {message.from_user.id}: {question[:50]}...")

    # Send typing indicator
    await message.bot.send_chat_action(message.chat.id, "typing")

    try:
        # Run inference in thread pool (blocking call)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: ask_labkovsky(
                question,
                temperature=TEMPERATURE,
                repetition_penalty=REPETITION_PENALTY,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS
            )
        )

        answer = result.get("answer", "Извините, произошла ошибка.")

        # Truncate if too long for Telegram (4096 chars max)
        if len(answer) > 4000:
            answer = answer[:4000] + "..."

        await message.answer(answer)
        log.info(f"Answered {message.from_user.id} ({len(answer)} chars)")

    except Exception as e:
        log.error(f"Error: {e}")
        await message.answer("Извините, произошла ошибка при обработке вопроса.")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Initialize models and start bot."""
    log.info("=" * 50)
    log.info("Labkovsky Telegram Bot")
    log.info("=" * 50)

    # Initialize RAG pipeline (blocking, do once at startup)
    log.info("Loading models...")

    init_embedding()
    init_retrieval(CHROMA_DIR)
    init_llm(
        use_lora=False,
        lora_path=None,
        base_model_path=str(AWQ_MODEL),
        skip_quantization=True
    )

    log.info("Models loaded! Starting bot...")
    log.info("Press Ctrl+C to stop")

    # Start polling
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Bot stopped")
