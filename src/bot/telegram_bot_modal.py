#!/usr/bin/env python3
"""
Telegram bot for Labkovsky AI psychologist.

Connects to Modal API for inference (no local GPU required).

Usage:
    set TELEGRAM_BOT_TOKEN=your_token_here
    python src/bot/telegram_bot_modal.py
"""

import asyncio
import logging
import os
import aiohttp

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart

# =============================================================================
# CONFIG
# =============================================================================

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    print("ERROR: Set TELEGRAM_BOT_TOKEN environment variable")
    print("  Windows: set TELEGRAM_BOT_TOKEN=your_token")
    print("  Linux:   export TELEGRAM_BOT_TOKEN=your_token")
    exit(1)

# Modal API endpoint
MODAL_API_URL = "https://iisrail--labkovsky-bot-ask-labkovsky.modal.run"

# Generation parameters
TEMPERATURE = 0.5
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


async def call_modal_api(question: str) -> str:
    """Call Modal API and return answer."""
    payload = {
        "question": question,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            MODAL_API_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300)  # 5 min for cold start
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("answer", "Извините, не удалось получить ответ.")
            else:
                log.error(f"API error: {response.status}")
                return "Извините, сервис временно недоступен."


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
        # Call Modal API
        answer = await call_modal_api(question)

        # Truncate if too long for Telegram (4096 chars max)
        if len(answer) > 4000:
            answer = answer[:4000] + "..."

        await message.answer(answer)
        log.info(f"Answered {message.from_user.id} ({len(answer)} chars)")

    except asyncio.TimeoutError:
        log.error("API timeout")
        await message.answer(
            "Извините, сервис загружается. Попробуйте ещё раз через минуту."
        )
    except Exception as e:
        log.error(f"Error: {e}")
        await message.answer("Извините, произошла ошибка при обработке вопроса.")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Start bot."""
    log.info("=" * 50)
    log.info("Labkovsky Telegram Bot (Modal API)")
    log.info("=" * 50)
    log.info(f"API: {MODAL_API_URL}")
    log.info("Press Ctrl+C to stop")

    # Start polling
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Bot stopped")
