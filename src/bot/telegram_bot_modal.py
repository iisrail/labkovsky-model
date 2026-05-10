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
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart, Command
from aiogram.enums import ParseMode
from aiogram.utils.markdown import hbold, hcode

# =============================================================================
# CONFIG
# =============================================================================

# Load .env from repo root (two levels up from src/bot/)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    print("ERROR: Set TELEGRAM_BOT_TOKEN environment variable")
    print("  Windows: set TELEGRAM_BOT_TOKEN=your_token")
    print("  Linux:   export TELEGRAM_BOT_TOKEN=your_token")
    exit(1)

# Modal API endpoint (labkovsky-bot-v2: fp16, violence detection, RAG cards)
MODAL_API_URL = "https://iisrail--labkovsky-bot-v2-ask-labkovsky.modal.run"

# Generation parameters (match server defaults)
TEMPERATURE = 0.3
MAX_TOKENS = 512

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


async def call_modal_api(question: str) -> dict:
    """Call Modal API and return parsed response dict."""
    params = {
        "question": question,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            MODAL_API_URL,
            params=params,  # v2 endpoint reads simple params from query string
            timeout=aiohttp.ClientTimeout(total=300)  # 5 min for cold start
        ) as response:
            if response.status == 200:
                return await response.json()
            log.error(f"API error: {response.status} {await response.text()}")
            return {"answer": "Извините, сервис временно недоступен."}


def _format_cards_log(cards: list) -> str:
    """Compact one-line summary of retrieved cards for logs."""
    if not cards:
        return "[]"
    return "[" + ", ".join(f"{c['id']}@{c['sim']:.3f}" for c in cards) + "]"


def _format_debug_report(result: dict) -> str:
    """Format full retrieval trace as a Telegram HTML message."""
    lines = []
    lines.append(hbold("RAG Debug Trace"))
    lines.append(f"{hbold('Question:')} {result.get('question', '')}")
    lines.append("")
    lines.append(f"{hbold('decision_type:')} {hcode(result.get('decision_type'))}")
    lines.append(f"{hbold('dt_source:')} {hcode(result.get('dt_source'))}")
    nearest_id = result.get("nearest_qa_id")
    nearest_sim = result.get("nearest_qa_similarity")
    if nearest_id is not None:
        sim_str = f"{nearest_sim:.4f}" if isinstance(nearest_sim, (int, float)) else "n/a"
        lines.append(f"{hbold('nearest_qa:')} {hcode(nearest_id)} sim={sim_str}")
    vscore = result.get("violence_score")
    if isinstance(vscore, (int, float)):
        flag = "TRUE" if result.get("is_violence") else "false"
        lines.append(f"{hbold('violence:')} {flag} (score={vscore:.4f})")

    lines.append("")
    cards = result.get("cards") or []
    lines.append(f"{hbold(f'Cards ({len(cards)}):')}")
    for i, c in enumerate(cards, 1):
        dt_str = ",".join(c.get("dt", [])) if c.get("dt") else "-"
        lines.append(
            f"  {i}. {hcode(c['id'])} dt={dt_str} sim={c['sim']:.4f} "
            f"score={c['score']:.4f} lex={c['lex_boost']:.4f} ({c['match_type']})"
        )
        preview = c.get("principle_preview", "")
        if preview:
            lines.append(f"     <i>{preview}</i>")

    qa = result.get("qa_example")
    lines.append("")
    lines.append(hbold("QA example:"))
    if qa:
        lines.append(f"  {hcode(qa['id'])} dt={qa['dt']} sim={qa['sim']:.4f}")
        if qa.get("question_preview"):
            lines.append(f"  <i>{qa['question_preview']}</i>")
    else:
        lines.append("  (none)")

    lines.append("")
    answer = result.get("answer", "")
    if len(answer) > 1500:
        answer = answer[:1500] + "..."
    lines.append(hbold("Answer:"))
    lines.append(answer)

    return "\n".join(lines)


@dp.message(CommandStart())
async def handle_start(message: Message):
    """Handle /start command."""
    await message.answer(
        "Здравствуйте! Я AI-психолог в стиле Михаила Лабковского.\n\n"
        "Задайте мне вопрос о психологии, отношениях или личных проблемах, "
        "и я постараюсь помочь.\n\n"
        "Просто напишите свой вопрос.\n\n"
        "Команды:\n"
        "/help — справка\n"
        "/debug <вопрос> — показать RAG-трассу (какие карточки и QA использовались)"
    )


@dp.message(Command("help"))
async def handle_help(message: Message):
    await message.answer(
        "Команды:\n"
        "/start — приветствие\n"
        "/debug <вопрос> — вернёт ответ вместе с трассой RAG: "
        "decision_type, ближайший QA, 3 карточки с их similarity и фрагментами принципов.\n\n"
        "Пример: /debug Почему я не могу уйти от мужа?"
    )


@dp.message(Command("debug"))
async def handle_debug(message: Message):
    """Return full RAG trace alongside the answer."""
    # Extract everything after the /debug command
    text = (message.text or "").strip()
    parts = text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.answer(
            "Использование: /debug <вопрос>\nПример: /debug Почему я не могу уйти от мужа?"
        )
        return

    question = parts[1].strip()
    log.info(f"DEBUG question from {message.from_user.id}: {question[:50]}...")
    await message.bot.send_chat_action(message.chat.id, "typing")

    try:
        result = await call_modal_api(question)
        # Ensure the question echoed in the report matches what user asked,
        # even if the server returned a different field.
        result.setdefault("question", question)
        report = _format_debug_report(result)
        if len(report) > 4000:
            report = report[:3990] + "\n…"
        await message.answer(report, parse_mode=ParseMode.HTML)
        cards = result.get("cards") or []
        log.info(
            "DEBUG answered %s dt=%s cards=%s qa=%s",
            message.from_user.id,
            result.get("decision_type"),
            _format_cards_log(cards),
            (result.get("qa_example") or {}).get("id"),
        )
    except asyncio.TimeoutError:
        log.error("API timeout")
        await message.answer("Сервис загружается, попробуйте ещё раз через минуту.")
    except Exception as e:
        log.error(f"Error: {e}")
        await message.answer("Ошибка при обработке /debug.")


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
        result = await call_modal_api(question)
        answer = result.get("answer", "Извините, не удалось получить ответ.")

        # Truncate if too long for Telegram (4096 chars max)
        if len(answer) > 4000:
            answer = answer[:4000] + "..."

        await message.answer(answer)
        cards = result.get("cards") or []
        qa = result.get("qa_example") or {}
        log.info(
            "Answered %s (%d chars) dt=%s src=%s violence=%s score=%.3f cards=%s qa=%s",
            message.from_user.id,
            len(answer),
            result.get("decision_type"),
            result.get("dt_source"),
            result.get("is_violence"),
            float(result.get("violence_score", 0.0)),
            _format_cards_log(cards),
            qa.get("id"),
        )

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
