#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
NEXUS TELEGRAM BOT v3.1 — BACKEND INTEGRADO
Conecta el pipeline matemático (Vectorizador → Capa1 → Capa2) con el bot v3.

CAMBIOS vs v3.0:
  - Reemplaza NexusEngine por NexusEngineIntegrated (pipeline matemático real)
  - El tensor analiza el input antes de llamar a OpenAI
  - El contexto dinámico del tensor se inyecta en el system prompt

DEPLOY:
  Railway / Render / cualquier servidor con Python 3.9+

VARIABLES DE ENTORNO:
  TELEGRAM_BOT_TOKEN   → token de @BotFather
  OPENAI_API_KEY       → key de OpenAI
  OPENAI_MODEL         → (opcional) default: gpt-4o-mini
  ALLOWED_USER_IDS     → (opcional) IDs separados por coma para restringir acceso

AUTOR: Nexus v3.1 build — Junio 2026
================================================================================
"""

import os
import logging
import sys
from typing import Optional

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
)

# ── Insertar directorio actual en el path para los módulos Nexus ─────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Importar el backend integrado (drop-in replacement) ────────────────────────
from nexus_backend_integrated import NexusEngineIntegrated as NexusEngine

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "TU_TOKEN_AQUI")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "TU_KEY_AQUI")
OPENAI_MODEL       = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_allowed_raw = os.getenv("ALLOWED_USER_IDS", "")
ALLOWED_USER_IDS = set(int(x) for x in _allowed_raw.split(",") if x.strip().isdigit())

# =============================================================================
# LOGGER
# =============================================================================

logging.basicConfig(
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# =============================================================================
# MEMORIA POR USUARIO (compartida con el engine)
# =============================================================================

class NexusMemory:
    """Memoria de usuario: historial + conversación activa."""

    def __init__(self):
        self.history: dict = {}
        self.conversation: dict = {}
        self.last_energy: dict = {}
        self.mode_counts: dict = {}

    def get_conversation(self, user_id: int) -> list:
        return self.conversation.get(user_id, [])

    def add_turn(self, user_id: int, role: str, content: str):
        if user_id not in self.conversation:
            self.conversation[user_id] = []
        self.conversation[user_id].append({"role": role, "content": content})
        self.conversation[user_id] = self.conversation[user_id][-20:]

    def reset_conversation(self, user_id: int):
        self.conversation[user_id] = []

    def log_interaction(self, user_id: int, text: str, mode: str, energy: float):
        from datetime import datetime
        if user_id not in self.history:
            self.history[user_id] = []
        self.history[user_id].append({
            "timestamp": datetime.now().isoformat(),
            "input": text[:150],
            "mode": mode,
            "energy": energy,
        })
        self.history[user_id] = self.history[user_id][-30:]
        self.last_energy[user_id] = energy

        if user_id not in self.mode_counts:
            self.mode_counts[user_id] = {}
        self.mode_counts[user_id][mode] = self.mode_counts[user_id].get(mode, 0) + 1

    def build_context(self, user_id: int) -> str:
        if user_id not in self.history or not self.history[user_id]:
            return "Primera interacción con este usuario."
        recent = self.history[user_id][-5:]
        avg_energy = sum(r["energy"] for r in recent) / len(recent)
        total = len(self.history[user_id])
        top_mode = ""
        if user_id in self.mode_counts:
            top_mode = max(self.mode_counts[user_id], key=self.mode_counts[user_id].get)
        return (
            f"Total interacciones: {total}. "
            f"Energía promedio reciente: {avg_energy:.1f}/10. "
            f"Modo más frecuente: {top_mode}. "
            f"Última energía: {self.last_energy.get(user_id, 5)}/10."
        )

    def get_stats(self, user_id: int) -> str:
        if user_id not in self.history or not self.history[user_id]:
            return "Sin historial aún."
        h = self.history[user_id]
        avg_e = sum(x["energy"] for x in h) / len(h)
        modes = self.mode_counts.get(user_id, {})
        sorted_modes = sorted(modes.items(), key=lambda x: x[1], reverse=True)
        modes_text = " | ".join(f"{m}: {c}" for m, c in sorted_modes)
        return (
            f"📊 *Tus Estadísticas Nexus*\n\n"
            f"📝 Interacciones: {len(h)}\n"
            f"⚡ Energía promedio: {avg_e:.1f}/10\n"
            f"🎯 Modos: {modes_text}\n"
            f"🕐 Primera sesión: {h[0]['timestamp'][:10]}"
        )


# =============================================================================
# INSTANCIAS GLOBALES
# =============================================================================

memory = NexusMemory()
nexus = NexusEngine(memory=memory)

# =============================================================================
# GUARD
# =============================================================================

def is_allowed(user_id: int) -> bool:
    if not ALLOWED_USER_IDS:
        return True
    return user_id in ALLOWED_USER_IDS

# =============================================================================
# BOTONES DE FEEDBACK
# =============================================================================

FEEDBACK_KEYBOARD = InlineKeyboardMarkup([
    [
        InlineKeyboardButton("✅ Sí, orientó", callback_data="fb_yes"),
        InlineKeyboardButton("🔄 Más profundo", callback_data="fb_more"),
        InlineKeyboardButton("❌ No ayudó", callback_data="fb_no"),
    ]
])

# =============================================================================
# HANDLERS — COMANDOS
# =============================================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not is_allowed(user.id):
        return
    memory.reset_conversation(user.id)
    text = (
        f"🧠 *NEXUS ACTIVADO*\n\n"
        f"Hola {user.first_name}. No soy un asistente que obedece.\n"
        f"Soy un motor de inferencia. El tensor analiza, el LLM interpreta.\n\n"
        f"Cuéntame lo que sea.\n\n"
        f"_/help para comandos · /stats para tu historial · /reset para nueva sesión_"
    )
    await update.message.reply_text(text, parse_mode="Markdown")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "📋 *Comandos Nexus*\n\n"
        "/start — Activar Nexus\n"
        "/reset — Limpiar conversación\n"
        "/stats — Ver tu historial\n"
        "/calm /analyze /decide /simulate /offload — Forzar modo\n\n"
        "Escribe directamente y Nexus detecta el modo automáticamente."
    )
    await update.message.reply_text(text, parse_mode="Markdown")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        return
    await update.message.reply_text(memory.get_stats(user_id), parse_mode="Markdown")

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        return
    memory.reset_conversation(user_id)
    await update.message.reply_text("🔄 Conversación reiniciada. Cuéntame.")

async def force_mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        return
    cmd = update.message.text.split()[0].replace("/", "").lower()
    mode_map = {"calm": "Calm", "analyze": "Analyze", "decide": "Decide", "simulate": "Simulate", "offload": "Offload"}
    mode = mode_map.get(cmd, "Calm")
    context.user_data["forced_mode"] = mode
    await update.message.reply_text(f"🎯 Modo forzado: *{mode}*\n\nEscribe tu mensaje.", parse_mode="Markdown")

# =============================================================================
# HANDLER — MENSAJE PRINCIPAL
# =============================================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not is_allowed(user.id):
        return

    text = update.message.text
    user_id = user.id

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    forced_mode = context.user_data.pop("forced_mode", None)

    try:
        reply = await nexus.process(user_id, text, forced_mode=forced_mode)
        await update.message.reply_text(reply, parse_mode="Markdown")
        await update.message.reply_text("¿Esto te orientó?", reply_markup=FEEDBACK_KEYBOARD)
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text("🔴 Error en el pipeline. Intenta de nuevo.")

# =============================================================================
# HANDLER — CALLBACKS DE FEEDBACK
# =============================================================================

async def handle_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "fb_yes":
        await query.edit_message_text("✅ Bien. Sigue cuando quieras.")
    elif data == "fb_more":
        await query.edit_message_text("🔄 ¿Qué parte quieres que profundice? Escríbelo.")
    elif data == "fb_no":
        await query.edit_message_text("❌ Entendido. Reformúlalo y lo intento de nuevo.")

# =============================================================================
# MAIN
# =============================================================================

def main():
    if TELEGRAM_BOT_TOKEN == "TU_TOKEN_AQUI":
        logger.error("❌ TELEGRAM_BOT_TOKEN no configurado. Saliendo.")
        return

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("calm", force_mode_command))
    app.add_handler(CommandHandler("analyze", force_mode_command))
    app.add_handler(CommandHandler("decide", force_mode_command))
    app.add_handler(CommandHandler("simulate", force_mode_command))
    app.add_handler(CommandHandler("offload", force_mode_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(handle_feedback, pattern="^fb_"))

    logger.info("🧠 NEXUS BOT v3.1 INICIADO (Backend Integrado)")
    app.run_polling()

if __name__ == "__main__":
    main()
