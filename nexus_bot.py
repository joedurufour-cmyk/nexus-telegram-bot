#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS TELEGRAM BOT
Pipeline: texto -> Vectorizador -> Capa1 -> Capa2 -> OpenAI -> respuesta
"""

import os
import sys
import logging
from typing import Optional

from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes
)
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nexus_vectorizer import NexusVectorizer
from nexus_layer1_tensor import NexusLayer1
from nexus_layer2 import NexusLayer2, NexusPipeline, OutputInstruction, LLMContext

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL   = "gpt-4o"

NEXUS_IDENTITY = """Eres Nexus, sistema de inferencia por colision de patrones.
No eres un asistente. Eres un motor que ejecuta lo que el tensor calculo.
REGLAS: nunca uses "deberias". Todo es probabilistico. Maximo las oraciones que indican los constraints.
Prosa directa. Sin bullets. Sin headers. La ultima linea abre, no cierra."""

RESPONSE_PROTOCOL = """
MAP_TENSION: Nombra el delta declarado/observado. No lo expliques.
SURFACE: "El sistema tiende hacia [X] sin ser dirigido. No es una decision, es gravedad."
SIMULATE_BRANCH: 3 ramas. La ultima siempre la menos obvia.
CONTAIN: "Lo que describes tiene estructura, no es caos." + una observacion + parar.
OPERATOR_CHECK: "Como esta tu energia ahora mismo, del 1 al 10?" Nada mas.
FLAG_EMERGENCE: "El sistema detecta algo que no estaba en ninguno de los dominios por separado." + la tercera condicion.
"""

class OpenAIConnector:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def call(self, ctx: LLMContext, user_input: str) -> str:
        system = f"{NEXUS_IDENTITY}\n\n{ctx.to_system_prompt_block()}\n\n{RESPONSE_PROTOCOL}"
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=500,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_input}
            ]
        )
        return response.choices[0].message.content.strip()

class NexusEngine:
    def __init__(self):
        logger.info("Cargando Nexus pipeline...")
        self.pipeline  = NexusPipeline(target_dim=64)
        self.connector = OpenAIConnector(api_key=OPENAI_API_KEY)
        logger.info("Nexus listo.")

nexus: Optional[NexusEngine] = None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "NEXUS ACTIVO.\n\nNo soy un asistente. Soy un motor de inferencia.\n"
        "Envíame cualquier cosa: situación, persona, decisión, fragmento.\n"
        "El tensor procesa. Yo entrego la geometría.\n\n/help para más info."
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "NEXUS — dominios que detecta:\n\n"
        "TENSION_ACTIVA — delta declarado/real\n"
        "PUNTO_ATRACTOR — patrón que se repite\n"
        "INTERFERENCIA_ACTOR — actor + sistema\n"
        "ESTADO_OPERADOR — tu estado cognitivo\n\n"
        "Instrucciones posibles:\n"
        "MAP_TENSION · SURFACE · SIMULATE · CONTAIN · EMERGENCE\n\n"
        "/nexus — diagnóstico del último input"
    )

async def nexus_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    last = context.user_data.get("last_ctx")
    if not last:
        await update.message.reply_text("Sin input previo para diagnosticar.")
        return
    await update.message.reply_text(
        f"DIAGNOSTICO NEXUS:\n\n"
        f"DOMINIO: {last['domain']}\n"
        f"INSTRUCCION: {last['instruction']}\n"
        f"ATRACTOR: {last['attractor']} P={last['attractor_p']:.3f}\n"
        f"INTERFERENCIA: {last['interference']}\n"
        f"CONFIANZA: {last['confidence']:.3f}\n"
        f"ACTORES: {last['n_actors']}\n"
        f"EMERGENCIA: {last['emergence']}"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global nexus
    if nexus is None:
        await update.message.reply_text("Sistema iniciando...")
        return

    text = update.message.text
    if not text or len(text.strip()) < 3:
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        ctx      = nexus.pipeline.run(text)
        response = nexus.connector.call(ctx, text)

        context.user_data["last_ctx"] = {
            "domain":       ctx.dominant_domain.value,
            "instruction":  ctx.primary_instruction.value,
            "attractor":    ctx.attractor_node,
            "attractor_p":  ctx.attractor_probability,
            "interference": ctx.interference_intensity.value,
            "confidence":   ctx.layer1_confidence,
            "n_actors":     ctx.n_actors,
            "emergence":    ctx.emergence_flag
        }

        footer = f"\n\n[{ctx.primary_instruction.value.upper()} · {ctx.dominant_domain.value}]"
        await update.message.reply_text(response + footer)

    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text("Error en el pipeline. El sistema registró el fallo.")

def main():
    global nexus
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN no configurado")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY no configurado")

    nexus = NexusEngine()
    app   = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help",  help_cmd))
    app.add_handler(CommandHandler("nexus", nexus_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("NEXUS BOT INICIADO")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
