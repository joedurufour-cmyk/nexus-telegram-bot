#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS TELEGRAM BOT v2 — calibrado para Guillermo
"""

import os, sys, logging
from typing import Optional
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nexus_vectorizer import NexusVectorizer
from nexus_layer1_tensor import NexusLayer1
from nexus_layer2 import NexusPipeline, LLMContext

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

SYSTEM_PROMPT = """Eres Nexus — interlocutor de alta densidad para Guillermo.

QUIÉN ES GUILLERMO:
Ingeniero industrial. Dos décadas entrando a sistemas organizacionales disfuncionales y mapeando su estructura real en semanas. Pensador paralelo y preverbal — la idea existe antes que las palabras. Piensa en geometrías y arquitecturas, no en narrativas lineales. Comunica comprimido y espera que el interlocutor expanda. Sus activadores: descubrimiento, frontera, conexión inesperada entre dominios, creación de sistema. Sus desactivadores: superficialidad, ruido, obviedad, validación genérica.

TU ROL:
No eres asistente. No resuelves, no diriges, no completas.
Eres el interlocutor que sigue su velocidad, nombra lo que ve, y planta semillas que él desarrolla.

CÓMO RESPONDER:
— Primera línea: lo más específico y denso del input. Sin preámbulo.
— Cuerpo: prosa directa. Sin bullets. Sin headers. Sin emojis. Sin nombres de dominio técnico.
— Última línea: una semilla — algo que abre, no que cierra.
— Longitud: nunca más de lo necesario. 3-6 oraciones es suficiente casi siempre.

INSTRUCCIONES POR MODO (el sistema ya calculó cuál aplica — ejecútala):

MAP_TENSION: hay un delta entre lo que se dice y lo que pasa. Nómbralo con precisión quirúrgica. Una frase que diga exactamente dónde está la fractura. No expliques por qué existe.

SURFACE: hay un patrón que se repite sin ser dirigido. Dilo así: "El sistema tiende hacia X. No es una decisión — es gravedad." Concreta. Sin análisis adicional.

SIMULATE_BRANCH: genera 3 rutas posibles desde donde está. Cada una en una oración. La primera la más probable. La tercera la que nadie ha considerado — esa es la más importante.

CONTAIN: el procesador está saturado o en estado preverbal. Ancla sin aplastar. "Lo que describes tiene estructura — no es caos." Una observación concreta. Para.

OPERATOR_CHECK: solo pregunta "¿Cómo está tu energía ahora mismo?" Nada más.

FLAG_EMERGENCE: algo nuevo emergió de la colisión. Una oración que nombre esa tercera condición. Sin explicar cómo llegaste.

REGLAS DURAS:
— Nunca uses "deberías"
— Nunca digas el nombre del dominio o la instrucción en la respuesta
— Nunca más de 6 oraciones
— Nunca termines con pregunta salvo OPERATOR_CHECK
— Si algo es obvio para Guillermo — no lo digas
— El reconocimiento específico activa. El genérico aplana.
"""

class NexusEngine:
    def __init__(self):
        logger.info("Cargando Nexus...")
        self.pipeline = NexusPipeline(target_dim=64)
        self.client   = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("Nexus listo.")

    def process(self, text: str) -> tuple:
        ctx = self.pipeline.run(text)
        instruction = ctx.primary_instruction.value.upper()
        domain      = ctx.dominant_domain.value
        attractor_p = ctx.attractor_probability

        user_msg = (
            f"[INSTRUCCIÓN DEL TENSOR: {instruction} | "
            f"dominio={domain} | P={attractor_p:.2f} | "
            f"actores={ctx.n_actors} | confianza={ctx.layer1_confidence:.2f}]\n\n"
            f"{text}"
        )

        resp = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=300,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg}
            ]
        )
        return resp.choices[0].message.content.strip(), instruction, domain, ctx

nexus: Optional[NexusEngine] = None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Nexus activo.\n\nMándame lo que sea — situación, fragmento, persona, decisión, algo que no tiene forma todavía.\nNo necesitas formato. Solo escribe."
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Nexus detecta la estructura por debajo de lo que escribes y responde desde ahí.\n\n/nexus — ver qué detectó el tensor en tu último mensaje."
    )

async def nexus_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    last = context.user_data.get("last_ctx")
    if not last:
        await update.message.reply_text("Sin input previo.")
        return
    await update.message.reply_text(
        f"TENSOR:\n"
        f"instrucción → {last['instruction']}\n"
        f"dominio → {last['domain']}\n"
        f"atractor → {last['attractor']} (P={last['attractor_p']:.3f})\n"
        f"interferencia → {last['interference']}\n"
        f"actores → {last['n_actors']}\n"
        f"emergencia → {last['emergence']}\n"
        f"confianza → {last['confidence']:.3f}"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global nexus
    if nexus is None:
        await update.message.reply_text("Iniciando...")
        return
    text = update.message.text
    if not text or len(text.strip()) < 3:
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        response, instruction, domain, ctx = nexus.process(text)

        context.user_data["last_ctx"] = {
            "instruction":  instruction,
            "domain":       domain,
            "attractor":    ctx.attractor_node,
            "attractor_p":  ctx.attractor_probability,
            "interference": ctx.interference_intensity.value,
            "n_actors":     ctx.n_actors,
            "emergence":    ctx.emergence_flag,
            "confidence":   ctx.layer1_confidence
        }

        await update.message.reply_text(response)

    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text("Error en el pipeline.")

def main():
    global nexus
    if not TELEGRAM_TOKEN: raise ValueError("TELEGRAM_BOT_TOKEN no configurado")
    if not OPENAI_API_KEY:  raise ValueError("OPENAI_API_KEY no configurado")
    nexus = NexusEngine()
    app   = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help",  help_cmd))
    app.add_handler(CommandHandler("nexus", nexus_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("NEXUS BOT INICIADO v2")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
