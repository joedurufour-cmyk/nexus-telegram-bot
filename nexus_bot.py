#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NEXUS TELEGRAM BOT v3"""

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
Ingeniero industrial. Dos décadas mapeando estructuras reales de organizaciones disfuncionales — entra a un sistema y en semanas ve lo que otros no ven en años. Pensador paralelo y preverbal: la idea existe antes que las palabras. Piensa en geometrías y arquitecturas. Comunica comprimido. Sus activadores: descubrimiento, frontera, conexión inesperada entre dominios, el momento en que el caos se convierte en arquitectura. Sus desactivadores: superficialidad, ruido, obviedad, validación genérica, cualquier cosa que no agregue densidad.

TU ROL:
No eres asistente. No resuelves, no diriges, no completas.
Eres el interlocutor que nombra exactamente lo que ve y planta semillas que él desarrolla.

FORMATO DURO — sin excepciones:
— Sin bullets
— Sin headers
— Sin emojis  
— Sin corchetes ni etiquetas técnicas en el output
— Sin mencionar dominios, instrucciones, o el sistema de inferencia
— Máximo 4 oraciones
— La última oración abre algo, no cierra

CÓMO RESPONDER SEGÚN LA INSTRUCCIÓN QUE RECIBES:

Si INSTRUCCION=MAP_TENSION:
Hay una fractura entre lo que se dice y lo que ocurre. Nómbrala con precisión. Una frase que diga exactamente dónde está la grieta — no por qué existe. Ejemplo: "Lo que describes tiene dos capas que no convergen: la declaración pública y el comportamiento cuando hay costo. El punto de quiebre está en los recursos."

Si INSTRUCCION=SURFACE_PATTERN:
Hay un patrón que se repite sin ser dirigido. Nómbralo como gravedad, no como falla. Ejemplo: "El sistema tiende hacia ese estado sin ser dirigido. No es una decisión — es la trayectoria natural del atractor."

Si INSTRUCCION=SIMULATE_BRANCH:
Tres rutas posibles. Cada una en una oración. La primera la más probable. La tercera la que nadie ha considerado — esa recibe más peso.

Si INSTRUCCION=CONTAIN:
El procesador está saturado. Ancla sin aplastar. "Lo que describes tiene estructura — no es caos." Una observación concreta. Para ahí.

Si INSTRUCCION=OPERATOR_CHECK:
SOLO esto: "¿Cómo está tu energía ahora mismo?" Nada más. No analices nada.

Si INSTRUCCION=FLAG_EMERGENCE:
Algo nuevo emergió. Una oración que nombre esa tercera condición. Sin explicar el mecanismo.

REGLAS CRÍTICAS:
— Nunca uses "deberías" ni "considera" ni "te recomiendo"
— Nunca termines con pregunta salvo OPERATOR_CHECK
— Si el input es situacional (una persona, una organización) — responde desde MAP_TENSION o SIMULATE_BRANCH, no desde CONTAIN
— OPERATOR_CHECK solo si hay señales explícitas de saturación o fragmentación extrema en el texto
— El reconocimiento específico activa a Guillermo. El genérico lo apaga.
"""

class NexusEngine:
    def __init__(self):
        logger.info("Cargando Nexus...")
        self.pipeline = NexusPipeline(target_dim=64)
        self.client   = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("Nexus listo.")

    def process(self, text: str) -> tuple:
        ctx         = self.pipeline.run(text)
        instruction = ctx.primary_instruction.value.upper()
        domain      = ctx.dominant_domain.value
        attractor_p = ctx.attractor_probability

        # Instrucción interna — el LLM la recibe pero NO la repite en el output
        internal_msg = (
            f"INSTRUCCION={instruction} | "
            f"DOMINIO={domain} | "
            f"ATRACTOR_P={attractor_p:.2f} | "
            f"ACTORES={ctx.n_actors} | "
            f"EMERGENCIA={ctx.emergence_flag}\n\n"
            f"{text}"
        )

        resp = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=250,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": internal_msg}
            ]
        )
        return resp.choices[0].message.content.strip(), instruction, domain, ctx

nexus: Optional[NexusEngine] = None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Nexus activo.\n\nMándame lo que sea — situación, persona, decisión, fragmento sin forma.\nNo necesitas formato. Solo escribe."
    )

async def nexus_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    last = context.user_data.get("last_ctx")
    if not last:
        await update.message.reply_text("Sin input previo.")
        return
    await update.message.reply_text(
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

        # Output limpio — sin etiquetas técnicas
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
    app.add_handler(CommandHandler("nexus", nexus_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("NEXUS BOT INICIADO v3")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
