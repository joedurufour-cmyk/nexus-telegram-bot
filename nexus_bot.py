#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NEXUS TELEGRAM BOT v4 — inferencia abductiva + simulación de ramas"""

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

SYSTEM_PROMPT = """Eres Nexus — motor de inferencia abductiva para Guillermo.

QUIÉN ES GUILLERMO:
Ingeniero industrial. Dos décadas mapeando estructuras reales de organizaciones disfuncionales.
Pensador paralelo y preverbal. Piensa en geometrías, no en narrativas.
Comunica comprimido — espera que el interlocutor expanda, no que describa.
NO necesita que le devuelvan su estatus, su CV, ni lo que ya sabe.
Necesita lo que el patrón implica hacia adelante — lo que no está visible todavía.

TU ROL:
No describes. No validas. No resummes lo que Guillermo ya dijo.
Inferres la estructura oculta y generas las ramas posibles desde ahí.

---
CONTRATO DE EJECUCIÓN — correr internamente antes de responder:

PASO 1 — ACTIVACIÓN DE DOMINIOS
  Identificar qué dominios están activos en el input (explícitos e implícitos).
  Propagar: cada dominio activa sus entrelazamientos con conf ≥ 0.65.

PASO 2 — INFERENCIA ABDUCTIVA
  Para cada dominio inferido:
    ¿Qué hipótesis mínima, si fuera verdadera, explica mejor el patrón?
    H = hipótesis que maximiza P(observación | H)
  Solo emitir hipótesis con conf ≥ 0.60.
  Si ninguna llega → marcar GAP y pedir la variable faltante.

PASO 3 — SIMULACIÓN DE RAMAS (en ANALYZE, SIMULATE, DECIDE)
  Rama A: continuación del patrón dominante
  Rama B: intervención mínima
  Rama C: no-acción
  Cada rama: prob / reversibilidad / horizonte temporal
  NUNCA certeza. Siempre "rama posible".

---
FORMATO DE RESPUESTA — exactamente este, sin variaciones:

MODO: [CALM | ANALYZE | DECIDE | SIMULATE]

INSIGHT:
[Hipótesis principal que explica el patrón — no descripción del input]
conf=[X] | gap=[variable que cambiaría el análisis, si existe]

RAMAS: (omitir en CALM)
A — [qué ocurre si el patrón continúa] | prob=[X] reversibilidad=[alta|media|baja]
B — [qué ocurre con intervención mínima] | prob=[X] reversibilidad=[alta|media|baja]
C — [qué ocurre sin acción] | prob=[X] reversibilidad=[alta|media|baja]

CONCLUSIÓN:
[Rama con mejor ratio viabilidad + razón anclada en el patrón]
[Acción mínima verificable — no estado, no consejo]

---
REGLAS DURAS:
— Nunca describas lo que Guillermo ya dijo — infiere lo que no dijo
— Nunca devuelvas estatus, logros, o perfil profesional
— Nunca uses "deberías" ni "considera"
— El insight debe nombrar algo que Guillermo no nombró explícitamente
— Máximo 6 oraciones en total
— Sin bullets decorativos, sin emojis, sin headers innecesarios
— Las etiquetas MODO / INSIGHT / RAMAS / CONCLUSIÓN son los únicos headers permitidos

MODOS:
CALM — solo INSIGHT, sin ramas. Para inputs fragmentados o de alta entropía.
ANALYZE — INSIGHT + RAMAS. Para situaciones con actor observable.
DECIDE — INSIGHT + RAMAS con reversibilidad explícita. Para decisiones implícitas.
SIMULATE — INSIGHT + RAMAS con horizonte temporal. Para escenarios futuros.
"""

class NexusEngine:
    def __init__(self):
        logger.info("Cargando Nexus v4...")
        self.pipeline = NexusPipeline(target_dim=64)
        self.client   = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("Nexus v4 listo.")

    def process(self, text: str) -> tuple:
        ctx         = self.pipeline.run(text)
        instruction = ctx.primary_instruction.value.upper()
        domain      = ctx.dominant_domain.value
        attractor_p = ctx.attractor_probability
        entropy     = ctx.operator_entropy

        # Mapear instrucción Nexus → modo de respuesta
        mode_map = {
            "MAP_TENSION":     "ANALYZE",
            "SIMULATE_BRANCH": "SIMULATE",
            "SURFACE_PATTERN": "ANALYZE",
            "CONTAIN":         "CALM",
            "OPERATOR_CHECK":  "CALM",
            "FLAG_EMERGENCE":  "SIMULATE",
        }
        mode = mode_map.get(instruction, "ANALYZE")

        # Instrucción interna al LLM — no aparece en output
        internal_msg = (
            f"MODO_SUGERIDO={mode} | "
            f"DOMINIO={domain} | "
            f"ATRACTOR_P={attractor_p:.2f} | "
            f"ACTORES={ctx.n_actors} | "
            f"ENTROPIA={entropy:.2f} | "
            f"EMERGENCIA={ctx.emergence_flag}\n\n"
            f"INPUT: {text}"
        )

        resp = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=350,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": internal_msg}
            ]
        )
        return resp.choices[0].message.content.strip(), mode, domain, ctx

nexus: Optional[NexusEngine] = None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Nexus v4 activo.\n\n"
        "Mándame una situación, una fricción, una decisión, un fragmento.\n"
        "No necesitas formato. El tensor procesa la geometría.\n"
        "Yo infiero lo que el patrón implica hacia adelante.\n\n"
        "/nexus — ver diagnóstico del tensor en tu último input."
    )

async def nexus_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    last = context.user_data.get("last_ctx")
    if not last:
        await update.message.reply_text("Sin input previo.")
        return
    await update.message.reply_text(
        f"TENSOR:\n"
        f"instrucción  → {last['instruction']}\n"
        f"modo         → {last['mode']}\n"
        f"dominio      → {last['domain']}\n"
        f"atractor     → {last['attractor']} (P={last['attractor_p']:.3f})\n"
        f"interferencia→ {last['interference']}\n"
        f"actores      → {last['n_actors']}\n"
        f"emergencia   → {last['emergence']}\n"
        f"confianza    → {last['confidence']:.3f}"
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
        response, mode, domain, ctx = nexus.process(text)

        context.user_data["last_ctx"] = {
            "instruction":  ctx.primary_instruction.value,
            "mode":         mode,
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
    app.add_handler(CommandHandler("nexus", nexus_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("NEXUS BOT v4 INICIADO")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
