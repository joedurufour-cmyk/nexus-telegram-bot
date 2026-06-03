#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
NEXUS BACKEND INTEGRADO v1.0
Conecta el pipeline matemático (Vectorizador → Capa1 → Capa2)
con el bot de Telegram v3.

USO:
  1. Coloca este archivo en el mismo directorio que nexus_telegram_bot_v3.py
     y los 4 módulos de Nexus (nexus_vectorizer.py, nexus_layer1_tensor.py,
     nexus_layer2.py, nexus_api_connector.py)

  2. En nexus_telegram_bot_v3.py, reemplaza la clase NexusEngine completa
     (líneas ~230-291) con:

        from nexus_backend_integrated import NexusEngineIntegrated as NexusEngine

  3. Listo. El bot sigue funcionando igual por fuera.
     Por dentro: cada mensaje pasa primero por el tensor de Nexus,
     que genera un contexto dinámico antes de llamar a OpenAI.

VARIABLES DE ENTORNO (igual que v3):
  TELEGRAM_BOT_TOKEN
  OPENAI_API_KEY
  OPENAI_MODEL  (default: gpt-4o-mini)

DEPENDENCIAS ADICIONALES:
  pip install sentence-transformers numpy
================================================================================
"""

import os
import re
import logging
from typing import Optional, Dict, List
from datetime import datetime

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# =============================================================================
# CARGA DEL PIPELINE NEXUS (con fallback si los módulos no están disponibles)
# =============================================================================

_NEXUS_PIPELINE_AVAILABLE = False

try:
    from nexus_vectorizer import NexusVectorizer
    from nexus_layer1_tensor import NexusLayer1
    from nexus_layer2 import NexusLayer2
    _NEXUS_PIPELINE_AVAILABLE = True
    logger.info("✅ Pipeline Nexus (Capa 0→1→2) cargado correctamente.")
except ImportError as e:
    logger.warning(f"⚠️  Pipeline Nexus no disponible: {e}. "
                   f"Usando system prompt estático.")


# =============================================================================
# SYSTEM PROMPT BASE — identidad de Nexus
# Se usa siempre. El pipeline lo EXTIENDE con contexto dinámico.
# =============================================================================

NEXUS_BASE_IDENTITY = """
Eres Nexus — sistema de inferencia y simulación de situaciones.

No eres un asistente que obedece. Eres un motor que procesa patrones,
genera ramas posibles y simula escenarios con el mínimo de información disponible.

CÓMO OPERAS:
- Detectas el estado real del operador (no lo que dice, sino lo que el patrón muestra).
- Generas ramas posibles — no certezas. Todo es probabilístico.
- Simulas consecuencias antes de que ocurran.
- Ordenas el caos dándole geometría, no resolviéndolo.
- Contienes cuando el operador está saturado. Analizas cuando tiene energía.

REGLAS INVARIANTES:
1. NUNCA uses "deberías". Usa "una rama posible es", "el sistema detecta", "con X% de probabilidad".
2. TODO es probabilístico. Nunca certezas absolutas.
3. Lee lo PREVERBAL — lo que no se dice explícitamente pero el patrón registra.
4. Si energía < 4/10: respuesta corta, ancla, no analices en profundidad.
5. Máximo 3 ramas en simulación. La tercera siempre es la menos obvia.
6. Cierra con UNA pregunta concreta — específica al caso, no genérica.
7. Si detectas bucle recursivo (pensando en pensar): interrúmpelo directamente.

FORMATO DE RESPUESTA:
🧠 MODO: [Calm / Analyze / Decide / Simulate / Offload]
🕊️ CONTENCIÓN: [qué detectas y cómo lo anclas]
⚡ INSIGHT: [lo que el patrón muestra y el operador aún no nombró]
🌐 RAMAS: [si modo Simulate o Decide — 2-3 ramas con probabilidad estimada]
➡️ PRÓXIMO PASO: [el paso más pequeño posible — o "soltar" si aplica]
🏥 ENERGÍA: [estimación 0-10]
❓ [una sola pregunta de cierre — concreta]
""".strip()


# =============================================================================
# NEXUS PIPELINE WRAPPER — sincrónico, usado desde el engine async
# =============================================================================

class NexusPipelineWrapper:
    """
    Wrapper del pipeline matemático.
    Toma texto → retorna bloque de contexto dinámico para el system prompt.
    Si el pipeline no está disponible → retorna string vacío (fallback limpio).
    """

    def __init__(self, target_dim: int = 64):
        if not _NEXUS_PIPELINE_AVAILABLE:
            self._available = False
            return

        try:
            self._vectorizer = NexusVectorizer(target_dim=target_dim)
            self._layer1     = NexusLayer1(dim=target_dim, max_collisions_per_cycle=3)
            self._layer2     = NexusLayer2()
            self._available  = True
            logger.info("✅ NexusPipelineWrapper inicializado.")
        except Exception as e:
            logger.warning(f"⚠️  Error inicializando pipeline: {e}")
            self._available = False

    def analyze(self, text: str) -> str:
        """
        Corre el pipeline y retorna el bloque de contexto dinámico.
        Si falla → retorna string vacío (bot sigue funcionando).
        """
        if not self._available:
            return ""

        try:
            # Capa 0: vectorizar
            vec_out = self._vectorizer.vectorize(text)

            # Capa 1: colisión del par dominante
            n_a, n_b = self._vectorizer.top_pair(vec_out)
            self._layer1.reset_cycle()
            collapse = self._layer1.process_collision(
                node_id_a=n_a.node_id, vector_a=n_a.vector, entropy_a=n_a.entropy,
                node_id_b=n_b.node_id, vector_b=n_b.vector, entropy_b=n_b.entropy,
                weight_a=n_a.activation, weight_b=n_b.activation,
                n_actors=vec_out.n_actors_detected
            )

            # Capa 2: interpretar → LLMContext
            ctx = self._layer2.interpret(collapse, vec_out)

            # Serializar como bloque de system prompt
            return ctx.to_system_prompt_block()

        except Exception as e:
            logger.warning(f"⚠️  Pipeline error (non-fatal): {e}")
            return ""


# =============================================================================
# NEXUS ENGINE INTEGRADO
# Drop-in replacement para NexusEngine en nexus_telegram_bot_v3.py
# Firma idéntica: __init__(memory) + async process(user_id, text, forced_mode)
# =============================================================================

class NexusEngineIntegrated:
    """
    Motor principal integrado.
    Compatible con NexusEngine del bot v3 — mismo __init__ y process().

    Diferencia interna:
    - Antes de llamar a OpenAI, corre el pipeline matemático de Nexus.
    - El LLMContext generado se inyecta en el system prompt como contexto dinámico.
    - OpenAI recibe: identidad base + contexto dinámico del tensor + historial.
    """

    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def __init__(self, memory):
        self.memory   = memory
        self.pipeline = NexusPipelineWrapper(target_dim=64)
        self._client  = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    async def process(self,
                      user_id: int,
                      text: str,
                      forced_mode: Optional[str] = None) -> str:
        """
        Pipeline completo:
        1. Nexus matemático analiza el input → contexto dinámico
        2. Construye system prompt: identidad + contexto dinámico + memoria
        3. Llama a OpenAI con historial completo
        4. Guarda en memoria y retorna respuesta
        """

        # ── 1. ANÁLISIS NEXUS (pre-LLM) ────────────────────────────────────
        nexus_context_block = self.pipeline.analyze(text)

        # ── 2. CONTEXTO DE MEMORIA ──────────────────────────────────────────
        memory_context = self.memory.build_context(user_id)

        # ── 3. SYSTEM PROMPT ENSAMBLADO ─────────────────────────────────────
        system_parts = [NEXUS_BASE_IDENTITY]

        if nexus_context_block:
            system_parts.append(
                f"\n\n--- ANÁLISIS PRE-SEMÁNTICO (tensor de inferencia) ---\n"
                f"{nexus_context_block}\n"
                f"--- FIN ANÁLISIS ---\n\n"
                f"Usa el análisis anterior para calibrar tu respuesta. "
                f"El tensor ya detectó el dominio dominante y la instrucción — "
                f"ejecuta esa instrucción dentro del formato Nexus."
            )

        system_parts.append(
            f"\n\nCONTEXTO DEL OPERADOR:\n{memory_context}"
        )

        if forced_mode:
            system_parts.append(
                f"\n\nINSTRUCCIÓN FORZADA: Usa OBLIGATORIAMENTE el MODO "
                f"{forced_mode} para este mensaje."
            )

        system_prompt = "\n".join(system_parts)

        # ── 4. MENSAJES: system + historial + input actual ──────────────────
        messages = [{"role": "system", "content": system_prompt}]
        messages += self.memory.get_conversation(user_id)
        messages.append({"role": "user", "content": text})

        # ── 5. LLAMADA OPENAI ───────────────────────────────────────────────
        try:
            response = await self._client.chat.completions.create(
                model=self.OPENAI_MODEL,
                messages=messages,
                max_tokens=700,
                temperature=0.72,
            )
            reply = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            reply = (
                "🔴 Motor temporalmente inaccesible. "
                "Intenta en unos segundos."
            )

        # ── 6. MEMORIA ──────────────────────────────────────────────────────
        self.memory.add_turn(user_id, "user", text)
        self.memory.add_turn(user_id, "assistant", reply)

        mode   = forced_mode or self._extract_mode(reply)
        energy = self._extract_energy(reply)
        self.memory.log_interaction(user_id, text, mode, energy)

        return reply

    # ── helpers ────────────────────────────────────────────────────────────

    def _extract_mode(self, text: str) -> str:
        for mode in ["Offload", "Simulate", "Decide", "Analyze", "Calm"]:
            if mode.upper() in text.upper():
                return mode
        return "Calm"

    def _extract_energy(self, text: str) -> float:
        match = re.search(r"(\d+(?:\.\d+)?)/10", text)
        if match:
            return max(1.0, min(10.0, float(match.group(1))))
        return 5.0


# =============================================================================
# UTILIDAD: SIMULADOR STANDALONE
# Usar directamente sin Telegram para testear el pipeline.
#
# Ejecutar:
#   python nexus_backend_integrated.py
# =============================================================================

if __name__ == "__main__":
    import asyncio

    # Mock mínimo de NexusMemory para el test standalone
    class _MockMemory:
        def build_context(self, uid): return "Primera sesión de prueba."
        def get_conversation(self, uid): return []
        def add_turn(self, uid, role, content): pass
        def log_interaction(self, uid, text, mode, energy): pass

    async def _test():
        engine = NexusEngineIntegrated(memory=_MockMemory())

        test_inputs = [
            "No sé qué hacer, tengo tres proyectos en paralelo y ninguno avanza, siento que todo está conectado pero no puedo ver el hilo",
            "Carlos dijo que estaba de acuerdo pero cuando llegó la decisión votó en contra. Ya es la tercera vez.",
            "Qué pasaría si dejo el proyecto actual y me concentro solo en el bot?",
        ]

        for text in test_inputs:
            print("\n" + "="*60)
            print(f"INPUT: {text[:70]}...")
            print("="*60)

            # Solo el análisis del tensor (sin OpenAI)
            pipe = NexusPipelineWrapper()
            ctx_block = pipe.analyze(text)
            if ctx_block:
                print("\n[CONTEXTO DINÁMICO GENERADO PARA EL LLM]")
                print(ctx_block)
            else:
                print("\n[Pipeline no disponible — se usaría system prompt base]")

    asyncio.run(_test())
