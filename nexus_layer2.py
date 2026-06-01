#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
NEXUS — CAPA 2: INTÉRPRETE DE COLAPSO → CONTEXTO LLM

Responsabilidad única:
    CollapseResult (números de Capa 1)
        →  LLMContext (estructura semántica para el LLM)

El LLM recibe LLMContext como su system prompt dinámico.
No recibe los números crudos. No recibe el tensor.
Recibe una estructura que le dice QUÉ inferir y CÓMO ponderarlo.

PIPELINE INTERNO:
    CollapseResult  +  VectorizerOutput
        ↓
    DomainInterpreter     — traduce atractor + rutas a marco cognitivo
        ↓
    IntensityClassifier   — clasifica magnitud de interferencia
        ↓
    EmergenceTranslator   — si CHSH > 2.0 → instrucción de rama nueva
        ↓
    OperatorStateAdapter  — ajusta tono/profundidad según estado_operador
        ↓
    LLMContext            — system prompt dinámico + señales de ponderación
        ↓
    Anthropic API call    — LLM genera output conversacional

INVARIANTE CRÍTICO:
    Capa 2 NO genera lenguaje natural propio.
    Construye el MARCO para que el LLM lo genere.
    La diferencia: Capa 2 es estructura, LLM es expresión.
================================================================================
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

# Importar estructuras de Capa 1
import sys
sys.path.insert(0, '/mnt/user-data/outputs')
from nexus_layer1_tensor import (
    CollapseResult, CollapseStatus,
    InterferenceType, EmergenceEvent
)
from nexus_vectorizer import VectorizerOutput, DomainSignal


# =============================================================================
# TAXONOMÍA DE ESTADOS — lo que Capa 2 puede inferir de los números
# =============================================================================

class CognitiveDomain(Enum):
    """
    Traducción de DomainSignal a marco cognitivo para el LLM.
    El LLM opera en este vocabulario, no en el de Capa 1.
    """
    SYSTEM_VS_DECLARATION  = "tension_activa"      # delta declarado/real
    GRAVITATIONAL_PATTERN  = "punto_atractor"      # patrón que se repite
    ACTOR_SYSTEM_COLLISION = "interferencia_actor" # actor + sistema co-activos
    OPERATOR_STATE         = "estado_operador"     # estado del emisor


class InterferenceIntensity(Enum):
    NULL         = "null"         # |Int| < 0.05  — nodos independientes
    LOW          = "low"          # |Int| 0.05-0.20
    MEDIUM       = "medium"       # |Int| 0.20-0.50
    HIGH         = "high"         # |Int| > 0.50


class OutputInstruction(Enum):
    """
    Instrucción primaria que Capa 2 envía al LLM.
    El LLM sigue esta instrucción — no decide por sí mismo el modo.
    """
    CONTAIN          = "contain"       # contener, no resolver
    SURFACE_PATTERN  = "surface"       # nombrar el patrón que se repite
    MAP_TENSION      = "map_tension"   # mapear delta declarado/real
    SIMULATE_BRANCH  = "simulate"      # generar ramas posibles
    FLAG_EMERGENCE   = "emergence"     # señalar que algo nuevo está surgiendo
    OPERATOR_CHECK   = "operator"      # verificar estado del operador primero


# =============================================================================
# ESTRUCTURA DE OUTPUT — LLMContext
# =============================================================================

@dataclass
class SignalWeight:
    """
    Una señal con su peso — lo que el LLM debe ponderar al responder.
    """
    signal: str           # descripción de la señal (concisa, sin narrativa)
    weight: float         # [0, 1] — importancia relativa
    domain: str           # dominio de origen


@dataclass
class LLMContext:
    """
    Contexto dinámico para el LLM.
    
    El LLM recibe esto como parte de su system prompt en cada llamada.
    Cambia con cada input — no es estático.
    
    Contiene:
    - primary_instruction: qué hacer primero
    - dominant_domain: sobre qué eje operar
    - weighted_signals: qué ponderar y cuánto
    - emergence_flag: si debe generar rama nueva
    - operator_state: en qué condición está el emisor
    - confidence: cuánto confiar en el análisis de Capa 1
    - response_constraints: límites para el output del LLM
    """
    # Instrucción primaria
    primary_instruction: OutputInstruction
    dominant_domain: CognitiveDomain
    
    # Señales ponderadas para el LLM
    weighted_signals: List[SignalWeight]
    
    # Estado de emergencia
    emergence_flag: bool
    emergence_probability: float        # [0,1] — si emergence_flag=True
    
    # Estado del operador
    operator_entropy: float             # entropía del nodo estado_operador
    operator_activation: float          # qué tan presente está el operador en el input
    n_actors: int                       # número de actores detectados
    yukalov_uncertainty: float          # ±0.25 si multi-actor
    
    # Confianza del sistema
    layer1_confidence: float
    interference_intensity: InterferenceIntensity
    attractor_node: str
    attractor_probability: float
    
    # Constraints para el LLM
    response_constraints: Dict[str, object]
    
    # Metadata de trazabilidad
    collision_id: str
    raw_input_length: int

    def to_system_prompt_block(self) -> str:
        """
        Serializa LLMContext como bloque de system prompt.
        Este es el único punto donde Capa 2 produce texto —
        y es texto de instrucción, no de respuesta.
        """
        signals_formatted = "\n".join(
            f"  [{s.weight:.2f}] {s.signal} ({s.domain})"
            for s in sorted(self.weighted_signals, key=lambda x: x.weight, reverse=True)
        )

        constraints = "\n".join(
            f"  {k}: {v}"
            for k, v in self.response_constraints.items()
        )

        emergence_block = ""
        if self.emergence_flag:
            emergence_block = f"""
EMERGENCIA DETECTADA (P={self.emergence_probability:.2f}):
  El sistema detecta que la colisión de patrones produce algo
  que no estaba en ninguno de los dominios por separado.
  Instrucción: proponer una rama que el emisor no ha considerado.
  No promediar las opciones conocidas. Generar una tercera.
"""

        operator_block = ""
        if self.operator_activation > 0.15:
            entropy_label = (
                "ALTA (estado ambiguo — contener antes de analizar)"
                if self.operator_entropy > 0.6
                else "MEDIA" if self.operator_entropy > 0.3
                else "BAJA (estado claro — proceder al análisis)"
            )
            operator_block = f"""
ESTADO DEL OPERADOR:
  entropía: {self.operator_entropy:.3f} → {entropy_label}
  activación: {self.operator_activation:.3f}
  instrucción: {"CONTENER PRIMERO. No procesar análisis complejo." if self.operator_entropy > 0.6 else "Proceder con análisis normal."}
"""

        multi_actor_block = ""
        if self.n_actors > 1:
            multi_actor_block = f"""
SISTEMA MULTI-ACTOR ({self.n_actors} actores detectados):
  Corrección Yukalov activa: ±{self.yukalov_uncertainty:.2f}
  Las probabilidades tienen incertidumbre inherente de ±25%.
  No presentar rutas como deterministas.
"""

        return f"""
=== NEXUS LAYER 2 — CONTEXTO DINÁMICO ===

INSTRUCCIÓN PRIMARIA: {self.primary_instruction.value.upper()}
DOMINIO DOMINANTE:    {self.dominant_domain.value}
ATRACTOR:             {self.attractor_node} (P={self.attractor_probability:.3f})
CONFIANZA SISTEMA:    {self.layer1_confidence:.3f}
INTERFERENCIA:        {self.interference_intensity.value} 
COLISIÓN ID:          {self.collision_id}

SEÑALES PONDERADAS (de mayor a menor peso):
{signals_formatted}
{emergence_block}{operator_block}{multi_actor_block}
CONSTRAINTS DE RESPUESTA:
{constraints}

=== FIN CONTEXTO DINÁMICO ===
""".strip()

    def to_dict(self) -> Dict:
        """Serialización para logging y trazabilidad."""
        return {
            "collision_id": self.collision_id,
            "primary_instruction": self.primary_instruction.value,
            "dominant_domain": self.dominant_domain.value,
            "attractor_node": self.attractor_node,
            "attractor_probability": self.attractor_probability,
            "layer1_confidence": self.layer1_confidence,
            "interference_intensity": self.interference_intensity.value,
            "emergence_flag": self.emergence_flag,
            "emergence_probability": self.emergence_probability,
            "n_actors": self.n_actors,
            "yukalov_uncertainty": self.yukalov_uncertainty,
            "operator_entropy": self.operator_entropy,
            "weighted_signals": [
                {"signal": s.signal, "weight": s.weight, "domain": s.domain}
                for s in self.weighted_signals
            ],
            "response_constraints": self.response_constraints
        }


# =============================================================================
# COMPONENTES DE INTERPRETACIÓN
# =============================================================================

class IntensityClassifier:
    """Clasifica la magnitud de interferencia en categoría operacional."""

    THRESHOLDS = {
        InterferenceIntensity.NULL:   (0.00, 0.05),
        InterferenceIntensity.LOW:    (0.05, 0.20),
        InterferenceIntensity.MEDIUM: (0.20, 0.50),
        InterferenceIntensity.HIGH:   (0.50, 1.00),
    }

    def classify(self, magnitude: float) -> InterferenceIntensity:
        for intensity, (low, high) in self.THRESHOLDS.items():
            if low <= magnitude < high:
                return intensity
        return InterferenceIntensity.HIGH


class DomainInterpreter:
    """
    Traduce el atractor y las rutas de Capa 1
    a instrucción primaria y dominio cognitivo.
    
    Lógica:
    - El atractor dice SOBRE QUÉ operar
    - La interferencia dice CON QUÉ INTENSIDAD
    - La emergencia dice SI generar rama nueva
    - El estado_operador dice SI contener primero
    """

    # Mapa: nodo atractor + intensidad → instrucción
    INSTRUCTION_MAP = {
        ("tension_activa",      InterferenceIntensity.NULL):   OutputInstruction.MAP_TENSION,
        ("tension_activa",      InterferenceIntensity.LOW):    OutputInstruction.MAP_TENSION,
        ("tension_activa",      InterferenceIntensity.MEDIUM): OutputInstruction.SIMULATE_BRANCH,
        ("tension_activa",      InterferenceIntensity.HIGH):   OutputInstruction.SIMULATE_BRANCH,
        ("punto_atractor",      InterferenceIntensity.NULL):   OutputInstruction.SURFACE_PATTERN,
        ("punto_atractor",      InterferenceIntensity.LOW):    OutputInstruction.SURFACE_PATTERN,
        ("punto_atractor",      InterferenceIntensity.MEDIUM): OutputInstruction.SIMULATE_BRANCH,
        ("punto_atractor",      InterferenceIntensity.HIGH):   OutputInstruction.SIMULATE_BRANCH,
        ("interferencia_actor", InterferenceIntensity.NULL):   OutputInstruction.CONTAIN,
        ("interferencia_actor", InterferenceIntensity.LOW):    OutputInstruction.MAP_TENSION,
        ("interferencia_actor", InterferenceIntensity.MEDIUM): OutputInstruction.SIMULATE_BRANCH,
        ("interferencia_actor", InterferenceIntensity.HIGH):   OutputInstruction.FLAG_EMERGENCE,
        ("estado_operador",     InterferenceIntensity.NULL):   OutputInstruction.OPERATOR_CHECK,
        ("estado_operador",     InterferenceIntensity.LOW):    OutputInstruction.CONTAIN,
        ("estado_operador",     InterferenceIntensity.MEDIUM): OutputInstruction.CONTAIN,
        ("estado_operador",     InterferenceIntensity.HIGH):   OutputInstruction.OPERATOR_CHECK,
    }

    def interpret(self,
                  collapse: CollapseResult,
                  intensity: InterferenceIntensity,
                  operator_entropy: float) -> Tuple[OutputInstruction, CognitiveDomain]:

        # Override: si entropía del operador es crítica → contener primero
        if operator_entropy > 0.75:
            return OutputInstruction.OPERATOR_CHECK, CognitiveDomain.OPERATOR_STATE

        # Override: si emergencia detectada → siempre flag
        if collapse.emergence_detected:
            return OutputInstruction.FLAG_EMERGENCE, CognitiveDomain.ACTOR_SYSTEM_COLLISION

        # Lookup en mapa
        key = (collapse.attractor_node, intensity)
        instruction = self.INSTRUCTION_MAP.get(key, OutputInstruction.CONTAIN)

        # Dominio cognitivo desde atractor
        domain_map = {
            "tension_activa":      CognitiveDomain.SYSTEM_VS_DECLARATION,
            "punto_atractor":      CognitiveDomain.GRAVITATIONAL_PATTERN,
            "interferencia_actor": CognitiveDomain.ACTOR_SYSTEM_COLLISION,
            "estado_operador":     CognitiveDomain.OPERATOR_STATE,
        }
        domain = domain_map.get(collapse.attractor_node, CognitiveDomain.OPERATOR_STATE)

        return instruction, domain


class SignalWeightBuilder:
    """
    Construye la lista de señales ponderadas para el LLM.
    
    Cada señal es una observación derivada de los números de Capa 1.
    El peso determina cuánta atención debe darle el LLM.
    """

    # Señales por dominio — concisas, sin narrativa
    DOMAIN_SIGNALS = {
        "tension_activa": [
            ("delta declarado/observado activo",               0.90),
            ("comportamiento bajo presión difiere de declaración", 0.85),
            ("punto de quiebre no conocido aún",               0.70),
        ],
        "punto_atractor": [
            ("patrón de retorno detectado",                    0.88),
            ("sistema converge sin ser dirigido",              0.82),
            ("resistencia al cambio estructural",              0.65),
        ],
        "interferencia_actor": [
            ("actor y sistema co-activos con tensión",         0.92),
            ("tipo de interferencia a determinar",             0.78),
            ("emergencia posible en intersección",             0.60),
        ],
        "estado_operador": [
            ("estado cognitivo del emisor presente",           0.85),
            ("ambigüedad preverbal detectada",                 0.72),
            ("capacidad de procesamiento a evaluar",           0.55),
        ],
    }

    def build(self,
              collapse: CollapseResult,
              vectorizer_out: VectorizerOutput,
              intensity: InterferenceIntensity) -> List[SignalWeight]:

        signals = []

        # Señales base del dominio atractor
        base_signals = self.DOMAIN_SIGNALS.get(collapse.attractor_node, [])
        for signal_text, base_weight in base_signals:
            # Modular por probabilidad del atractor
            weight = base_weight * collapse.attractor_probability
            signals.append(SignalWeight(
                signal=signal_text,
                weight=float(np.clip(weight, 0.0, 1.0)),
                domain=collapse.attractor_node
            ))

        # Señal de interferencia si no es nula
        if intensity != InterferenceIntensity.NULL:
            itype = collapse.interference_type.name.lower()
            signals.append(SignalWeight(
                signal=f"interferencia {itype} entre dominios ({collapse.interference_magnitude:.4f})",
                weight=float(np.clip(collapse.interference_magnitude * 2, 0.0, 1.0)),
                domain="tensor"
            ))

        # Señal de multi-actor si aplica
        if vectorizer_out.n_actors_detected > 1:
            signals.append(SignalWeight(
                signal=f"{vectorizer_out.n_actors_detected} actores detectados — incertidumbre ±25%",
                weight=0.75,
                domain="yukalov"
            ))

        # Señal de emergencia si aplica
        if collapse.emergence_detected and collapse.emergence_event:
            signals.append(SignalWeight(
                signal=f"emergencia cuántica CHSH={collapse.emergence_event.chsh_value:.3f} — rama nueva disponible",
                weight=collapse.emergence_event.emergence_probability,
                domain="emergence"
            ))

        # Señal de entropía global
        signals.append(SignalWeight(
            signal=f"entropía global del input: {vectorizer_out.global_entropy:.3f}",
            weight=vectorizer_out.global_entropy * 0.6,
            domain="entropy"
        ))

        return sorted(signals, key=lambda x: x.weight, reverse=True)


class ResponseConstraintBuilder:
    """
    Define los constraints de respuesta para el LLM
    basados en el estado del sistema.
    """

    def build(self,
              instruction: OutputInstruction,
              intensity: InterferenceIntensity,
              operator_entropy: float,
              n_actors: int,
              confidence: float) -> Dict[str, object]:

        constraints = {}

        # Longitud de respuesta
        if instruction == OutputInstruction.OPERATOR_CHECK:
            constraints["max_length"] = "2-3 oraciones"
            constraints["tone"] = "contenedor, sin análisis"
        elif instruction == OutputInstruction.CONTAIN:
            constraints["max_length"] = "3-4 oraciones"
            constraints["tone"] = "ancla, espacio sin aplastar"
        elif instruction == OutputInstruction.SURFACE_PATTERN:
            constraints["max_length"] = "4-5 oraciones"
            constraints["tone"] = "nombrar el patrón sin juzgarlo"
        elif instruction == OutputInstruction.MAP_TENSION:
            constraints["max_length"] = "5-7 oraciones"
            constraints["tone"] = "mapear delta, no resolver"
        elif instruction == OutputInstruction.SIMULATE_BRANCH:
            constraints["max_length"] = "6-8 oraciones"
            constraints["tone"] = "ramas posibles, probabilístico"
            constraints["n_branches"] = min(3, max(2, n_actors))
        elif instruction == OutputInstruction.FLAG_EMERGENCE:
            constraints["max_length"] = "5-6 oraciones"
            constraints["tone"] = "señalar lo inesperado sin alarmar"
            constraints["requires_new_branch"] = True

        # Determinismo
        if n_actors > 1:
            constraints["certainty_language"] = "probabilístico — nunca determinista"
        else:
            constraints["certainty_language"] = "condicional — 'una ruta posible es'"

        # Confianza del sistema
        if confidence < 0.5:
            constraints["confidence_note"] = f"señal débil (conf={confidence:.2f}) — respuesta cautelosa"

        # Entropía del operador
        if operator_entropy > 0.6:
            constraints["operator_note"] = "estado ambiguo — priorizar contención sobre insight"

        return constraints


# =============================================================================
# CAPA 2 — ORQUESTADOR
# =============================================================================

class NexusLayer2:
    """
    Intérprete principal de Capa 2.
    
    Recibe:
        collapse:        CollapseResult de Capa 1
        vectorizer_out:  VectorizerOutput del vectorizador
    
    Entrega:
        LLMContext — estructura lista para system prompt del LLM
    
    Uso:
        layer2 = NexusLayer2()
        ctx = layer2.interpret(collapse_result, vectorizer_output)
        system_prompt = ctx.to_system_prompt_block()
        # → llamar API con system_prompt + input del usuario
    """

    def __init__(self):
        self.intensity_classifier   = IntensityClassifier()
        self.domain_interpreter     = DomainInterpreter()
        self.signal_builder         = SignalWeightBuilder()
        self.constraint_builder     = ResponseConstraintBuilder()

    def _get_operator_state(self, vectorizer_out: VectorizerOutput) -> Tuple[float, float]:
        """
        Extrae entropía y activación del nodo estado_operador.
        Si no está presente → defaults conservadores.
        """
        for node in vectorizer_out.active_nodes:
            if node.node_id == DomainSignal.ESTADO_OPERADOR.value:
                return node.entropy, node.activation
        return 0.5, 0.0

    def interpret(self,
                  collapse: CollapseResult,
                  vectorizer_out: VectorizerOutput) -> LLMContext:
        """
        Pipeline completo de interpretación.
        
        1. Clasificar intensidad de interferencia
        2. Determinar instrucción primaria y dominio
        3. Construir señales ponderadas
        4. Construir constraints de respuesta
        5. Ensamblar LLMContext
        """
        # 1. Intensidad
        intensity = self.intensity_classifier.classify(collapse.interference_magnitude)

        # 2. Estado del operador
        operator_entropy, operator_activation = self._get_operator_state(vectorizer_out)

        # 3. Instrucción y dominio
        instruction, domain = self.domain_interpreter.interpret(
            collapse, intensity, operator_entropy
        )

        # 4. Señales ponderadas
        signals = self.signal_builder.build(collapse, vectorizer_out, intensity)

        # 5. Constraints
        constraints = self.constraint_builder.build(
            instruction=instruction,
            intensity=intensity,
            operator_entropy=operator_entropy,
            n_actors=vectorizer_out.n_actors_detected,
            confidence=collapse.confidence
        )

        # 6. Emergencia
        emergence_probability = 0.0
        if collapse.emergence_detected and collapse.emergence_event:
            emergence_probability = collapse.emergence_event.emergence_probability

        return LLMContext(
            primary_instruction=instruction,
            dominant_domain=domain,
            weighted_signals=signals,
            emergence_flag=collapse.emergence_detected,
            emergence_probability=emergence_probability,
            operator_entropy=operator_entropy,
            operator_activation=operator_activation,
            n_actors=vectorizer_out.n_actors_detected,
            yukalov_uncertainty=collapse.yukalov_correction,
            layer1_confidence=collapse.confidence,
            interference_intensity=intensity,
            attractor_node=collapse.attractor_node,
            attractor_probability=collapse.attractor_probability,
            response_constraints=constraints,
            collision_id=collapse.collision_id,
            raw_input_length=len(vectorizer_out.raw_input.split())
        )


# =============================================================================
# ORQUESTADOR COMPLETO — Capa 0 → Capa 1 → Capa 2
# =============================================================================

class NexusPipeline:
    """
    Pipeline completo: texto → LLMContext.
    
    Encapsula Vectorizador + Capa1 + Capa2.
    El llamador solo necesita pasar texto y recibe LLMContext.
    
    Uso mínimo:
        pipeline = NexusPipeline()
        ctx = pipeline.run("Carlos evadió la auditoría")
        print(ctx.to_system_prompt_block())
    """

    def __init__(self, target_dim: int = 64):
        from nexus_vectorizer import NexusVectorizer
        from nexus_layer1_tensor import NexusLayer1

        self.vectorizer = NexusVectorizer(target_dim=target_dim)
        self.layer1     = NexusLayer1(dim=target_dim, max_collisions_per_cycle=3)
        self.layer2     = NexusLayer2()

    def run(self, text: str,
            attractor_bias: Optional[Dict[str, float]] = None) -> LLMContext:
        """
        Pipeline completo para un input de texto.
        Retorna LLMContext listo para el LLM.
        """
        # Capa 0: vectorizar
        vec_out = self.vectorizer.vectorize(text)

        # Seleccionar par principal para colisión
        n_a, n_b = self.vectorizer.top_pair(vec_out)

        # Capa 1: colisión
        self.layer1.reset_cycle()
        collapse = self.layer1.process_collision(
            node_id_a=n_a.node_id,   vector_a=n_a.vector,   entropy_a=n_a.entropy,
            node_id_b=n_b.node_id,   vector_b=n_b.vector,   entropy_b=n_b.entropy,
            weight_a=n_a.activation, weight_b=n_b.activation,
            attractor_bias=attractor_bias,
            n_actors=vec_out.n_actors_detected
        )

        # Capa 2: interpretar
        ctx = self.layer2.interpret(collapse, vec_out)
        return ctx

    def run_multi(self, text: str,
                  attractor_bias: Optional[Dict[str, float]] = None) -> List[LLMContext]:
        """
        Pipeline con múltiples colisiones (hasta 3).
        Retorna lista de LLMContext — uno por colisión.
        El LLM recibe el de mayor confianza, los demás como contexto adicional.
        """
        from nexus_vectorizer import NexusVectorizer
        vec_out = self.vectorizer.vectorize(text)
        pairs   = self.vectorizer.all_pairs(vec_out, max_pairs=3)

        self.layer1.reset_cycle()
        contexts = []

        for n_a, n_b in pairs:
            try:
                collapse = self.layer1.process_collision(
                    node_id_a=n_a.node_id,   vector_a=n_a.vector,   entropy_a=n_a.entropy,
                    node_id_b=n_b.node_id,   vector_b=n_b.vector,   entropy_b=n_b.entropy,
                    weight_a=n_a.activation, weight_b=n_b.activation,
                    attractor_bias=attractor_bias,
                    n_actors=vec_out.n_actors_detected
                )
                ctx = self.layer2.interpret(collapse, vec_out)
                contexts.append(ctx)
            except RuntimeError:
                break  # límite de colisiones alcanzado

        # Ordenar por confianza descendente
        contexts.sort(key=lambda x: x.layer1_confidence, reverse=True)
        return contexts


# =============================================================================
# TEST — pipeline completo + system prompt generado
# =============================================================================

if __name__ == "__main__":

    pipeline = NexusPipeline(target_dim=64)

    test_cases = [
        "Carlos en la reunión dijo que todo está bien pero su tono cambió cuando mencioné la auditoría",
        "No sé por dónde empezar, todo está conectado, la empresa siempre vuelve al mismo patrón disfuncional",
        "Siento que estoy en el borde de algo, no encuentro las palabras pero hay algo ahí que no puedo articular"
    ]

    for text in test_cases:
        print("\n" + "=" * 65)
        print(f"INPUT: \"{text[:62]}...\"" if len(text) > 62 else f"INPUT: \"{text}\"")
        print("=" * 65)

        ctx = pipeline.run(text)

        print(f"\n[DIAGNÓSTICO NUMÉRICO]")
        print(f"  instrucción     : {ctx.primary_instruction.value}")
        print(f"  dominio         : {ctx.dominant_domain.value}")
        print(f"  atractor        : {ctx.attractor_node} (P={ctx.attractor_probability:.4f})")
        print(f"  interferencia   : {ctx.interference_intensity.value}")
        print(f"  emergencia      : {ctx.emergence_flag}")
        print(f"  confianza       : {ctx.layer1_confidence:.4f}")
        print(f"  actores         : {ctx.n_actors}")
        print(f"  yukalov         : {ctx.yukalov_uncertainty:.4f}")
        print(f"  op_entropy      : {ctx.operator_entropy:.4f}")

        print(f"\n[SYSTEM PROMPT DINÁMICO → LLM]")
        print(ctx.to_system_prompt_block())
