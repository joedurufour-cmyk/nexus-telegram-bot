#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
NEXUS — VECTORIZADOR (Conector Capa 0 → Capa 1)

Responsabilidad única:
    Texto / señales de entrada  →  vectores complejos en ℂᴺ
    listos para NexusLayer1.process_collision()

El vectorizador es el único punto donde el lenguaje natural existe
como dato de entrada. A partir de aquí todo es matemática.

PIPELINE INTERNO:
    raw_text
        ↓
    SentenceEmbedding (all-MiniLM-L6-v2, dim=384)
        ↓
    DomainProjection (384 → dim, por nodo)
        ↓
    PhaseEncoder (real → complejo, fase desde señales de estado)
        ↓
    EntropyEstimator (ambigüedad semántica del input)
        ↓
    VectorizedNode → listo para Capa 1

SEÑALES DE ESTADO que modulan la fase:
    tension_activa       — delta declarado/observado en el texto
    punto_atractor       — repetición de patrones en historial
    interferencia_actor  — co-presencia de actor + sistema en texto
    estado_operador      — señales cognitivas del emisor

DEPENDENCIAS:
    sentence-transformers
    numpy

NOTA: Este módulo NO llama a NexusLayer1.
      Entrega VectorizedNode — el llamador decide cuándo y cómo colisionar.
================================================================================
"""

import numpy as np
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


# =============================================================================
# NODOS CONOCIDOS — Los 4 dominios formalizados + extensibles
# =============================================================================

class DomainSignal(Enum):
    """
    Los 4 nodos formalizados de Nexus.
    Cada uno tiene un conjunto de señales léxicas que lo activan.
    No son reglas semánticas — son detectores de superficie para
    estimar qué dominio es más relevante en el input.
    """
    TENSION_ACTIVA       = "tension_activa"
    PUNTO_ATRACTOR       = "punto_atractor"
    INTERFERENCIA_ACTOR  = "interferencia_actor"
    ESTADO_OPERADOR      = "estado_operador"


# Señales léxicas por dominio — lista mínima, no exhaustiva
# La detección es probabilística, no booleana
DOMAIN_LEXICAL_SIGNALS: Dict[DomainSignal, List[str]] = {
    DomainSignal.TENSION_ACTIVA: [
        "dice", "dijo", "declaró", "afirma", "pero", "sin embargo",
        "aunque", "contradice", "cambió", "antes", "ahora", "presión",
        "tensión", "diferente", "opuesto", "real", "verdad", "fachada",
        "auditoría", "revisión", "evidencia", "comportamiento", "actuó"
    ],
    DomainSignal.PUNTO_ATRACTOR: [
        "siempre", "nunca", "otra vez", "de nuevo", "patrón", "repite",
        "vuelve", "regresa", "igual", "mismo", "historial", "historia",
        "costumbre", "tendencia", "típico", "predecible", "conocido",
        "esperado", "inevitable", "ciclo", "loop", "bucle"
    ],
    DomainSignal.INTERFERENCIA_ACTOR: [
        "organización", "empresa", "sistema", "estructura", "equipo",
        "grupo", "institución", "cultura", "política", "norma", "regla",
        "jerarquía", "liderazgo", "gestión", "proceso", "dinámica",
        "persona", "actor", "jugador", "stakeholder", "interés"
    ],
    DomainSignal.ESTADO_OPERADOR: [
        "siento", "pienso", "creo", "intuyo", "veo", "noto", "percibo",
        "estoy", "me siento", "tengo", "mi", "yo", "nosotros",
        "saturado", "cansado", "energía", "claro", "confuso", "bloqueado",
        "flujo", "flow", "insight", "idea", "conexión", "patrón"
    ]
}


# =============================================================================
# ESTRUCTURAS DE SALIDA
# =============================================================================

@dataclass
class DomainActivation:
    """
    Activación detectada de un dominio en el input.
    Probabilística — no binaria.
    """
    domain: DomainSignal
    activation_score: float     # [0, 1] — qué tan presente está el dominio
    signal_count: int           # cuántas señales léxicas dispararon
    signals_found: List[str]    # cuáles específicamente


@dataclass
class VectorizedNode:
    """
    Output del vectorizador — input directo para NexusLayer1.
    
    node_id:    identificador del dominio (DomainSignal.value)
    vector:     embedding complejo shape=(dim,) — listo para QuantumStateVector
    entropy:    ambigüedad semántica estimada [0, 1]
    activation: score de activación del dominio en este input [0, 1]
    metadata:   señales auxiliares para Capa 2
    """
    node_id: str
    vector: np.ndarray              # dtype=complex128, shape=(dim,)
    entropy: float
    activation: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class VectorizerOutput:
    """
    Output completo del vectorizador para un input dado.
    Contiene todos los nodos activados, listos para colisionar en Capa 1.
    """
    raw_input: str
    active_nodes: List[VectorizedNode]          # ordenados por activation desc
    domain_activations: List[DomainActivation]  # diagnóstico de activación
    base_embedding: np.ndarray                  # embedding original 384-dim
    n_actors_detected: int                      # para corrección Yukalov
    global_entropy: float                       # ambigüedad global del input


# =============================================================================
# COMPONENTES INTERNOS
# =============================================================================

class LexicalSignalDetector:
    """
    Detecta qué dominios están presentes en el texto.
    Opera sobre superficie léxica — no semántica profunda.
    La semántica profunda la maneja el embedding.
    """

    def __init__(self, min_score: float = 0.05):
        self.min_score = min_score

    def detect(self, text: str) -> List[DomainActivation]:
        """
        Retorna lista de DomainActivation para todos los dominios.
        score = signals_found / total_signals_in_domain (normalizado)
        """
        text_lower = text.lower()
        results = []

        for domain, signals in DOMAIN_LEXICAL_SIGNALS.items():
            found = [s for s in signals if s in text_lower]
            score = len(found) / max(len(signals), 1)

            # Boost si hay muchas señales concentradas
            if len(found) >= 3:
                score = min(score * 1.5, 1.0)

            results.append(DomainActivation(
                domain=domain,
                activation_score=float(score),
                signal_count=len(found),
                signals_found=found
            ))

        # Ordenar por score descendente
        results.sort(key=lambda x: x.activation_score, reverse=True)
        return results

    def count_actors(self, text: str) -> int:
        """
        Estima número de actores en el texto.
        Actor = nombre propio + pronombre de tercera persona + cargo.
        Mínimo 1 (el emisor).
        """
        text_lower = text.lower()
        actor_signals = [
            "carlos", "maría", "juan", "el jefe", "mi jefe", "el equipo",
            "ellos", "ella", "él", "nosotros", "la empresa", "dirección",
            "gerencia", "cliente", "proveedor", "colega", "compañero"
        ]
        found = sum(1 for s in actor_signals if s in text_lower)
        # Nombres propios — palabras con mayúscula que no son inicio de oración
        proper_names = re.findall(r'(?<!\. )[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}', text)
        found += len(proper_names)
        return max(1, min(found, 10))


class EntropyEstimator:
    """
    Estima ambigüedad semántica del input.
    Alta entropía = input vago, fragmentado, múltiples interpretaciones posibles.
    Baja entropía = input específico, concreto, unívoco.
    """

    # Señales de alta entropía
    HIGH_ENTROPY_SIGNALS = [
        "no sé", "quizás", "tal vez", "podría", "no estoy seguro",
        "depende", "complicado", "complejo", "todo", "nada", "siempre",
        "nunca", "muchas cosas", "varias", "diferentes", "no sé cómo decirlo",
        "algo", "no encuentro las palabras", "difícil de explicar"
    ]

    # Señales de baja entropía
    LOW_ENTROPY_SIGNALS = [
        "exactamente", "específicamente", "en concreto", "el problema es",
        "la causa es", "porque", "por eso", "definitivamente", "claramente",
        "obviamente", "está claro", "es evidente", "sin duda"
    ]

    def estimate(self, text: str, activations: List[DomainActivation]) -> Tuple[float, float]:
        """
        Retorna (global_entropy, per_node_entropy_default)
        """
        text_lower = text.lower()

        high_count = sum(1 for s in self.HIGH_ENTROPY_SIGNALS if s in text_lower)
        low_count  = sum(1 for s in self.LOW_ENTROPY_SIGNALS  if s in text_lower)

        # Longitud del texto como señal adicional
        # Texto muy corto → alta entropía (poco contexto)
        # Texto muy largo → alta entropía (muchas variables)
        words = len(text.split())
        length_entropy = 0.3 if words < 10 else (0.5 if words < 50 else 0.7)

        # Entropía por activaciones
        # Muchos dominios activados simultáneamente → más ambiguo
        active_domains = sum(1 for a in activations if a.activation_score > 0.1)
        domain_entropy = min(active_domains / 4.0, 1.0)

        # Combinar
        signal_entropy = 0.5
        if high_count > 0 and low_count == 0:
            signal_entropy = min(0.5 + high_count * 0.1, 0.9)
        elif low_count > 0 and high_count == 0:
            signal_entropy = max(0.5 - low_count * 0.1, 0.1)

        global_entropy = float(np.clip(
            0.4 * signal_entropy +
            0.3 * length_entropy +
            0.3 * domain_entropy,
            0.0, 1.0
        ))

        # Entropía por nodo = global ± variación pequeña
        per_node = global_entropy
        return global_entropy, per_node


class PhaseEncoder:
    """
    Convierte vector real (embedding) en vector complejo (estado cuántico).
    
    La fase compleja codifica el estado del sistema — no es arbitraria:
    - Alta tensión activa → fase cercana a π (oposición)
    - Alta certeza (baja entropía) → fase cercana a 0 (coherencia)
    - Alta emergencia esperada → fase intermedia (superposición máxima)
    
    Ecuación:
        ψ = v · e^(iφ)
        donde v = embedding real normalizado
        φ = fase determinada por estado del dominio
    """

    PHASE_MAP = {
        DomainSignal.TENSION_ACTIVA:      np.pi * 0.75,   # fase de oposición
        DomainSignal.PUNTO_ATRACTOR:      np.pi * 0.25,   # fase de convergencia
        DomainSignal.INTERFERENCIA_ACTOR: np.pi * 0.50,   # superposición máxima
        DomainSignal.ESTADO_OPERADOR:     np.pi * 0.10,   # coherencia (observador)
    }

    def encode(self,
               real_vector: np.ndarray,
               domain: DomainSignal,
               entropy: float,
               activation_score: float) -> np.ndarray:
        """
        Aplica fase compleja al vector real.
        La fase base se modula por entropía y activación.
        
        φ_final = φ_base · entropy + φ_base · (1 - activation_score) · 0.3
        """
        base_phase = self.PHASE_MAP.get(domain, np.pi * 0.5)

        # Modular fase por estado
        phase = base_phase * (0.7 + entropy * 0.3) * (0.85 + (1 - activation_score) * 0.15)

        # Aplicar rotación de fase uniforme al vector
        complex_vector = real_vector.astype(complex) * np.exp(1j * phase)

        # Normalizar
        norm = np.linalg.norm(complex_vector)
        if norm > 1e-10:
            complex_vector = complex_vector / norm

        return complex_vector


class DomainProjector:
    """
    Proyecta el embedding de dimensión alta (384) a la dimensión
    del espacio de Hilbert de Capa 1 (dim=64 por defecto).
    
    Cada dominio tiene su propia matriz de proyección — aprendida
    o inicializada con componentes principales aleatorios.
    
    En producción: fine-tuning de matrices por feedback de Capa 2.
    En esta versión: inicialización determinista por semilla de dominio.
    """

    def __init__(self, input_dim: int = 384, output_dim: int = 64):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._projectors: Dict[str, np.ndarray] = {}

        # Inicializar proyectores por dominio
        for domain in DomainSignal:
            seed = hash(domain.value) % (2**31)
            rng = np.random.RandomState(seed)
            # Matriz de proyección ortogonal (aproximada)
            M = rng.randn(output_dim, input_dim)
            # Ortonormalizar filas
            Q, _ = np.linalg.qr(M.T)
            self._projectors[domain.value] = Q[:, :output_dim].T

    def project(self, embedding: np.ndarray, domain: DomainSignal) -> np.ndarray:
        """
        Proyectar embedding a espacio de dominio.
        output shape = (output_dim,), dtype=float32
        """
        P = self._projectors[domain.value]
        projected = P @ embedding
        norm = np.linalg.norm(projected)
        if norm > 1e-10:
            projected = projected / norm
        return projected.astype(np.float32)


# =============================================================================
# VECTORIZADOR PRINCIPAL
# =============================================================================

class NexusVectorizer:
    """
    Conector entre texto y Capa 1.
    
    Uso:
        vectorizer = NexusVectorizer(target_dim=64)
        output = vectorizer.vectorize("Carlos evadió la auditoría")
        
        # output.active_nodes contiene VectorizedNode listos para Capa 1:
        for node in output.active_nodes:
            print(node.node_id, node.vector.shape, node.entropy)
    
    El llamador decide qué pares de nodos colisionar en Capa 1.
    """

    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 target_dim: int = 64,
                 min_activation: float = 0.05):
        """
        model_name:     modelo de sentence-transformers
        target_dim:     dimensión del espacio de Hilbert en Capa 1
        min_activation: umbral mínimo para incluir un nodo como activo
        """
        self.target_dim = target_dim
        self.min_activation = min_activation

        # Componentes
        self.signal_detector  = LexicalSignalDetector()
        self.entropy_estimator = EntropyEstimator()
        self.projector        = DomainProjector(input_dim=384, output_dim=target_dim)
        self.phase_encoder    = PhaseEncoder()

        # Modelo de embeddings
        if _ST_AVAILABLE:
            self._model = SentenceTransformer(model_name)
            self._model_available = True
        else:
            self._model = None
            self._model_available = False
            print("[NexusVectorizer] WARN: sentence-transformers no disponible. "
                  "Usando embeddings aleatorios deterministas.")

    def _embed(self, text: str) -> np.ndarray:
        """
        Generar embedding de 384 dimensiones desde texto.
        Fallback determinista si el modelo no está disponible.
        """
        if self._model_available:
            emb = self._model.encode(text, normalize_embeddings=True)
            return emb.astype(np.float32)
        else:
            # Fallback: hash determinista del texto
            seed = hash(text) % (2**31)
            rng = np.random.RandomState(seed)
            v = rng.randn(384).astype(np.float32)
            return v / (np.linalg.norm(v) + 1e-10)

    def vectorize(self, text: str) -> VectorizerOutput:
        """
        Pipeline completo: texto → VectorizerOutput.
        
        1. Embedding base (384-dim)
        2. Detección de señales léxicas por dominio
        3. Estimación de entropía
        4. Proyección al espacio de Hilbert (target_dim)
        5. Codificación de fase compleja
        6. Construcción de VectorizedNode por dominio activo
        """
        # 1. Embedding
        base_emb = self._embed(text)

        # 2. Detección de dominios
        activations = self.signal_detector.detect(text)
        n_actors = self.signal_detector.count_actors(text)

        # 3. Entropía
        global_entropy, per_node_entropy = self.entropy_estimator.estimate(
            text, activations
        )

        # 4-6. Construir nodos vectorizados
        active_nodes = []

        for activation in activations:
            if activation.activation_score < self.min_activation:
                # Incluir igualmente con score mínimo — todo dominio tiene
                # alguna presencia, aunque sea débil
                activation.activation_score = self.min_activation

            # Proyectar al espacio del dominio
            projected = self.projector.project(base_emb, activation.domain)

            # Modular entropía por nodo
            # Dominio más activado → menor entropía (más certeza sobre él)
            node_entropy = float(np.clip(
                per_node_entropy * (1.0 - activation.activation_score * 0.4),
                0.05, 0.95
            ))

            # Codificar fase compleja
            complex_vector = self.phase_encoder.encode(
                real_vector=projected,
                domain=activation.domain,
                entropy=node_entropy,
                activation_score=activation.activation_score
            )

            active_nodes.append(VectorizedNode(
                node_id=activation.domain.value,
                vector=complex_vector,
                entropy=node_entropy,
                activation=activation.activation_score,
                metadata={
                    "signals_found": activation.signals_found,
                    "signal_count": activation.signal_count,
                    "text_length": len(text.split()),
                    "n_actors": n_actors
                }
            ))

        # Ordenar por activación descendente
        active_nodes.sort(key=lambda x: x.activation, reverse=True)

        return VectorizerOutput(
            raw_input=text,
            active_nodes=active_nodes,
            domain_activations=activations,
            base_embedding=base_emb,
            n_actors_detected=n_actors,
            global_entropy=global_entropy
        )

    def top_pair(self, output: VectorizerOutput) -> Tuple[VectorizedNode, VectorizedNode]:
        """
        Retorna el par de nodos con mayor activación.
        Son los candidatos naturales para la primera colisión en Capa 1.
        """
        if len(output.active_nodes) < 2:
            raise ValueError("Se necesitan al menos 2 nodos para colisionar.")
        return output.active_nodes[0], output.active_nodes[1]

    def all_pairs(self, output: VectorizerOutput,
                  max_pairs: int = 3) -> List[Tuple[VectorizedNode, VectorizedNode]]:
        """
        Retorna hasta max_pairs pares ordenados por activación combinada.
        Para alimentar multi-colisión en Capa 1.
        """
        nodes = output.active_nodes
        pairs = []
        for i in range(len(nodes) - 1):
            pairs.append((nodes[i], nodes[i+1]))
            if len(pairs) >= max_pairs:
                break
        return pairs


# =============================================================================
# TEST — pipeline completo texto → vectores complejos
# =============================================================================

if __name__ == "__main__":

    vectorizer = NexusVectorizer(target_dim=64, min_activation=0.05)

    test_inputs = [
        "Carlos en la reunión dijo que todo está bien pero su tono cambió cuando mencioné la auditoría",
        "No sé por dónde empezar, todo está conectado, la empresa siempre vuelve al mismo patrón",
        "Siento que estoy en el borde de algo, no encuentro las palabras pero hay algo ahí"
    ]

    print("=" * 65)
    print("NEXUS VECTORIZADOR — OUTPUT NUMÉRICO")
    print("=" * 65)

    for text in test_inputs:
        print(f"\nINPUT: \"{text[:60]}...\"" if len(text) > 60 else f"\nINPUT: \"{text}\"")
        out = vectorizer.vectorize(text)

        print(f"  global_entropy   : {out.global_entropy:.4f}")
        print(f"  n_actors         : {out.n_actors_detected}")
        print(f"  activos (orden)  :")

        for node in out.active_nodes:
            print(f"    [{node.node_id:28s}]  "
                  f"activation={node.activation:.3f}  "
                  f"entropy={node.entropy:.3f}  "
                  f"vector_norm={np.linalg.norm(node.vector):.4f}  "
                  f"dtype={node.vector.dtype}")

        # Par principal para Capa 1
        n_a, n_b = vectorizer.top_pair(out)
        print(f"  → par para Capa1 : {n_a.node_id}  ↔  {n_b.node_id}")
