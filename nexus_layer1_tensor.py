#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
NEXUS — CAPA 1: TENSOR DE INTERFERENCIA
Pre-semántico. Opera por debajo del lenguaje natural.
El LLM nunca toca este módulo. Recibe su output numérico colapsado.

SUSTRATO MATEMÁTICO:
- Espacios de Hilbert (dim=N, complejo)
- Operadores de proyección hermíticos
- Término de interferencia Busemeyer (Int)
- Umbral de emergencia CHSH (> 2.0)
- Disipación Lindblad simplificada → colapso a probabilidad clásica

ARQUITECTURA:
    INPUT vectorizado
        ↓
    QuantumStateVector         — |ψ⟩ en ℂᴺ, norma L2
        ↓
    InterferenceTensor         — colisión de dos estados, calcula Int
        ↓
    EmergenceDetector          — test CHSH, detona rama generativa si > 2.0
        ↓
    LindbladCollapse           — disipación → ρ_ss (probabilidad clásica)
        ↓
    CollapseResult             — números listos para Capa 2 (LLM)

NOTA: Ninguna función retorna strings narrativos.
      Todo output es numérico o estructuras de datos tipadas.
================================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum


# =============================================================================
# TIPOS DE INTERFERENCIA
# =============================================================================

class InterferenceType(Enum):
    CONSTRUCTIVE  =  1   # Int > 0  — amplificación
    DESTRUCTIVE   = -1   # Int < 0  — cancelación
    NULL          =  0   # Int = 0  — nodos independientes, sin colisión real


class CollapseStatus(Enum):
    CLASSICAL     = "classical"    # ruta estable, probabilidad limpia
    EMERGENT      = "emergent"     # CHSH > 2.0, rama generativa nueva
    OSCILLATING   = "oscillating"  # disipación insuficiente, inestable


# =============================================================================
# ESTRUCTURAS DE DATOS — tipadas, sin semántica
# =============================================================================

@dataclass
class QuantumStateVector:
    """
    |ψ⟩ ∈ ℂᴺ — vector de estado de un nodo en superposición.
    norm(ψ) = 1 bajo L2.
    amplitudes: array complejo de dimensión N.
    node_id: identificador externo (no semántico aquí).
    """
    node_id: str
    amplitudes: np.ndarray          # dtype=complex128, shape=(N,)
    entropy: float = 0.5            # H ∈ [0,1] — incertidumbre del nodo

    def __post_init__(self):
        norm = np.linalg.norm(self.amplitudes)
        if norm < 1e-10:
            raise ValueError(f"[{self.node_id}] amplitudes degeneradas — norma ≈ 0")
        self.amplitudes = self.amplitudes / norm

    def projection_matrix(self) -> np.ndarray:
        """P = |ψ⟩⟨ψ| — operador de proyección hermítico"""
        return np.outer(self.amplitudes, self.amplitudes.conj())

    def inner_product(self, other: "QuantumStateVector") -> complex:
        """⟨ψ_a|ψ_b⟩ — producto interno en ℂᴺ"""
        return np.dot(self.amplitudes.conj(), other.amplitudes)


@dataclass
class InterferenceResult:
    """
    Output crudo de la colisión entre dos estados.
    No contiene lenguaje. Solo métricas.
    """
    node_a: str
    node_b: str
    p_a: float                      # P(G|A solo)
    p_b: float                      # P(G|B solo)
    p_classical: float              # suma clásica ponderada (sin interferencia)
    interference_term: float        # Int = 2·Re(⟨ψ_a|P_G|ψ_b⟩·⟨ψ_b|ψ_a⟩)
    p_quantum: float                # P_quantum = p_classical + Int
    interference_type: InterferenceType
    phase_angle: float              # ángulo de fase en radianes
    chsh_value: float               # correlación CHSH para test de emergencia


@dataclass
class EmergenceEvent:
    """
    Generado cuando CHSH > 2.0.
    Señal de que la colisión produce algo fuera del grafo existente.
    """
    source_nodes: List[str]
    chsh_value: float               # debe ser > 2.0
    emergence_probability: float    # P de que la rama sea real
    tensor_signature: np.ndarray    # huella matemática del evento


@dataclass
class CollapseResult:
    """
    Output final de Capa 1 — listo para Capa 2 (LLM).
    Solo números. El LLM interpreta, no calcula.
    """
    # Identidad de la colisión
    collision_id: str
    nodes_collided: List[str]

    # Probabilidades finales (clásicas, post-disipación)
    route_probabilities: Dict[str, float]   # {node_id: P_collapsed}

    # Tipo de resultado
    status: CollapseStatus
    interference_type: InterferenceType

    # Métricas para Capa 2
    interference_magnitude: float           # |Int| — intensidad de la colisión
    emergence_detected: bool
    emergence_event: Optional[EmergenceEvent]

    # Atractor — nodo con mayor probabilidad post-colapso
    attractor_node: str
    attractor_probability: float

    # Factor multi-actor Yukalov (± 0.25 si hay > 1 actor)
    yukalov_correction: float               # 0.0 si single-actor

    # Confianza general del resultado
    confidence: float


# =============================================================================
# COMPONENTE 1 — TENSOR DE INTERFERENCIA
# =============================================================================

class InterferenceTensor:
    """
    Calcula la colisión entre dos estados cuánticos.

    Ecuación central (Busemeyer):
        P_quantum(G) = P_classical(G) + Int
        Int = 2·Re[⟨ψ_a|P_G|ψ_b⟩ · ⟨ψ_b|ψ_a⟩]

    donde P_G es el operador de proyección sobre el subespacio de decisión G.

    El signo de Int determina el tipo de interferencia:
        Int > 0 → constructiva (amplificación)
        Int < 0 → destructiva (cancelación)
        Int ≈ 0 → nula (independencia)
    """

    def __init__(self, dim: int = 64, null_threshold: float = 0.05):
        """
        dim: dimensión del espacio de Hilbert
        null_threshold: |Int| < threshold → interferencia nula
        """
        self.dim = dim
        self.null_threshold = null_threshold

    def _build_decision_projector(self, decision_vector: np.ndarray) -> np.ndarray:
        """
        P_G = |g⟩⟨g| — proyector sobre subespacio de decisión G.
        decision_vector: vector normalizado que representa la dirección de decisión.
        """
        g = decision_vector / (np.linalg.norm(decision_vector) + 1e-10)
        return np.outer(g, g.conj())

    def _compute_chsh(self, state_a: QuantumStateVector,
                      state_b: QuantumStateVector) -> float:
        """
        Aproximación del valor CHSH para dos estados.
        CHSH clásico ≤ 2.0. Cuántico puede alcanzar 2√2 ≈ 2.828.
        Cuando CHSH > 2.0 → emergencia.

        Usamos la correlación cruzada normalizada como proxy:
            E(A,B) = |⟨ψ_a|ψ_b⟩|²  (producto interno al cuadrado)
        CHSH ≈ 4 · E(A,B) — estimación para sistema de 2 nodos
        """
        inner = abs(state_a.inner_product(state_b)) ** 2
        # Escalar a rango CHSH [0, 2√2]
        chsh = 4.0 * inner
        return float(np.clip(chsh, 0.0, 2.0 * np.sqrt(2)))

    def collide(self,
                state_a: QuantumStateVector,
                state_b: QuantumStateVector,
                decision_vector: Optional[np.ndarray] = None,
                weight_a: float = 0.5,
                weight_b: float = 0.5) -> InterferenceResult:
        """
        Colisión entre dos estados cuánticos.

        state_a, state_b: los dos nodos en superposición
        decision_vector: dirección del subespacio de decisión G
                         (si None → se usa la suma normalizada de ambos estados)
        weight_a, weight_b: pesos relativos de cada estado (deben sumar 1.0)

        Retorna InterferenceResult con todas las métricas.
        """
        # Normalizar pesos
        total_w = weight_a + weight_b
        wa = weight_a / total_w
        wb = weight_b / total_w

        # Construir proyector de decisión
        if decision_vector is None:
            # Dirección de decisión: combinación lineal de ambos estados
            d = wa * state_a.amplitudes + wb * state_b.amplitudes
        else:
            d = decision_vector.astype(complex)

        P_G = self._build_decision_projector(d)

        # Probabilidades individuales
        # P(G|A) = ⟨ψ_a|P_G|ψ_a⟩ (real)
        p_a = float(np.real(state_a.amplitudes.conj() @ P_G @ state_a.amplitudes))
        p_b = float(np.real(state_b.amplitudes.conj() @ P_G @ state_b.amplitudes))

        # Probabilidad clásica ponderada (sin interferencia)
        p_classical = wa * p_a + wb * p_b

        # Término de interferencia Busemeyer:
        # Int = 2·Re[⟨ψ_a|P_G|ψ_b⟩ · ⟨ψ_b|ψ_a⟩]
        cross_term = (state_a.amplitudes.conj() @ P_G @ state_b.amplitudes) * \
                     (state_b.amplitudes.conj() @ state_a.amplitudes)
        interference_term = float(2.0 * np.real(cross_term))

        # Probabilidad cuántica
        p_quantum = float(np.clip(p_classical + interference_term, 0.0, 1.0))

        # Tipo de interferencia
        if abs(interference_term) < self.null_threshold:
            itype = InterferenceType.NULL
        elif interference_term > 0:
            itype = InterferenceType.CONSTRUCTIVE
        else:
            itype = InterferenceType.DESTRUCTIVE

        # Ángulo de fase
        inner = state_a.inner_product(state_b)
        phase_angle = float(np.angle(inner))

        # CHSH
        chsh_value = self._compute_chsh(state_a, state_b)

        return InterferenceResult(
            node_a=state_a.node_id,
            node_b=state_b.node_id,
            p_a=p_a,
            p_b=p_b,
            p_classical=p_classical,
            interference_term=interference_term,
            p_quantum=p_quantum,
            interference_type=itype,
            phase_angle=phase_angle,
            chsh_value=chsh_value
        )


# =============================================================================
# COMPONENTE 2 — DETECTOR DE EMERGENCIA (CHSH)
# =============================================================================

class EmergenceDetector:
    """
    Test CHSH: cuando la correlación entre dos tensores supera 2.0,
    el motor genera una rama nueva — no promedia.

    Umbral clásico: ≤ 2.0
    Umbral Tsirelson (cuántico máximo): 2√2 ≈ 2.828

    Cuando CHSH > 2.0:
        - No promediar pesos
        - Instanciar EmergenceEvent
        - Capa 2 recibe señal de "rama generativa" — output que no
          existía en ninguno de los dos nodos por separado
    """

    CLASSICAL_LIMIT = 2.0
    TSIRELSON_LIMIT = 2.0 * np.sqrt(2)   # ≈ 2.828

    def __init__(self, chsh_threshold: float = 2.0):
        self.chsh_threshold = chsh_threshold

    def test(self, result: InterferenceResult,
             state_a: QuantumStateVector,
             state_b: QuantumStateVector) -> Optional[EmergenceEvent]:
        """
        Evalúa si la colisión supera el umbral de emergencia.
        Retorna EmergenceEvent si CHSH > threshold, None si no.
        """
        if result.chsh_value <= self.chsh_threshold:
            return None

        # Probabilidad de emergencia — escalar linealmente entre umbral y Tsirelson
        emergence_range = self.TSIRELSON_LIMIT - self.chsh_threshold
        excess = result.chsh_value - self.chsh_threshold
        emergence_probability = float(np.clip(excess / emergence_range, 0.0, 1.0))

        # Huella tensorial del evento emergente
        # = producto tensorial de los dos estados
        tensor_signature = np.outer(state_a.amplitudes, state_b.amplitudes)
        # Comprimir a vector de norma (tomar diagonal real como firma)
        signature_compressed = np.abs(np.diag(
            tensor_signature[:min(16, tensor_signature.shape[0]),
                             :min(16, tensor_signature.shape[1])]
        ))

        return EmergenceEvent(
            source_nodes=[state_a.node_id, state_b.node_id],
            chsh_value=result.chsh_value,
            emergence_probability=emergence_probability,
            tensor_signature=signature_compressed
        )


# =============================================================================
# COMPONENTE 3 — DISIPACIÓN LINDBLAD (colapso a probabilidad clásica)
# =============================================================================

class LindbladCollapse:
    """
    Versión simplificada de la ecuación maestra GKSL.

    dρ/dt = -i[H, ρ] + Σ_ij γ_ij (L_ij ρ L_ij† - ½{L_ij†L_ij, ρ})

    Para Nexus: el Hamiltoniano H modela la oscilación entre rutas.
    Los operadores Lindblad L_ij fuerzan la disipación hacia atractores.

    Estado estacionario: dρ_ss/dt = 0
    → Los elementos off-diagonal (interferencia) decaen a 0
    → La diagonal da probabilidades clásicas finales

    Implementación: iteración discreta hasta convergencia.
    """

    def __init__(self,
                 decay_rate: float = 0.3,
                 max_iterations: int = 100,
                 convergence_tol: float = 1e-6):
        """
        decay_rate: γ — tasa de disipación de elementos off-diagonal
        max_iterations: límite de iteraciones antes de declarar oscillating
        convergence_tol: tolerancia para estado estacionario
        """
        self.decay_rate = decay_rate
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol

    def collapse(self,
                 state_a: QuantumStateVector,
                 state_b: QuantumStateVector,
                 interference_result: InterferenceResult,
                 attractor_bias: Optional[Dict[str, float]] = None) -> Tuple[Dict[str, float], CollapseStatus]:
        """
        Colapsa la superposición de dos estados hacia probabilidades clásicas.

        attractor_bias: pesos externos que sesgan hacia ciertos nodos
                        (para implementar punto_de_atractor)

        Retorna:
            route_probabilities: {node_id: P_final}
            status: CollapseStatus
        """
        # Matriz de densidad inicial (2x2 para dos nodos)
        # ρ = |c_a|² |a⟩⟨a| + |c_b|² |b⟩⟨b| + c_a c_b* |a⟩⟨b| + c_b c_a* |b⟩⟨a|
        # Representamos en base {a, b}
        p_a = interference_result.p_a
        p_b = interference_result.p_b

        # Amplitudes complejas relativas
        inner = state_a.inner_product(state_b)
        rho_off = np.sqrt(p_a * p_b) * inner  # elemento off-diagonal

        # Matriz de densidad 2x2
        rho = np.array([
            [p_a,      rho_off],
            [rho_off.conj(), p_b]
        ], dtype=complex)

        # Aplicar sesgo de atractor si existe
        if attractor_bias:
            a_bias = attractor_bias.get(state_a.node_id, 1.0)
            b_bias = attractor_bias.get(state_b.node_id, 1.0)
            rho[0, 0] *= a_bias
            rho[1, 1] *= b_bias

        # Iteración Lindblad simplificada
        # En cada paso: off-diagonals decaen por γ, diagonal se preserva
        prev_diag = np.array([rho[0, 0].real, rho[1, 1].real])
        status = CollapseStatus.OSCILLATING

        for _ in range(self.max_iterations):
            # Disipación: suprimir off-diagonals
            rho[0, 1] *= (1.0 - self.decay_rate)
            rho[1, 0] *= (1.0 - self.decay_rate)

            # Re-normalizar diagonal
            diag_sum = rho[0, 0].real + rho[1, 1].real
            if diag_sum > 1e-10:
                rho[0, 0] /= diag_sum
                rho[1, 1] /= diag_sum

            curr_diag = np.array([rho[0, 0].real, rho[1, 1].real])

            # Test de convergencia
            delta = np.linalg.norm(curr_diag - prev_diag)
            if delta < self.convergence_tol:
                status = CollapseStatus.CLASSICAL
                break
            prev_diag = curr_diag

        # Probabilidades finales
        p_a_final = float(np.clip(rho[0, 0].real, 0.0, 1.0))
        p_b_final = float(np.clip(rho[1, 1].real, 0.0, 1.0))

        # Normalizar
        total = p_a_final + p_b_final
        if total > 1e-10:
            p_a_final /= total
            p_b_final /= total

        route_probabilities = {
            state_a.node_id: p_a_final,
            state_b.node_id: p_b_final
        }

        return route_probabilities, status


# =============================================================================
# MOTOR CAPA 1 — ORQUESTADOR
# =============================================================================

class NexusLayer1:
    """
    Orquestador de Capa 1.
    Recibe vectores, ejecuta el pipeline completo,
    entrega CollapseResult a Capa 2.

    Pipeline:
        QuantumStateVector × 2
            → InterferenceTensor.collide()
            → EmergenceDetector.test()
            → LindbladCollapse.collapse()
            → CollapseResult
    """

    def __init__(self,
                 dim: int = 64,
                 null_threshold: float = 0.05,
                 chsh_threshold: float = 2.0,
                 decay_rate: float = 0.3,
                 max_collisions_per_cycle: int = 3):
        """
        dim: dimensión del espacio de Hilbert
        null_threshold: |Int| mínimo para interferencia no-nula
        chsh_threshold: umbral de emergencia CHSH
        decay_rate: velocidad de disipación Lindblad
        max_collisions_per_cycle: límite de colisiones simultáneas (control de explosión)
        """
        self.dim = dim
        self.max_collisions_per_cycle = max_collisions_per_cycle

        self.tensor     = InterferenceTensor(dim=dim, null_threshold=null_threshold)
        self.emergence  = EmergenceDetector(chsh_threshold=chsh_threshold)
        self.lindblad   = LindbladCollapse(decay_rate=decay_rate)

        # Registro de colisiones del ciclo actual
        self._cycle_collisions: int = 0
        self._collision_counter: int = 0

    def _make_state(self, node_id: str,
                    vector: np.ndarray,
                    entropy: float = 0.5) -> QuantumStateVector:
        """
        Construir QuantumStateVector desde vector real o complejo.
        Si el vector es real → embedder a complejo con fase nula.
        """
        if not np.iscomplexobj(vector):
            # Convertir a complejo — fase nula en primera instancia
            # La fase se actualizará con contexto acumulado
            amplitudes = vector.astype(complex)
        else:
            amplitudes = vector.copy()

        # Asegurar dimensión correcta
        if len(amplitudes) < self.dim:
            pad = np.zeros(self.dim - len(amplitudes), dtype=complex)
            amplitudes = np.concatenate([amplitudes, pad])
        elif len(amplitudes) > self.dim:
            amplitudes = amplitudes[:self.dim]

        return QuantumStateVector(node_id=node_id,
                                  amplitudes=amplitudes,
                                  entropy=entropy)

    def _apply_yukalov_correction(self,
                                  route_probs: Dict[str, float],
                                  n_actors: int) -> Tuple[Dict[str, float], float]:
        """
        Corrección Yukalov (2015): en sistemas multi-actor,
        el factor de interferencia oscila hacia q ≈ ±0.25.
        Aplicar solo cuando n_actors > 1.
        """
        if n_actors <= 1:
            return route_probs, 0.0

        correction = 0.25  # Quarter Law de Yukalov
        corrected = {}
        for node_id, p in route_probs.items():
            # Añadir incertidumbre ±0.25 normalizada
            p_corrected = float(np.clip(p + correction * (2 * np.random.random() - 1), 0.0, 1.0))
            corrected[node_id] = p_corrected

        # Re-normalizar
        total = sum(corrected.values())
        if total > 1e-10:
            corrected = {k: v / total for k, v in corrected.items()}

        return corrected, correction

    def process_collision(self,
                          node_id_a: str,
                          vector_a: np.ndarray,
                          node_id_b: str,
                          vector_b: np.ndarray,
                          entropy_a: float = 0.5,
                          entropy_b: float = 0.5,
                          weight_a: float = 0.5,
                          weight_b: float = 0.5,
                          decision_vector: Optional[np.ndarray] = None,
                          attractor_bias: Optional[Dict[str, float]] = None,
                          n_actors: int = 1) -> CollapseResult:
        """
        Pipeline completo de colisión entre dos nodos.

        Parámetros:
            node_id_a/b: identificadores externos de los nodos
            vector_a/b:  embeddings (real o complejo, shape=(dim,))
            entropy_a/b: incertidumbre de cada nodo [0,1]
            weight_a/b:  peso relativo de cada nodo en la colisión
            decision_vector: dirección de decisión (None = auto)
            attractor_bias: {node_id: factor} para sesgar hacia atractores conocidos
            n_actors: número de actores en juego (activa corrección Yukalov si > 1)

        Retorna CollapseResult — input numérico para Capa 2.
        """
        # Control de límite de colisiones por ciclo
        if self._cycle_collisions >= self.max_collisions_per_cycle:
            raise RuntimeError(
                f"Límite de colisiones por ciclo alcanzado ({self.max_collisions_per_cycle}). "
                f"Llamar reset_cycle() antes de continuar."
            )
        self._cycle_collisions += 1
        self._collision_counter += 1
        collision_id = f"COL_{self._collision_counter:04d}"

        # 1. Construir estados cuánticos
        state_a = self._make_state(node_id_a, vector_a, entropy_a)
        state_b = self._make_state(node_id_b, vector_b, entropy_b)

        # 2. Calcular interferencia
        ir = self.tensor.collide(state_a, state_b,
                                 decision_vector=decision_vector,
                                 weight_a=weight_a,
                                 weight_b=weight_b)

        # 3. Test de emergencia CHSH
        emergence_event = self.emergence.test(ir, state_a, state_b)
        emergence_detected = emergence_event is not None

        # 4. Colapso Lindblad
        route_probs, collapse_status = self.lindblad.collapse(
            state_a, state_b, ir, attractor_bias=attractor_bias
        )

        # Si hay emergencia → el status se eleva
        if emergence_detected:
            collapse_status = CollapseStatus.EMERGENT

        # 5. Corrección Yukalov (multi-actor)
        route_probs, yukalov_corr = self._apply_yukalov_correction(route_probs, n_actors)

        # 6. Identificar atractor (nodo con mayor P)
        attractor_node = max(route_probs, key=route_probs.get)
        attractor_prob = route_probs[attractor_node]

        # 7. Confianza general
        # Alta si: interferencia clara + colapso limpio + CHSH < límite
        confidence = float(np.clip(
            (1.0 - abs(ir.interference_term)) * 0.4 +
            (1.0 if collapse_status == CollapseStatus.CLASSICAL else 0.5) * 0.4 +
            (1.0 - ir.chsh_value / (2.0 * np.sqrt(2))) * 0.2,
            0.0, 1.0
        ))

        return CollapseResult(
            collision_id=collision_id,
            nodes_collided=[node_id_a, node_id_b],
            route_probabilities=route_probs,
            status=collapse_status,
            interference_type=ir.interference_type,
            interference_magnitude=abs(ir.interference_term),
            emergence_detected=emergence_detected,
            emergence_event=emergence_event,
            attractor_node=attractor_node,
            attractor_probability=attractor_prob,
            yukalov_correction=yukalov_corr,
            confidence=confidence
        )

    def process_multi_collision(self,
                                nodes: List[Tuple[str, np.ndarray, float]],
                                weights: Optional[List[float]] = None,
                                attractor_bias: Optional[Dict[str, float]] = None,
                                n_actors: int = 1) -> List[CollapseResult]:
        """
        Colisiones generacionales: los resultados del ciclo N
        alimentan las colisiones del ciclo N+1.

        nodes: lista de (node_id, vector, entropy)
        weights: pesos de cada nodo (None = uniforme)
        Máximo max_collisions_per_cycle colisiones por llamada.

        Retorna lista de CollapseResult en orden generacional.
        """
        if len(nodes) < 2:
            raise ValueError("Se necesitan al menos 2 nodos para colisionar.")

        if weights is None:
            weights = [1.0 / len(nodes)] * len(nodes)

        results = []
        self.reset_cycle()

        # Colisiones por pares — orden por peso descendente
        sorted_nodes = sorted(zip(weights, nodes), key=lambda x: x[0], reverse=True)
        pairs = []
        for i in range(len(sorted_nodes) - 1):
            pairs.append((sorted_nodes[i], sorted_nodes[i + 1]))
            if len(pairs) >= self.max_collisions_per_cycle:
                break

        for (wa, node_a), (wb, node_b) in pairs:
            id_a, vec_a, ent_a = node_a
            id_b, vec_b, ent_b = node_b

            result = self.process_collision(
                node_id_a=id_a, vector_a=vec_a, entropy_a=ent_a,
                node_id_b=id_b, vector_b=vec_b, entropy_b=ent_b,
                weight_a=wa, weight_b=wb,
                attractor_bias=attractor_bias,
                n_actors=n_actors
            )
            results.append(result)

        return results

    def reset_cycle(self):
        """Reiniciar contador de colisiones del ciclo actual."""
        self._cycle_collisions = 0

    def summarize(self, results: List[CollapseResult]) -> Dict:
        """
        Resumen numérico de múltiples CollapseResults.
        Output directo para Capa 2 — sin narrativa.
        """
        if not results:
            return {}

        emergences = [r for r in results if r.emergence_detected]
        constructive = [r for r in results if r.interference_type == InterferenceType.CONSTRUCTIVE]
        destructive  = [r for r in results if r.interference_type == InterferenceType.DESTRUCTIVE]

        # Atractor global — el nodo que aparece más como atractor
        all_attractors = [r.attractor_node for r in results]
        from collections import Counter
        attractor_freq = Counter(all_attractors)
        global_attractor = attractor_freq.most_common(1)[0][0]

        # Probabilidad agregada por nodo
        agg_probs: Dict[str, List[float]] = {}
        for r in results:
            for node_id, p in r.route_probabilities.items():
                agg_probs.setdefault(node_id, []).append(p)
        mean_probs = {k: float(np.mean(v)) for k, v in agg_probs.items()}

        return {
            "total_collisions": len(results),
            "emergences_detected": len(emergences),
            "constructive_count": len(constructive),
            "destructive_count": len(destructive),
            "global_attractor": global_attractor,
            "mean_route_probabilities": mean_probs,
            "mean_confidence": float(np.mean([r.confidence for r in results])),
            "max_chsh": float(max(
                (r.emergence_event.chsh_value for r in emergences), default=0.0
            )),
            "collision_ids": [r.collision_id for r in results]
        }


# =============================================================================
# TEST MÍNIMO — verifica que el pipeline corre sin errores
# =============================================================================

if __name__ == "__main__":

    np.random.seed(42)
    DIM = 64

    engine = NexusLayer1(
        dim=DIM,
        null_threshold=0.05,
        chsh_threshold=2.0,
        decay_rate=0.3,
        max_collisions_per_cycle=3
    )

    # Simular dos nodos con vectores de embedding reales
    # (en producción estos vienen de un modelo de embeddings)
    vec_tension_activa      = np.random.randn(DIM)
    vec_punto_atractor      = np.random.randn(DIM)
    vec_interferencia_actor = np.random.randn(DIM)
    vec_estado_operador     = np.random.randn(DIM)

    # Colisión 1: tension_activa ↔ punto_atractor
    r1 = engine.process_collision(
        node_id_a="tension_activa",       vector_a=vec_tension_activa,       entropy_a=0.7,
        node_id_b="punto_atractor",       vector_b=vec_punto_atractor,       entropy_b=0.4,
        weight_a=0.6, weight_b=0.4
    )

    # Colisión 2: interferencia_actor_sistema ↔ estado_cognitivo_operador
    r2 = engine.process_collision(
        node_id_a="interferencia_actor",  vector_a=vec_interferencia_actor,  entropy_a=0.6,
        node_id_b="estado_operador",      vector_b=vec_estado_operador,       entropy_b=0.3,
        weight_a=0.5, weight_b=0.5,
        n_actors=2   # multi-actor → corrección Yukalov activa
    )

    # Colisión 3: tension_activa ↔ estado_operador
    r3 = engine.process_collision(
        node_id_a="tension_activa",       vector_a=vec_tension_activa,       entropy_a=0.7,
        node_id_b="estado_operador",      vector_b=vec_estado_operador,       entropy_b=0.3,
        weight_a=0.4, weight_b=0.6
    )

    summary = engine.summarize([r1, r2, r3])

    # Output — solo números, sin narrativa
    print("=" * 60)
    print("NEXUS CAPA 1 — OUTPUT NUMÉRICO")
    print("=" * 60)

    for r in [r1, r2, r3]:
        print(f"\n[{r.collision_id}] {r.nodes_collided[0]} ↔ {r.nodes_collided[1]}")
        print(f"  interferencia : {r.interference_type.name}  |Int|={r.interference_magnitude:.4f}")
        print(f"  CHSH          : {r.emergence_event.chsh_value if r.emergence_event else 'N/A'}")
        print(f"  emergencia    : {r.emergence_detected}")
        print(f"  status        : {r.status.value}")
        print(f"  atractor      : {r.attractor_node}  P={r.attractor_probability:.4f}")
        print(f"  yukalov_corr  : {r.yukalov_correction:.4f}")
        print(f"  confianza     : {r.confidence:.4f}")
        print(f"  rutas         : { {k: f'{v:.4f}' for k,v in r.route_probabilities.items()} }")

    print("\n" + "=" * 60)
    print("RESUMEN DEL CICLO")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
