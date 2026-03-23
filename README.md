# DQ-MPCC vs MPCC — Comparative Framework

## Tabla de contenido

1. [Diagnóstico: falla crítica en el tuning del DQ-MPCC](#1-diagnóstico-falla-crítica-en-el-tuning-del-dq-mpcc)
2. [Frontera de Pareto: Velocidad vs Precisión](#2-frontera-de-pareto-velocidad-vs-precisión)
3. [Cómo ejecutar los experimentos](#3-cómo-ejecutar-los-experimentos)

---

## 1. Diagnóstico: falla crítica en el tuning del DQ-MPCC

### 1.1 Contexto del problema

El framework de bilevel tuning utiliza Optuna para optimizar los 17 pesos de costo
del controlador DQ-MPCC. En cada trial, Optuna propone un vector de pesos, el
simulador `DQ_MPCC_simulation_tuner.py` ejecuta la simulación completa del drone
a lo largo de una trayectoria Lissajous de 100 m, y devuelve el costo
J_MPCC como función objetivo.

El problema observado: **el tuner producía resultados inutilizables** (J ≥ 9800,
RMSE_c ≥ 68 m, path_completed ≈ 0%) incluso con pesos que manualmente
funcionaban a la perfección.

### 1.2 Causa raíz 1 — Pesos por defecto erróneos (copy-paste de MPCC)

**Qué estaba mal:** Los pesos `DEFAULT_*` en `dq_mpcc_controller.py` eran una
copia directa de los pesos del MPCC clásico (los mismos valores del trial que
dio J=32.71 en el baseline MPCC). Estos pesos producían `status = 4` (solver
failure) a partir de k≈434 y el drone se atascaba en θ≈19.41 m indefinidamente.

```python
# ❌ ANTES — límites MPCC copy-pasted (no funcionan para DQ-MPCC)
DEFAULT_T_MAX    = 5 * G     # solo 49 N de empuje (insuficiente)
DEFAULT_TAUX_MAX = 0.1       # torque máximo muy restrictivo
DEFAULT_TAUY_MAX = 0.1
DEFAULT_TAUZ_MAX = 0.1
```

```python
# ✅ DESPUÉS — límites calibrados para el modelo DQ-MPCC
DEFAULT_T_MAX    = 10 * G    # 98.1 N (suficiente para maniobras agresivas)
DEFAULT_TAUX_MAX = 0.5       # torques con margen adecuado
DEFAULT_TAUY_MAX = 0.5
DEFAULT_TAUZ_MAX = 0.5
```

**Por qué importa:** El modelo DQ-MPCC tiene 15 estados (dual quaternion + velocidades
+ θ) vs 14 del MPCC clásico (posición + quaternion + velocidades + θ). La
dinámica en formulación de cuerpo rígido con cuaterniones duales requiere
**más autoridad de control** (torques) y **más empuje** porque la representación
de rotación acoplada amplifica las no-linealidades. Los límites restrictivos
del MPCC clásico hacían que el solver QP fuera infeasible en los giros
de la Lissajous.

### 1.3 Causa raíz 2 — Sin reset del solver entre trials (BUG CRÍTICO)

**Qué estaba mal:** El solver de Acados mantiene estado interno entre llamadas:
variables primales (x, u), variables duales (multiplicadores de Lagrange λ),
y variables de slack (s_l, s_u). Cuando un trial divergía
(ej: Q_ec=30 → NaN), estos estados internos se **contaminaban**.
Todos los trials posteriores heredaban multiplicadores corruptos y fallaban
inmediatamente con path=0%, RMSE≈1900, incluso usando los pesos que antes
funcionaban perfectamente.

```
Trial 1: Q_ec=10 → path=100%, J=64    ← funciona perfecto
Trial 2: Q_ec=30 → NaN diverge         ← contamina el solver
Trial 3: Q_ec=10 → path=0%, RMSE=1900  ← FALLA (mismos pesos que Trial 1!)
Trial 4: Q_ec=5  → path=0%, RMSE=1900  ← FALLA
...todos los siguientes → FALLAN
```

**Causa técnica:** El solver de Acados usa un **warm-start** que inicializa la
solución del QP en cada llamada a `solver.solve()`. Si el warm-start contiene
NaN o valores extremos, el QP solver (HPIPM) no converge y retorna
`status ≠ 0`. El código original solo reseteaba `x` y `u`, pero no los
multiplicadores:

```python
# ❌ ANTES — warm-start incompleto
for stage in range(N_prediction + 1):
    solver.set(stage, "x", x0)
for stage in range(N_prediction):
    solver.set(stage, "u", np.zeros(nu))
# ⚠️ λ, sl, su quedan con valores del trial anterior (posiblemente NaN)
```

```python
# ✅ DESPUÉS — reset completo (previene contaminación entre trials)
for stage in range(N_prediction + 1):
    solver.set(stage, "x", x0)
    solver.set(stage, "p", p_vec)
for stage in range(N_prediction):
    solver.set(stage, "u", np.zeros(nu))
# Reset variables duales
try:
    for stage in range(N_prediction + 1):
        solver.set(stage, "lam", np.zeros(nx))
    for stage in range(N_prediction):
        solver.set(stage, "sl", np.zeros(1))
        solver.set(stage, "su", np.zeros(1))
except Exception:
    pass
```

### 1.4 Causa raíz 3 — Rangos de búsqueda demasiado amplios

**Qué estaba mal:** Optuna exploraba rangos donde el solver divergía (NaN):

| Parámetro | Rango anterior | Problema | Rango corregido |
|-----------|---------------|----------|-----------------|
| Q_ec      | [1, 50]       | Q_ec>30 → NaN | [1, 30] |
| Q_el      | [0.1, 50]     | divergencia | [0.5, 30] |
| Q_s       | [0.5, 10]     | Q_s>3 → θ se estanca (path=0%) | [0.1, 3] |
| U_tau     | [10, 800]     | torques inestables | [20, 500] |
| Q_omega   | [0.01, 10]    | sobre-amortiguamiento | [0.01, 2] |
| Q_rot     | [0.1, 50]     | divergencia rotacional | [0.1, 20] |

### 1.5 Protecciones adicionales

Se agregaron 4 mecanismos de defensa:

1. **NaN/Inf guard:** Si el estado contiene NaN o Inf, se aborta el trial inmediatamente.
2. **Contador de fallos consecutivos:** Si el solver falla 50 veces seguidas, se aborta (el drone está atascado).
3. **Supresión de salida C:** Los warnings del QP solver (HPIPM) se silencian a nivel de `os.dup2()` para mantener logs limpios.
4. **NaN guard en el objetivo de Optuna:** Si J=NaN, se retorna J=10⁶ para penalizar el trial.

### 1.6 Resultado

Después de las correcciones, 50 trials de Optuna se ejecutaron exitosamente:

| Métrica | Antes del fix | Después del fix (best trial #34) |
|---------|--------------|----------------------------------|
| J_MPCC  | ≥ 9839       | **12.27** |
| RMSE_contorno | ≥ 68 m | **0.227 m** |
| RMSE_lag      | N/A    | **0.194 m** |
| Path completed | 0%    | **99.98%** |
| Trials exitosos | ~1/50 | ~35/50 |

---

## 2. Frontera de Pareto: Velocidad vs Precisión

### 2.1 ¿Por qué RMSE solo no es suficiente?

En un controlador MPCC (Model Predictive Contouring Control), la velocidad de
progreso a lo largo de la trayectoria θ̇ es una **variable de decisión**
del optimizador, no una referencia impuesta. El controlador elige cuánto
avanzar en cada paso, negociando entre:

- **Ir rápido** → completar el circuito en menos tiempo, pero con mayor error
  de seguimiento (hay menos tiempo para corregir desviaciones).
- **Ir lento** → menor error porque el drone tiene más tiempo en cada punto
  para acercarse a la trayectoria deseada.

Comparar solo RMSE entre dos controladores es **injusto**: un controlador
puede tener RMSE bajo simplemente porque avanza más lento. Ejemplo:

```
Controlador A: RMSE = 0.15 m,  t_lap = 18 s   ← preciso pero lento
Controlador B: RMSE = 0.25 m,  t_lap = 9 s    ← menos preciso pero 2× más rápido
```

¿Cuál es mejor? Depende del contexto de uso. La respuesta correcta es
**comparar ambos en el mismo gráfico**.

### 2.2 El gráfico de Pareto: velocidad-precisión

La frontera de Pareto evalúa **simultáneamente** la precisión de seguimiento
(RMSE_pos) y la velocidad de completación (t_lap). Cada controlador se
evalúa bajo las **mismas condiciones experimentales**:

- **Misma trayectoria:** Lissajous 3D de S_max = 100 m.
- **Misma restricción de velocidad:** v_{θ,max} ∈ {8, 12, 15} m/s.
- **Mismas condiciones iniciales:** N corridas Monte Carlo con perturbaciones
  aleatorias idénticas (misma semilla, σ_p = 0.05 m, σ_q = 0.05 rad).
- **Mismo presupuesto de tiempo:** T_final = 30 s.

Para cada combinación (controlador, v_{θ,max}), el experimento registra:

| Métrica | Símbolo | Cómo se calcula |
|---------|---------|-----------------|
| Tiempo de vuelta | t_lap | k_final × Δt, donde k_final es el paso en que θ ≥ S_max |
| Velocidad virtual media | v̄_θ | θ_final / t_lap |
| Error de posición | RMSE_pos | √(1/K · Σ ‖p_k − γ(θ_k)‖²) |
| Error de orientación | RMSE_ori | √(1/K · Σ ‖Log(q_d⁻¹ ⊗ q_k)‖²) |

### 2.3 Estructura del gráfico

```
RMSE_pos [m]  (mediana ± IQR sobre N runs)
    │
    │  ●─┤├── MPCC  v=15 m/s     ← punto más alto (menos preciso, más rápido)
    │    ●─┤├── MPCC  v=12 m/s
    │      ●─┤├── MPCC  v=8 m/s  ← punto más bajo (más preciso, más lento)
    │
    │   ■──┤├── DQ v=15 m/s
    │     ■──┤├── DQ v=12 m/s
    │       ■──┤├── DQ v=8 m/s
    └──────────────────────────────→  t_lap mediana [s]
                                      (← más rápido)
```

**Elementos del gráfico:**

| Elemento | Qué representa |
|----------|---------------|
| **Cada punto (●/■)** | Mediana de (t_lap, RMSE_pos) sobre las N corridas Monte Carlo |
| **Barra horizontal** | IQR (P25–P75) del t_lap → consistencia temporal |
| **Barra vertical** | IQR (P25–P75) del RMSE_pos → robustez ante perturbaciones |
| **Línea conectora** | Frontera de Pareto del controlador a través de las velocidades |
| **Anotaciones** | Valor de v_{θ,max} junto a cada punto |

### 2.4 Cómo interpretar los resultados

**Lectura rápida:** El controlador cuya curva está más **abajo y a la izquierda** es superior — tiene menor error Y completa más rápido.

**Interpretación caso por caso:**

| Situación observada | Conclusión |
|---------------------|------------|
| DQ-MPCC siempre abajo-izquierda de MPCC | DQ-MPCC **domina** en sentido Pareto: más preciso Y más rápido a toda velocidad |
| Las curvas se cruzan | A velocidades bajas un controlador es mejor, a altas el otro. La superioridad es **condicional al régimen de operación** |
| DQ-MPCC abajo pero más a la derecha | DQ-MPCC es más preciso pero más conservador. La mejora en precisión compensa o no según la aplicación |
| Barras de error más cortas en DQ-MPCC | DQ-MPCC es más **robusto** ante perturbaciones de condición inicial |
| Un controlador tiene puntos a v=15 pero el otro no | El controlador sin punto **divergió** a esa velocidad — no puede operar en ese régimen |

**Métricas derivadas para el análisis cuantitativo:**

- **Mejora en RMSE a igual velocidad:**

```
Δ%_RMSE(v) = (RMSE_MPCC(v) − RMSE_DQ(v)) / RMSE_MPCC(v) × 100
```

- **Mejora en tiempo de vuelta a igual velocidad:**

```
Δ%_tlap(v) = (tlap_MPCC(v) − tlap_DQ(v)) / tlap_MPCC(v) × 100
```

- **Normalized Performance Index (NPI):** Métrica escalar compuesta:

```
J_NPI = (1/|V|) Σ_v [ w_e · RMSE(v)/RMSE_ref + w_t · tlap(v)/tref ]
```

donde (w_e, w_t) son pesos de compromiso (ej: 0.5, 0.5) y los valores de
referencia son los del MPCC baseline a v = 8 m/s.

### 2.5 Plantilla de análisis post-experimento

Una vez ejecutado el experimento, el análisis debería reportar algo como:

> *"Para v_{θ,max} = X m/s, DQ-MPCC alcanzó un RMSE_pos de A ± B m
> (mediana ± IQR) completando el circuito en C ± D s, mientras que el MPCC
> baseline registró E ± F m en G ± H s. Esto representa una mejora del Δ%
> en precisión [y una aceleración/desaceleración del Δ% en tiempo de vuelta].
> A velocidades altas (v = Y m/s), la ventaja del DQ-MPCC [se mantiene / se
> reduce / se invierte], evidenciando que la formulación con cuaterniones
> duales [aprovecha mejor / se limita ante] las no-linealidades del
> acoplamiento traslación-rotación en maniobras agresivas."*

---

## 3. Cómo ejecutar los experimentos

### 3.1 Experimento 2 — Velocity Sweep

```bash
# Ejecutar el sweep completo
cd /path/to/DQ-MPCC_vs_MPCC_baseline
python3 2_run_experiment2_sweep.py

# Generar gráfico de boxplots (RMSE vs velocidad)
python3 plot_experiment2_boxplot.py

# Generar gráfico de Pareto (RMSE vs t_lap)
python3 plot_experiment2_pareto.py
```

### 3.2 Previsualización con datos sintéticos

```bash
python3 plot_experiment2_pareto.py --mock
```

### 3.3 Configuración

Editar `experiment2_config.py`:

```python
VELOCITIES = [8, 12, 15]   # velocidades a evaluar [m/s]
N_RUNS     = 5              # corridas Monte Carlo por punto (50 para paper)
S_MAX      = 100.0          # longitud del circuito [m]
SIGMA_P    = 0.05           # perturbación de posición inicial [m]
SIGMA_Q    = 0.05           # perturbación de orientación inicial [rad]
```

### 3.4 Archivos de salida

| Archivo | Contenido |
|---------|-----------|
| `experiment2_results/velocity_sweep_data.mat` | Datos crudos: RMSE, t_lap, mean_vtheta por (ctrl, v, run) |
| `experiment2_results/fig_pareto_rmse_vs_tlap.pdf` | Gráfico Pareto (publicación) |
| `experiment2_results/fig_velocity_sweep_boxplot.pdf` | Boxplots RMSE vs velocidad |
| `experiment2_results/velocity_sweep_table.tex` | Tabla LaTeX con medianas e IQR |
