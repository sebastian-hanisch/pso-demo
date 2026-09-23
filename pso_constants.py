"""Konstanten der PSO-Demo: Vehikel (wie genetic-algorithm-demo/cma-es-demo/differential-evolution-demo/l-shade-demo,
kontinuierliche Standortwahl), PSO-Regler, Presets (Presets folgen nach den Messungen)."""

# --- Vehikel: Standortwahl (wortgleich aus genetic-algorithm-demo/ga_constants.py) -------------------------------------------------------

AREA = 100.0
K_WELLS = 5
WELL_MARGIN = 12.0                # Schwerpunkte liegen mindestens so weit vom Rand entfernt
AMP_MIN, AMP_MAX = 15.0, 40.0     # Tiefe eines Trichters
SIGMA_MIN, SIGMA_MAX = 6.0, 14.0  # Breite eines Trichters
GRID_STEP = 1.0                   # Auflösung des Referenz-Gitters (Grid-Search-Minimum) in km

DIM = 2
BOUNDS = (0.0, AREA)              # Box-Constraint (Clipping), wie differential-evolution-demo/l-shade-demo

# --- Partikelschwarm-Optimierung --------------------------------------------------------------------------------------------------------------

POP_MIN, POP_MAX, DEFAULT_POP, POP_STEP = 6, 200, 30, 2         # Schwarmgröße N
GEN_MIN, GEN_MAX, DEFAULT_GEN, GEN_STEP = 10, 400, 100, 10
W_MIN, W_MAX, DEFAULT_W, W_STEP = 0.1, 1.5, 0.7, 0.05            # Trägheitsgewicht (konstant, kein linear abnehmendes Schema)
C1_MIN, C1_MAX, DEFAULT_C1, C1_STEP = 0.5, 3.0, 2.0, 0.1         # kognitiver Koeffizient (Kennedy & Eberhart/Shi & Eberhart Standard)
C2_MIN, C2_MAX, DEFAULT_C2, C2_STEP = 0.5, 3.0, 2.0, 0.1         # sozialer Koeffizient
VMAX_FRACTION = 0.2               # Geschwindigkeits-Kappung als Anteil der Gebietsbreite (fest, kein Regler - siehe README)
SEED_MAX = 999999
DEFAULT_SEED = 35                 # Vehikel-Seed (wie genetic-algorithm-demo/cma-es-demo/differential-evolution-demo/l-shade-demo)
DEFAULT_RUN_SEED = 7              # Seed des PSO-Laufs selbst

GLOBAL_TOL_KM = 3.0                # Standort gilt als "im globalen Trichter" gefunden (wie GA/CMA-ES/DE/L-SHADE)

# --- Kopfexperiment: PSO gegen die in den Vorgänger-Demos gemessenen Zahlen ------------------------------------------------------------------
# GA maß auf DERSELBEN Landschaft (Vehikel-Seed 35): Populationsgröße 10 (150 Generationen, 1510 Auswertungen) trifft
# die globale Mulde in 55 % von 20 Läufen, Populationsgröße 100 (15100 Auswertungen) in 95 %. cma-es-demo maß auf
# denselben zwei Budgets 15 % bei BEIDEN. differential-evolution-demo maß 100 % bei BEIDEN.

GA_SUCCESS_SMALL, GA_EVALS_SMALL = 0.55, 1510
GA_SUCCESS_LARGE, GA_EVALS_LARGE = 0.95, 15100
CMA_SUCCESS_SMALL, CMA_SUCCESS_LARGE = 0.15, 0.15
DE_SUCCESS_SMALL, DE_SUCCESS_LARGE = 1.00, 1.00
COMPARISON_SEEDS = tuple(range(2200000, 2200020))

# --- Eigener Regler: Trägheitsgewicht w -------------------------------------------------------------------------------------------------------

W_VALUES = (0.2, 0.4, 0.7, 1.0, 1.3)
W_EXPERIMENT_SEEDS = tuple(range(2300000, 2300030))    # 30 Läufe je w-Wert - eigenes knappes Budget unten, sonst Ceiling-Effekt
W_EXPERIMENT_POP = 10
W_EXPERIMENT_GENS = 40

SWEEP_SEEDS = tuple(range(2400000, 2400005))
SWEEP_VALUES = {"pop": (6, 10, 20, 30, 50), "w": W_VALUES}
SWEEP_LABELS = {"pop": "Schwarmgröße N", "w": "Trägheitsgewicht w"}


def _preset(pop=DEFAULT_POP, gens=DEFAULT_GEN, w=DEFAULT_W, c1=DEFAULT_C1, c2=DEFAULT_C2, seed=DEFAULT_SEED, run_seed=DEFAULT_RUN_SEED):
    return {"pop": pop, "gens": gens, "w": w, "c1": c1, "c2": c2, "seed": seed, "run_seed": run_seed}


PRESETS = {
    "Standardfall": _preset(),
    "Kleines Trägheitsgewicht": _preset(w=W_VALUES[0]),
    "Großes Trägheitsgewicht": _preset(w=W_VALUES[-1]),
    "Große Schwarmgröße": _preset(pop=50),
}
PRESET_HELP = {
    "Standardfall": "N=30, w=0,7, 100 Generationen (Standard-Seed): trifft die globale Mulde fast exakt (-0,02 % Abstand), Diversität am Ende 0,51 - der Schwarm oszilliert noch leicht um gbest.",
    "Kleines Trägheitsgewicht": "w=0,2 statt 0,7: trifft hier ebenfalls die globale Mulde, der Schwarm kollabiert aber vollständig (Diversität 0,0) - das Momentum bricht schnell zusammen.",
    "Großes Trägheitsgewicht": "w=1,3 statt 0,7: trifft ebenfalls die globale Mulde, der Schwarm bleibt aber stark verstreut (Diversität 15,5 statt 0,51) - die Partikel schwingen noch deutlich um gbest.",
    "Große Schwarmgröße": "N=50 statt 30, sonst wie im Standardfall: trifft dieselbe Mulde ebenso zuverlässig, mit mehr Rechenaufwand je Generation - bei dieser Landschaft kein zusätzlicher Nutzen gegenüber N=30 (siehe Sweep).",
}
