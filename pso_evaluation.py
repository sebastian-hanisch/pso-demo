"""Auswertung der PSO-Demo: ein Lauf gegen das Gitter-Optimum, Sweep über N/w, und zwei Experimente -
Kopfexperiment (Trefferquote im globalen Trichter gegen die in den Vorgänger-Demos gemessenen Zahlen) und eigener
Regler (Trägheitsgewicht w)."""

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import pso_algorithm as A
import pso_constants as C
import pso_scenario as S


@dataclass(frozen=True)
class Settings:
    seed: int = C.DEFAULT_SEED
    pop: int = C.DEFAULT_POP
    gens: int = C.DEFAULT_GEN
    w: float = C.DEFAULT_W
    c1: float = C.DEFAULT_C1
    c2: float = C.DEFAULT_C2
    run_seed: int = C.DEFAULT_RUN_SEED


@lru_cache(maxsize=64)
def instance(seed):
    inst = S.generate_real(seed)
    grid_xy, grid_cost = S.grid_optimum(inst)
    return inst, grid_xy, grid_cost


def cost_fn(settings):
    inst, grid_xy, grid_cost = instance(settings.seed)

    def cost(pop):
        return inst.cost(pop)
    return cost, grid_xy, grid_cost


def run(settings, keep_history=False):
    cost, grid_xy, grid_cost = cost_fn(settings)
    return A.run_pso(cost, 2, C.BOUNDS, settings.pop, settings.gens, settings.w, settings.c1, settings.c2, C.VMAX_FRACTION, settings.run_seed, keep_history=keep_history)


@dataclass
class Analysis:
    settings: Settings
    result: object
    inst: object
    grid_xy: np.ndarray
    grid_cost: float

    @property
    def gap(self):
        if self.grid_cost == 0:
            return float("nan")
        return 100.0 * (self.result.best_fitness - self.grid_cost) / abs(self.grid_cost)

    @property
    def found_global(self):
        return bool(np.sqrt(((self.result.best_individual - self.grid_xy) ** 2).sum()) <= C.GLOBAL_TOL_KM)


def analyse(settings, keep_history=True):
    result = run(settings, keep_history=keep_history)
    inst, grid_xy, grid_cost = instance(settings.seed)
    return Analysis(settings, result, inst, grid_xy, grid_cost)


# --- Sweep (wie die Vorgänger-Demos) ---------------------------------------------------------------------------------------------------------


def run_config(param, value, base, seeds=None):
    seeds = C.SWEEP_SEEDS if seeds is None else seeds
    s0 = replace(base, **{param: value})
    hits, gaps = [], []
    for run_seed in seeds:
        a = analyse(replace(s0, run_seed=run_seed), keep_history=False)
        hits.append(a.found_global)
        gaps.append(a.gap)
    return {"share_global": float(np.mean(hits)), "gap": float(np.mean(gaps))}


def sweep(param, base=None, values=None):
    base = Settings() if base is None else base
    values = C.SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(param, v, base)} for v in values]


# --- Experiment 1: Kopfexperiment gegen GA/CMA-ES/DE-Zahlen -----------------------------------------------------------------------------------


def comparison_experiment(seed=None, seeds=None, pop=None, w=None, c1=None, c2=None):
    """Trefferquote im globalen Trichter bei zwei Budgets (Gesamtauswertungen ~GA_EVALS_SMALL/LARGE), Standard-N/w -
    direkt vergleichbar mit den in genetic-algorithm-demo/cma-es-demo/differential-evolution-demo gemessenen Zahlen."""
    seed = C.DEFAULT_SEED if seed is None else seed
    seeds = C.COMPARISON_SEEDS if seeds is None else seeds
    pop = C.DEFAULT_POP if pop is None else pop
    w = C.DEFAULT_W if w is None else w
    c1 = C.DEFAULT_C1 if c1 is None else c1
    c2 = C.DEFAULT_C2 if c2 is None else c2

    rows = {}
    for label, evals in (("small", C.GA_EVALS_SMALL), ("large", C.GA_EVALS_LARGE)):
        gens = max(1, evals // pop)
        hits = []
        for run_seed in seeds:
            s = Settings(seed=seed, pop=pop, gens=gens, w=w, c1=c1, c2=c2, run_seed=run_seed)
            a = analyse(s, keep_history=False)
            hits.append(a.found_global)
        rows[label] = {"share_global": float(np.mean(hits)), "gens": gens, "pop": pop, "evals": pop * (gens + 1)}
    return {
        "pso_small": rows["small"]["share_global"], "pso_large": rows["large"]["share_global"],
        "pso_small_evals": rows["small"]["evals"], "pso_large_evals": rows["large"]["evals"],
        "ga_small": C.GA_SUCCESS_SMALL, "ga_large": C.GA_SUCCESS_LARGE,
        "cma_small": C.CMA_SUCCESS_SMALL, "cma_large": C.CMA_SUCCESS_LARGE,
        "de_small": C.DE_SUCCESS_SMALL, "de_large": C.DE_SUCCESS_LARGE,
    }


# --- Experiment 2: eigener Regler - Trägheitsgewicht w ----------------------------------------------------------------------------------------


def w_experiment(seed=None, values=None, seeds=None, gens=None, pop=None):
    seed = C.DEFAULT_SEED if seed is None else seed
    values = C.W_VALUES if values is None else values
    seeds = C.W_EXPERIMENT_SEEDS if seeds is None else seeds
    gens = C.W_EXPERIMENT_GENS if gens is None else gens
    pop = C.W_EXPERIMENT_POP if pop is None else pop

    rows = []
    for w in values:
        hits, final_divs = [], []
        for run_seed in seeds:
            s = Settings(seed=seed, pop=pop, gens=gens, w=w, run_seed=run_seed)
            r = run(s, keep_history=False)
            _, grid_xy, _ = instance(seed)
            hit = bool(np.sqrt(((r.best_individual - grid_xy) ** 2).sum()) <= C.GLOBAL_TOL_KM)
            hits.append(hit)
            final_divs.append(float(r.diversity_history[-1]))
        rows.append({"w": w, "share_global": float(np.mean(hits)), "diversity_end_median": float(np.median(final_divs))})
    return rows
