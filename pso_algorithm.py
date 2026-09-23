"""PSO-Kern: klassisches gbest-PSO mit Trägheitsgewicht (Kennedy & Eberhart 1995; Shi & Eberhart 1998). Kein Kern
der Vorgänger-Demos kopiert - andere Mechanik: Geschwindigkeits-Updates statt Mutation/Kovarianz-Adaption/
Differenzvektor, kein Crossover, keine Selektion im klassischen Sinn."""

from dataclasses import dataclass, field

import numpy as np


def init_swarm(pop_size, dim, bounds, rng):
    lo, hi = bounds
    x = lo + rng.random((pop_size, dim)) * (hi - lo)
    v = np.zeros((pop_size, dim))
    return x, v


def velocity_update(x, v, pbest_x, gbest_x, w, c1, c2, vmax, rng):
    """v <- w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x), r1/r2 ~ U(0,1) je Partikel und Dimension, auf [-vmax, vmax] gekappt."""
    r1 = rng.random(x.shape)
    r2 = rng.random(x.shape)
    v_new = w * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x[None, :] - x)
    return np.clip(v_new, -vmax, vmax)


def position_update(x, v, bounds):
    """x <- x + v, auf die Box geklemmt; bei Randberührung wird die betroffene Geschwindigkeitskomponente auf 0
    gesetzt (einfache, gängige Randbehandlung - verhindert, dass ein Partikel am Rand "klebt" und dagegen drückt)."""
    lo, hi = bounds
    x_new = x + v
    hit_bound = (x_new < lo) | (x_new > hi)
    x_new = np.clip(x_new, lo, hi)
    v_new = np.where(hit_bound, 0.0, v)
    return x_new, v_new


def diversity(x):
    """Mittlerer euklidischer Abstand vom Schwerpunkt (wie GA/DE/l-shade)."""
    centre = x.mean(axis=0)
    return float(np.mean(np.sqrt(((x - centre) ** 2).sum(axis=1))))


@dataclass
class Generation:
    positions: np.ndarray
    fitness: np.ndarray
    gbest: np.ndarray


@dataclass
class PSOResult:
    best_individual: np.ndarray
    best_fitness: float
    best_history: np.ndarray       # (generations + 1,)
    diversity_history: np.ndarray  # (generations + 1,)
    generations: list = field(default_factory=list)   # nur befüllt, wenn keep_history=True


def run_pso(cost_fn, dim, bounds, pop_size, generations, w, c1, c2, vmax_fraction, seed, keep_history=False):
    """Ein PSO-Lauf. `cost_fn(population) -> (pop_size,)`, niedriger ist besser."""
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    vmax = vmax_fraction * (hi - lo)

    x, v = init_swarm(pop_size, dim, bounds, rng)
    fitness = np.asarray(cost_fn(x))
    pbest_x = x.copy()
    pbest_f = fitness.copy()
    gbest_idx = int(np.argmin(pbest_f))
    gbest_x = pbest_x[gbest_idx].copy()
    gbest_f = float(pbest_f[gbest_idx])

    best_history = [gbest_f]
    diversity_history = [diversity(x)]
    gens_snapshots = [Generation(x.copy(), fitness.copy(), gbest_x.copy())] if keep_history else []

    for _ in range(generations):
        v = velocity_update(x, v, pbest_x, gbest_x, w, c1, c2, vmax, rng)
        x, v = position_update(x, v, bounds)
        fitness = np.asarray(cost_fn(x))

        improved = fitness < pbest_f
        pbest_x[improved] = x[improved]
        pbest_f[improved] = fitness[improved]

        gen_best_idx = int(np.argmin(pbest_f))
        if pbest_f[gen_best_idx] < gbest_f:
            gbest_f = float(pbest_f[gen_best_idx])
            gbest_x = pbest_x[gen_best_idx].copy()

        best_history.append(gbest_f)
        diversity_history.append(diversity(x))
        if keep_history:
            gens_snapshots.append(Generation(x.copy(), fitness.copy(), gbest_x.copy()))

    return PSOResult(gbest_x, gbest_f, np.array(best_history), np.array(diversity_history), gens_snapshots)
