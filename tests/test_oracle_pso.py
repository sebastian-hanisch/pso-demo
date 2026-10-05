"""Orakel: Partikel-für-Partikel-Schleifenimplementierung (reines Python, math.exp) auf demselben Zufallsstrom
gegen `run_pso` (vektorisiert). Verglichen werden die ganze Trajektorie (Positionen, Fitness, gbest je Generation),
die Verlaufskurven und die Diversität - auch bei Randfällen (N=1, 0 Generationen, w=0, großes w, kleines vmax)."""

import math

import numpy as np
import pytest

import pso_algorithm as A
import pso_constants as C
import pso_scenario as S


def _cost_scalar(inst, p):
    s = 0.0
    for k in range(len(inst.amplitudes)):
        d2 = (p[0] - inst.centres[k][0]) ** 2 + (p[1] - inst.centres[k][1]) ** 2
        s += inst.amplitudes[k] * math.exp(-d2 / (2 * inst.sigmas[k] ** 2))
    return inst.offset - s


def _oracle_run(inst, N, G, w, c1, c2, vfrac, seed, lo=0.0, hi=100.0, dim=2):
    rng = np.random.default_rng(seed)
    vmax = vfrac * (hi - lo)
    u = rng.random((N, dim))
    x = [[lo + u[i][d] * (hi - lo) for d in range(dim)] for i in range(N)]
    v = [[0.0] * dim for _ in range(N)]
    f = [_cost_scalar(inst, x[i]) for i in range(N)]
    pb, pf = [list(p) for p in x], list(f)
    g = min(range(N), key=lambda i: (pf[i], i))
    gx, gf = list(pb[g]), pf[g]

    def div(xs):
        c = [sum(xs[i][d] for i in range(N)) / N for d in range(dim)]
        return sum(math.sqrt(sum((xs[i][d] - c[d]) ** 2 for d in range(dim))) for i in range(N)) / N

    hist, divs, snaps = [gf], [div(x)], [(np.array(x), np.array(f), np.array(gx))]
    for _ in range(G):
        r1, r2 = rng.random((N, dim)), rng.random((N, dim))
        for i in range(N):
            for d in range(dim):
                nv = w * v[i][d] + c1 * r1[i][d] * (pb[i][d] - x[i][d]) + c2 * r2[i][d] * (gx[d] - x[i][d])
                nv = max(-vmax, min(vmax, nv))
                nx = x[i][d] + nv
                if nx < lo or nx > hi:
                    nv, nx = 0.0, min(hi, max(lo, nx))
                v[i][d], x[i][d] = nv, nx
        f = [_cost_scalar(inst, x[i]) for i in range(N)]
        for i in range(N):
            if f[i] < pf[i]:
                pf[i], pb[i] = f[i], list(x[i])
        g = min(range(N), key=lambda i: (pf[i], i))
        if pf[g] < gf:
            gf, gx = pf[g], list(pb[g])
        hist.append(gf)
        divs.append(div(x))
        snaps.append((np.array(x), np.array(f), np.array(gx)))
    return gx, gf, np.array(hist), np.array(divs), snaps


CASES = [
    # (Vehikel-Seed, N, Generationen, w, c1, c2, vmax-Anteil, Lauf-Seed)
    (35, 30, 25, 0.7, 2.0, 2.0, 0.2, 7),
    (35, 1, 10, 0.7, 2.0, 2.0, 0.2, 3),       # ein Partikel
    (3, 5, 0, 0.7, 2.0, 2.0, 0.2, 4),         # keine Generation
    (11, 8, 20, 0.0, 2.0, 2.0, 0.2, 5),       # kein Impuls
    (11, 8, 20, 1.5, 3.0, 3.0, 0.2, 6),       # Schwarm explodiert -> viele Randtreffer
    (21, 6, 20, 0.9, 1.0, 3.0, 0.05, 8),      # kleines vmax
    (21, 6, 20, 0.9, 0.5, 0.5, 3.0, 9),       # vmax praktisch wirkungslos
    (5, 12, 15, 0.4, 2.5, 0.5, 0.5, 10),
]


@pytest.mark.parametrize("vseed,N,G,w,c1,c2,vf,rseed", CASES)
def test_run_pso_trajectory_matches_loop_oracle(vseed, N, G, w, c1, c2, vf, rseed):
    inst = S.generate_real(vseed)
    res = A.run_pso(lambda P: inst.cost(P), 2, C.BOUNDS, N, G, w, c1, c2, vf, rseed, keep_history=True)
    gx, gf, hist, divs, snaps = _oracle_run(inst, N, G, w, c1, c2, vf, rseed)
    assert res.best_individual == pytest.approx(gx, abs=1e-8)
    assert res.best_fitness == pytest.approx(gf, abs=1e-9)
    assert res.best_history == pytest.approx(hist, abs=1e-9)
    assert res.diversity_history == pytest.approx(divs, abs=1e-8)
    assert len(res.generations) == G + 1
    for (px, pfit, pg), gen in zip(snaps, res.generations):
        assert gen.positions == pytest.approx(px, abs=1e-8)
        assert gen.fitness == pytest.approx(pfit, abs=1e-9)
        assert gen.gbest == pytest.approx(pg, abs=1e-8)
        assert gen.positions.min() >= 0.0 and gen.positions.max() <= 100.0


def test_grid_optimum_equals_brute_force_over_the_grid():
    for vseed in (35, 4, 17):
        inst = S.generate_real(vseed)
        _, cost = S.grid_optimum(inst)
        brute = min(_cost_scalar(inst, (float(i), float(j))) for i in range(101) for j in range(101))
        assert cost == pytest.approx(brute, abs=1e-9)
