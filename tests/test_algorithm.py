"""Handrechnungen für Geschwindigkeits-/Positions-Update und Diversität, End-to-End-Konvergenz auf Kugel-,
Ellipsoid- UND Rastrigin-Funktion (echt mehrgipflig) - eigene Implementierung UND `pyswarms` (verbreitetes
PSO-Referenzpaket) im Vergleich."""

import numpy as np
import pytest
from pyswarms.single import GlobalBestPSO

import pso_algorithm as A


def test_velocity_update_matches_hand_calculation():
    x = np.array([[0.0, 0.0]])
    v = np.array([[1.0, 1.0]])
    pbest_x = np.array([[2.0, 2.0]])
    gbest_x = np.array([4.0, 4.0])

    class FixedRng:
        def random(self, shape):
            return np.full(shape, 0.5)

    v_new = A.velocity_update(x, v, pbest_x, gbest_x, w=0.5, c1=1.0, c2=1.0, vmax=10.0, rng=FixedRng())
    # v_new = 0.5*[1,1] + 1.0*0.5*([2,2]-[0,0]) + 1.0*0.5*([4,4]-[0,0]) = [0.5,0.5]+[1,1]+[2,2] = [3.5,3.5]
    assert v_new.ravel().tolist() == pytest.approx([3.5, 3.5])


def test_velocity_update_is_clamped_to_vmax():
    x = np.array([[0.0, 0.0]])
    v = np.array([[0.0, 0.0]])
    pbest_x = np.array([[100.0, 100.0]])
    gbest_x = np.array([100.0, 100.0])

    class FixedRng:
        def random(self, shape):
            return np.full(shape, 1.0)

    v_new = A.velocity_update(x, v, pbest_x, gbest_x, w=0.5, c1=2.0, c2=2.0, vmax=5.0, rng=FixedRng())
    assert v_new.ravel().tolist() == pytest.approx([5.0, 5.0])


def test_position_update_matches_hand_calculation_without_boundary_hit():
    x = np.array([[1.0, 1.0]])
    v = np.array([[2.0, -0.5]])
    x_new, v_new = A.position_update(x, v, bounds=(0.0, 10.0))
    assert x_new.ravel().tolist() == pytest.approx([3.0, 0.5])
    assert v_new.ravel().tolist() == pytest.approx(v.ravel().tolist())


def test_position_update_zeros_velocity_on_boundary_hit():
    x = np.array([[9.5, 5.0]])
    v = np.array([[1.0, 0.5]])
    x_new, v_new = A.position_update(x, v, bounds=(0.0, 10.0))
    assert x_new.ravel().tolist() == pytest.approx([10.0, 5.5])
    assert v_new.ravel().tolist() == pytest.approx([0.0, 0.5])      # nur die Dimension, die den Rand traf


def test_diversity_matches_hand_calculation():
    x = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    assert A.diversity(x) == pytest.approx(np.sqrt(50.0))


def test_diversity_is_zero_for_a_collapsed_swarm():
    x = np.full((6, 2), 3.0)
    assert A.diversity(x) == pytest.approx(0.0)


# --- End-to-End: Konvergenz auf einfachen UND echt mehrgipfligen Testfunktionen ------------------------------------------------------------


def sphere(pop):
    return (np.asarray(pop) ** 2).sum(axis=-1)


def ellipsoid(pop, cond=100.0):
    pop = np.asarray(pop)
    dim = pop.shape[-1]
    scales = cond ** (np.arange(dim) / max(dim - 1, 1))
    return ((pop * scales) ** 2).sum(axis=-1)


def rastrigin(pop, a=10.0):
    pop = np.asarray(pop)
    dim = pop.shape[-1]
    return a * dim + (pop ** 2 - a * np.cos(2 * np.pi * pop)).sum(axis=-1)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_run_pso_converges_on_sphere_function(seed):
    dim = 3
    r = A.run_pso(sphere, dim, bounds=(-10.0, 10.0), pop_size=30, generations=200, w=0.7, c1=2.0, c2=2.0, vmax_fraction=0.2, seed=seed)
    assert r.best_fitness < 1e-6
    assert np.all(np.diff(r.best_history) <= 1e-12)      # best_history ist monoton fallend (gbest wird nie schlechter)


def test_run_pso_converges_on_ellipsoid_function():
    dim = 5
    r = A.run_pso(ellipsoid, dim, bounds=(-5.0, 5.0), pop_size=40, generations=300, w=0.7, c1=2.0, c2=2.0, vmax_fraction=0.2, seed=1)
    assert r.best_fitness < 1e-2


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_run_pso_converges_on_rastrigin_function(seed):
    """Rastrigin ist echt mehrgipflig - eine reine Bergsteiger-Suche bliebe im nächsten lokalen Minimum hängen."""
    dim = 3
    r = A.run_pso(rastrigin, dim, bounds=(-5.12, 5.12), pop_size=40, generations=200, w=0.7, c1=2.0, c2=2.0, vmax_fraction=0.2, seed=seed)
    assert r.best_fitness < 2.0


def test_run_pso_and_pyswarms_reach_a_comparably_good_optimum_on_the_sphere():
    """Kein Generation-für-Generation-Gleichlauf (unterschiedliche RNG-Nutzung) - aber beide Implementierungen
    sollen auf derselben einfachen konvexen Funktion mit vergleichbarem Budget nahe an 0 landen."""
    dim = 3
    r = A.run_pso(sphere, dim, bounds=(-10.0, 10.0), pop_size=30, generations=200, w=0.7, c1=2.0, c2=2.0, vmax_fraction=0.2, seed=1)

    options = {"c1": 2.0, "c2": 2.0, "w": 0.7}
    opt = GlobalBestPSO(n_particles=30, dimensions=dim, options=options, bounds=(np.full(dim, -10.0), np.full(dim, 10.0)))
    cost, _ = opt.optimize(lambda pop: (pop ** 2).sum(axis=1), iters=200, verbose=False)

    assert r.best_fitness < 1e-6
    assert cost < 1e-6


def test_run_pso_respects_bounds():
    r = A.run_pso(sphere, 2, bounds=(-1.0, 1.0), pop_size=10, generations=30, w=1.3, c1=2.0, c2=2.0, vmax_fraction=0.5, seed=1, keep_history=True)
    for g in r.generations:
        assert np.all(g.positions >= -1.0) and np.all(g.positions <= 1.0)


def test_run_pso_history_shapes_and_generations_snapshots():
    dim = 2
    pop_size = 8
    gens = 12
    r = A.run_pso(sphere, dim, bounds=(-10.0, 10.0), pop_size=pop_size, generations=gens, w=0.7, c1=2.0, c2=2.0, vmax_fraction=0.2, seed=3, keep_history=True)
    assert r.best_history.shape == (gens + 1,)
    assert r.diversity_history.shape == (gens + 1,)
    assert len(r.generations) == gens + 1
    for g in r.generations:
        assert g.positions.shape == (pop_size, dim)
        assert g.fitness.shape == (pop_size,)


def test_run_pso_without_history_leaves_generations_empty():
    r = A.run_pso(sphere, 2, bounds=(-10.0, 10.0), pop_size=8, generations=5, w=0.7, c1=2.0, c2=2.0, vmax_fraction=0.2, seed=1, keep_history=False)
    assert r.generations == []
