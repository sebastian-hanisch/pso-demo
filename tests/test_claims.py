"""Jede im README/PRESET_HELP/App genannte Zahl wird hier nachgerechnet - keine Behauptung ohne Test.

Einzelne 40-100-Generationen-Läufe sind chaotisch empfindlich gegenüber winziger Fließkomma-Rundung (siehe
feedback_ci_platform_robust_tests.md, und die eigene Erfahrung aus nsga2-demo/nsga3-demo/moead-demo/cma-es-demo/
differential-evolution-demo/l-shade-demo). Zahlen aus einem EINZELNEN Lauf (Presets) bekommen deshalb nur
Strukturgrenzen; Zahlen, die über mehrere Seeds mitteln (Experimente, Sweep), sind von Natur aus robuster und dürfen
engere (aber weiterhin großzügige) Bänder bekommen."""

import pytest

import pso_constants as C
import pso_evaluation as E


def _preset_analysis(name):
    p = C.PRESETS[name]
    s = E.Settings(seed=p["seed"], pop=p["pop"], gens=p["gens"], w=p["w"], c1=p["c1"], c2=p["c2"], run_seed=p["run_seed"])
    return E.analyse(s, keep_history=False)


# --- Einzelläufe (Presets) - nur Strukturgrenzen, keine Nähe zu einem Messwert ------------------------------------------------------------


def test_standardfall_preset_claims():
    a = _preset_analysis("Standardfall")
    assert -1.0 < a.gap < 50.0
    assert a.result.diversity_history[-1] >= 0.0


def test_kleines_traegheitsgewicht_preset_claims():
    a = _preset_analysis("Kleines Trägheitsgewicht")
    assert -1.0 < a.gap < 50.0


def test_grosses_traegheitsgewicht_preset_claims():
    a = _preset_analysis("Großes Trägheitsgewicht")
    assert -1.0 < a.gap < 50.0


def test_grosse_schwarmgroesse_preset_claims():
    a = _preset_analysis("Große Schwarmgröße")
    assert -1.0 < a.gap < 50.0


# --- Headlinezahlen der beiden Experimente + Sweep (mitteln über mehrere Seeds, robuster) -----------------------------------------------------


def test_comparison_experiment_headline_claims():
    report = E.comparison_experiment()
    assert report["ga_small"] == C.GA_SUCCESS_SMALL == 0.55 and report["ga_large"] == C.GA_SUCCESS_LARGE == 0.95
    assert report["cma_small"] == C.CMA_SUCCESS_SMALL == 0.15
    assert report["de_small"] == C.DE_SUCCESS_SMALL == 1.00
    # Kernbefund: PSO liegt bei BEIDEN Budgets klar vor GA und CMA-ES
    assert report["pso_small"] > report["ga_small"] + 0.1
    assert report["pso_small"] > report["cma_small"] + 0.5
    assert report["pso_large"] > report["cma_large"] + 0.5


def test_w_experiment_headline_claims():
    rows = E.w_experiment()
    by_w = {r["w"]: r for r in rows}
    assert set(by_w) == set(C.W_VALUES)
    for r in rows:
        assert 0.0 <= r["share_global"] <= 1.0
        assert r["diversity_end_median"] >= 0.0
    # Kernbefund: die Diversität am Ende wächst klar mit dem Trägheitsgewicht - unabhängig davon, ob die
    # Trefferquote selbst ein sauberes U zeigt oder nicht (siehe README, ehrlich als kein sauberes U berichtet).
    assert by_w[C.W_VALUES[-1]]["diversity_end_median"] > by_w[C.W_VALUES[0]]["diversity_end_median"]


def test_pop_sweep_headline_claims():
    rows = E.sweep("pop")
    assert [r["value"] for r in rows] == list(C.SWEEP_VALUES["pop"])
    for r in rows:
        assert 0.0 <= r["share_global"] <= 1.0
    # Kernbefund: eine zu kleine Schwarmgröße schadet klar erkennbar
    assert rows[0]["share_global"] < rows[-1]["share_global"]
