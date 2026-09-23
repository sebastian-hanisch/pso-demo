"""Presets: Vollständigkeit, gültige Werte, Grenzen/Schrittweiten - reine Datenprüfungen ohne Streamlit-Session
(Permalink-Klammern und Preset-Knöpfe werden über AppTest in test_app.py geprüft, wie im Rest des Portfolios üblich)."""

import pso_constants as C
import pso_evaluation as E
import pso_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for name, p in C.PRESETS.items():
        assert C.POP_MIN <= p["pop"] <= C.POP_MAX
        assert C.GEN_MIN <= p["gens"] <= C.GEN_MAX
        assert C.W_MIN <= p["w"] <= C.W_MAX
        assert C.C1_MIN <= p["c1"] <= C.C1_MAX and C.C2_MIN <= p["c2"] <= C.C2_MAX
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            spec.caster(p[key])


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Standardfall"]
    s = E.Settings(seed=p["seed"], pop=p["pop"], gens=p["gens"], w=p["w"], c1=p["c1"], c2=p["c2"], run_seed=p["run_seed"])
    assert s == E.Settings()


def test_bounds_and_steps_constants():
    assert P.bounds("pop_slider") == (C.POP_MIN, C.POP_MAX)
    assert P.bounds("w_slider") == (C.W_MIN, C.W_MAX)
    assert P.bounds("seed_input") == (0, C.SEED_MAX)
    assert set(P.STEPS) == {"pop_slider", "gens_slider", "w_slider", "c1_slider", "c2_slider"}


def test_url_params_are_unique():
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_large_swarm_preset_uses_a_bigger_pop_than_default():
    assert C.PRESETS["Große Schwarmgröße"]["pop"] > C.DEFAULT_POP
