"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Buttons (Standardmuster aus dem Demo-Portfolio, siehe
lshade_presets.py in l-shade-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import pso_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "pop_slider": SettingSpec("pop", int, C.DEFAULT_POP, C.POP_MIN, C.POP_MAX),
    "gens_slider": SettingSpec("gens", int, C.DEFAULT_GEN, C.GEN_MIN, C.GEN_MAX),
    "w_slider": SettingSpec("w", float, C.DEFAULT_W, C.W_MIN, C.W_MAX),
    "c1_slider": SettingSpec("c1", float, C.DEFAULT_C1, C.C1_MIN, C.C1_MAX),
    "c2_slider": SettingSpec("c2", float, C.DEFAULT_C2, C.C2_MIN, C.C2_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "run_seed_input": SettingSpec("rseed", int, C.DEFAULT_RUN_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"pop": "pop_slider", "gens": "gens_slider", "w": "w_slider", "c1": "c1_slider", "c2": "c2_slider", "seed": "seed_input", "run_seed": "run_seed_input"}
KEPT = {}
STEPS = {"pop_slider": C.POP_STEP, "gens_slider": C.GEN_STEP, "w_slider": C.W_STEP, "c1_slider": C.C1_STEP, "c2_slider": C.C2_STEP}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))    # Rundungs-Artefakte (z. B. 1.5000000000000002) nie über hi/unter lo lassen
            st.session_state[key] = int(snapped) if isinstance(spec.default, int) else snapped
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


def randomize_run_seed():
    st.session_state["run_seed_input"] = random.randint(0, C.SEED_MAX)
