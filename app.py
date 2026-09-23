"""Partikelschwarm-Optimierung (PSO) - Geschwindigkeit statt Mutation - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Achtes Stück der Populations-Metaheuristiken-Linie der "Konzepte"-Reihe, dritter KONTRAST zu GA (kein Fix) für
kontinuierliche Landschaften: PSO (Kennedy & Eberhart, 1995; Trägheitsgewicht nach Shi & Eberhart, 1998) bewegt eine
Population über Geschwindigkeits-Updates statt Mutation/Kovarianz-Adaption/Differenzvektor - jedes Partikel wird von
seinem eigenen bisher besten Fund UND dem besten Fund des ganzen Schwarms angezogen. Vehikel ist dieselbe
kontinuierliche Standortwahl wie genetic-algorithm-demo/cma-es-demo/differential-evolution-demo/l-shade-demo.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import pso_constants as C
from pso_evaluation import Settings, analyse, comparison_experiment, sweep, w_experiment
from pso_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_run_seed, randomize_seed, sync_query_params
from pso_visualization import build_best_curve, build_comparison, build_diversity_curve, build_growing_example, build_sweep, build_w_experiment

st.set_page_config(page_title="PSO – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings, keep_history=True)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _comparison():
    return comparison_experiment()


@st.cache_data(show_spinner=False)
def _w_experiment():
    return w_experiment()


st.title("🧬 Partikelschwarm-Optimierung – Geschwindigkeit statt Mutation")
st.markdown(
    """
GA kombiniert per Crossover, CMA-ES passt eine Kovarianzmatrix an, DE mutiert über den Differenzvektor der
Population. **PSO** (Kennedy & Eberhart, 1995) geht einen vierten Weg: **kein** Crossover, **keine** Selektion im
klassischen Sinn - jedes Partikel bewegt sich mit einer **Geschwindigkeit**, die von zwei Kräften gezogen wird: dem
eigenen bisher besten Fund (kognitiv) und dem besten Fund des ganzen Schwarms (sozial). Ein **Trägheitsgewicht** $w$
steuert, wie stark die bisherige Bewegungsrichtung erhalten bleibt.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - "
    "achtes Stück der Populations-Metaheuristiken-Linie der \"Konzepte\"-Reihe, ein **Kontrast** zu "
    "[genetic-algorithm-demo](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/) statt eines Fixes - "
    "**ein** Verfahren an einem wachsenden Beispiel. Vehikel ist dieselbe kontinuierliche Standortwahl wie dort/bei "
    "[cma-es-demo](https://sebastianhanisch-cma-es-demo.streamlit.app/)/[differential-evolution-demo](https://sebastianhanisch-differential-evolution-demo.streamlit.app/)."
)

with st.expander("So funktioniert PSO", expanded=True):
    st.markdown(
        r"""
1. **Geschwindigkeits-Update.** $v_i \leftarrow w \cdot v_i + c_1 r_1 (pbest_i - x_i) + c_2 r_2 (gbest - x_i)$,
   $r_1, r_2 \sim U(0,1)$ je Partikel und Dimension - kognitiver Zug zum eigenen besten Fund, sozialer Zug zum
   besten Fund des ganzen Schwarms.
2. **Geschwindigkeits-Kappung.** $v_i$ wird auf $[-V_{max}, V_{max}]$ begrenzt, sonst könnten Partikel beliebig weit
   hinausschießen.
3. **Positions-Update.** $x_i \leftarrow x_i + v_i$ - einfache Addition, keine Rekombination mit anderen Partikeln.
4. **pbest/gbest.** Jedes Partikel merkt sich seinen eigenen besten Fund; der Schwarm merkt sich den besten Fund
   aller Partikel - beide werden nie schlechter.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
cols = st.columns(len(preset_names))
for col, name in zip(cols, preset_names):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**PSO**")
    pop = st.slider("Schwarmgröße N", *bounds("pop_slider"), key="pop_slider", step=C.POP_STEP)
    generations = st.slider("Generationen", *bounds("gens_slider"), key="gens_slider", step=C.GEN_STEP)
    w = st.slider("Trägheitsgewicht w", *bounds("w_slider"), key="w_slider", step=C.W_STEP, help="Wie stark die bisherige Bewegungsrichtung erhalten bleibt (konstant, kein linear abnehmendes Schema).")
    c1 = st.slider("Kognitiver Koeffizient c1", *bounds("c1_slider"), key="c1_slider", step=C.C1_STEP, help="Zug zum eigenen bisher besten Fund.")
    c2 = st.slider("Sozialer Koeffizient c2", *bounds("c2_slider"), key="c2_slider", step=C.C2_STEP, help="Zug zum besten Fund des ganzen Schwarms.")
    seed = st.number_input("Zufalls-Seed des Vehikels", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Vehikel generieren", width="stretch", on_click=randomize_seed)
    run_seed = st.number_input("Zufalls-Seed des PSO-Laufs", *bounds("run_seed_input"), key="run_seed_input", step=1)
    st.button("🎲 Neuen Lauf würfeln", width="stretch", on_click=randomize_run_seed)

sync_query_params({
    "pop_slider": int(pop), "gens_slider": int(generations), "w_slider": float(w), "c1_slider": float(c1), "c2_slider": float(c2),
    "seed_input": int(seed), "run_seed_input": int(run_seed),
})

settings = Settings(seed=int(seed), pop=int(pop), gens=int(generations), w=float(w), c1=float(c1), c2=float(c2), run_seed=int(run_seed))
with st.spinner("Rechne..."):
    a = _analysis(settings)
result = a.result
n_gens_run = len(result.generations) - 1
data_key = settings

# --- PSO in Aktion ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 PSO in Aktion")
if "pso_gen" not in st.session_state or st.session_state.get("pso_gen_owner") != data_key:
    st.session_state["pso_gen"] = n_gens_run
    st.session_state["pso_gen_owner"] = data_key
gen_col, play_col = st.columns([5, 2])
with gen_col:
    gen = st.slider("Generation", 0, n_gens_run, key="pso_gen", help="0 = Startschwarm.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()


def _frames():
    if n_gens_run == 0:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n_gens_run, min(n_gens_run + 1, 40))})


def _render(g):
    gd = result.generations[g]
    with view_slot.container():
        c1_, c2_ = st.columns([3, 2])
        c1_.markdown(f"**Generation {g} von {n_gens_run} – Diversität: {result.diversity_history[g]:.3f}**")
        c1_.plotly_chart(build_growing_example(a.inst, gd, a.grid_xy), width="stretch", key=f"g_map_{g}")
        c2_.markdown("**Bester Fund bisher (gbest)**")
        c2_.plotly_chart(build_best_curve(result.best_history[:g + 1], reference=a.grid_cost), width="stretch", key=f"g_best_{g}")


if auto_play:
    for fr in _frames():
        _render(fr)
        time.sleep(0.15)
else:
    _render(gen)

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was PSO gefunden hat")
m1, m2, m3 = st.columns(3)
m1.metric("Im globalen Trichter gelandet?", "Ja" if a.found_global else "Nein")
m2.metric("Abstand zum Gitter-Optimum", f"{a.gap:+.1f} %")
m3.metric("Diversität am Ende", f"{result.diversity_history[-1]:.4f}", delta=f"Start {result.diversity_history[0]:.2f}", delta_color="off")
st.plotly_chart(build_diversity_curve(result.diversity_history), width="stretch")

st.markdown("---")

# --- Sweep -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt die Trefferquote von Schwarmgröße und Trägheitsgewicht ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(C.SWEEP_LABELS), format_func=lambda k: C.SWEEP_LABELS[k], key="sweep_select")
base_sweep = Settings(seed=settings.seed, gens=settings.gens, pop=settings.pop, w=settings.w, c1=settings.c1, c2=settings.c2)
if st.button("Sweep über 5 feste Vehikel berechnen (dauert etwa 10 bis 30 Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base_sweep)}
if (sweep_param, base_sweep) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep..."):
        rows_sweep = _sweep(sweep_param, base_sweep)
    st.plotly_chart(build_sweep(rows_sweep, C.SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_chart")

st.markdown("---")

# --- Experiment 1: Kopfexperiment gegen GA, CMA-ES, DE -----------------------------------------------------------------------------------------

st.subheader("🔬 Wie schlägt sich PSO gegen GA, CMA-ES und DE?")
st.caption(
    f"Dieselbe Standortwahl-Landschaft wie die Vorgänger-Demos (Vehikel-Seed {C.DEFAULT_SEED}) - dort trafen GA "
    f"{C.GA_SUCCESS_SMALL:.0%}/{C.GA_SUCCESS_LARGE:.0%}, CMA-ES {C.CMA_SUCCESS_SMALL:.0%}/{C.CMA_SUCCESS_LARGE:.0%} "
    f"und DE {C.DE_SUCCESS_SMALL:.0%}/{C.DE_SUCCESS_LARGE:.0%} die globale Mulde bei denselben zwei Budgets. PSO "
    f"bekommt dieselben Budgets, Standard-N/w - Ausgang vorab offen."
)
if st.button("PSO gegen GA, CMA-ES und DE rechnen (dauert etwa 10 Sekunden)", key="comparison_start"):
    st.session_state["comparison_on"] = True
if st.session_state.get("comparison_on"):
    with st.spinner("Rechne 20 PSO-Läufe je Budget..."):
        report = _comparison()
    st.plotly_chart(build_comparison(report), width="stretch", key="comparison_chart")
    c1, c2 = st.columns(2)
    c1.metric(f"PSO, {report['pso_small_evals']} Auswertungen", f"{report['pso_small']:.0%}", delta=f"GA {report['ga_small']:.0%} · CMA-ES {report['cma_small']:.0%} · DE {report['de_small']:.0%}", delta_color="off")
    c2.metric(f"PSO, {report['pso_large_evals']} Auswertungen", f"{report['pso_large']:.0%}", delta=f"GA {report['ga_large']:.0%} · CMA-ES {report['cma_large']:.0%} · DE {report['de_large']:.0%}", delta_color="off")
    st.warning(
        "**Ehrlicher Befund:** PSO trifft hier mit Standardeinstellungen bei BEIDEN Budgets zuverlässig die globale "
        "Mulde - klar vor GA und CMA-ES, knapp hinter DE. Wie DE hält PSO eine über den Raum verstreute Population "
        "(hier: einen Schwarm), aber die Bewegung läuft über Geschwindigkeit statt Differenzvektor-Mutation - ein "
        "dritter, unabhängiger Mechanismus, der auf dieser Landschaft ähnlich gut funktioniert. Kein Beweis "
        "genereller Überlegenheit gegenüber GA/CMA-ES (siehe cma-es-demo/differential-evolution-demo für deren "
        "eigene Ehrliche Grenzen) - nur für diese Landschaft gemessen."
    )

st.markdown("---")

# --- Experiment 2: eigener Regler - Trägheitsgewicht w ----------------------------------------------------------------------------------------

st.subheader("🔬 Wie stark hängt die Trefferquote vom Trägheitsgewicht w ab?")
st.caption("Zu klein: das Momentum bricht schnell zusammen, der Schwarm konvergiert vorzeitig. Zu groß: Partikel schwingen/überschießen.")
if st.button(f"Trägheitsgewichte {C.W_VALUES[0]:.1f} bis {C.W_VALUES[-1]:.1f} vergleichen (dauert etwa 5 Sekunden)", key="w_start"):
    st.session_state["w_on"] = True
if st.session_state.get("w_on"):
    with st.spinner("Rechne 5 Trägheitsgewichte × 30 Läufe..."):
        rows_w = _w_experiment()
    st.plotly_chart(build_w_experiment(rows_w), width="stretch", key="w_chart")
    st.warning(
        "**Ehrlicher Befund:** Über 30 Läufe je w-Wert (eigenes knappes Budget) zeigt sich NICHT das erwartete U -  "
        "die Trefferquote steigt hier mit größerem w eher an (33 % bei w=0,2 bis 60 % bei w=1,3), statt bei einem "
        "mittleren Wert am besten zu sein. Die Diversität am Ende folgt dagegen sauber der Theorie (wächst von "
        "praktisch 0 bei w=0,2 auf über 18 bei w=1,3) - ein größeres Trägheitsgewicht hält den Schwarm länger "
        "beweglich, was bei diesem knappen Budget offenbar mehr hilft als schadet. Mit mehr Generationen könnte "
        "sich das erwartete Überschießen bei großem w noch zeigen - hier nicht geprüft."
    )

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Trägheitsgewicht w ist gut gewählt** | Der erwartete U-förmige Effekt (zu klein: vorzeitige Konvergenz; zu groß: Überschießen) zeigt sich bei diesem knappen Budget NICHT sauber - größeres w half hier eher (siehe Experiment oben). | Muss von Hand eingestellt werden, wie bei jedem Regler dieser Linie |
| **Kein linear abnehmendes Trägheitsgewicht** | Diese Demo nutzt ein konstantes w - viele PSO-Varianten lassen w über den Lauf von hoch auf niedrig sinken (Shi & Eberhart 1998), was Exploration und Exploitation zeitlich trennt. | Bewusste Vereinfachung, hier nicht umgesetzt |
| **Geschwindigkeits-Kappung $V_{max}$ ist fest (20 % der Gebietsbreite)** | Kein eigener Regler in dieser Demo, obwohl $V_{max}$ in der Praxis ebenfalls einen echten Effekt hat. | Bewusste Vereinfachung, siehe README |
| **gbest-Topologie (voll vernetzter Schwarm)** | Alle Partikel ziehen zum selben globalen Besten - anfällig für vorzeitige Konvergenz auf ein einzelnes Optimum. Ring-/lbest-Topologien mildern das, sind hier nicht umgesetzt. | lbest-PSO, andere Nachbarschaftsstrukturen |
"""
)
st.caption(
    "PSO ist ein Geschwister von CMA-ES und der DE→L-SHADE-Kette (alle drei Kontrast-Kinder von GA für "
    "kontinuierliche Landschaften) - kein Nachfolger in dieser Demo geplant. Vorgänger: "
    "[genetic-algorithm-demo](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/), "
    "[cma-es-demo](https://sebastianhanisch-cma-es-demo.streamlit.app/) und "
    "[differential-evolution-demo](https://sebastianhanisch-differential-evolution-demo.streamlit.app/), deren Befunde hier direkt verglichen werden."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Geschwindigkeits-Update.** $v_i \leftarrow w v_i + c_1 r_1 (pbest_i - x_i) + c_2 r_2 (gbest - x_i)$,
$r_1, r_2 \sim U(0,1)$ je Partikel und Dimension, auf $[-V_{max}, V_{max}]$ gekappt.

**Positions-Update.** $x_i \leftarrow x_i + v_i$, auf die Box geklemmt; bei Randberührung wird die betroffene
Geschwindigkeitskomponente auf 0 gesetzt.

**pbest/gbest.** $pbest_i \leftarrow x_i$ falls $f(x_i) < f(pbest_i)$; $gbest \leftarrow pbest_i$ falls
$f(pbest_i) < f(gbest)$ - beide werden nie schlechter.

Implementiert in `pso_algorithm.py` (Geschwindigkeit, Position, pbest/gbest, Hauptschleife), `pso_scenario.py`
(Vehikel), `pso_evaluation.py` (Kennzahlen, Sweep, Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
