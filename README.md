# 🧬 Partikelschwarm-Optimierung – Geschwindigkeit statt Mutation

Achtes Stück der **Populations-Metaheuristiken-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) –
Operations Research und Machine Learning. Dritter Kontrast zu [genetic-algorithm-demo](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/)
für kontinuierliche Landschaften, Geschwister von [cma-es-demo](https://sebastianhanisch-cma-es-demo.streamlit.app/)
und der DE→L-SHADE-Kette: Partikelschwarm-Optimierung (PSO, Kennedy & Eberhart 1995; Trägheitsgewicht nach Shi &
Eberhart 1998) bewegt eine Population über **Geschwindigkeits-Updates** statt Mutation/Kovarianz-Adaption/
Differenzvektor - kein Crossover, keine Selektion im klassischen Sinn. Vehikel ist dieselbe kontinuierliche
Standortwahl wie genetic-algorithm-demo/cma-es-demo/differential-evolution-demo/l-shade-demo.

## Warum dieses Problem

GA, CMA-ES und DE erzeugen neue Kandidaten explizit (Crossover, Kovarianzmatrix-Stichprobe, Differenzvektor) und
wählen dann per Selektion aus. PSO tut beides nicht: jedes Partikel bewegt sich kontinuierlich mit einer
**Geschwindigkeit**, die von zwei Kräften gezogen wird - dem eigenen bisher besten Fund (kognitiv, "was hat für MICH
funktioniert") und dem besten Fund des ganzen Schwarms (sozial, "was hat für UNS funktioniert"). Ein
**Trägheitsgewicht** $w$ steuert, wie stark die bisherige Bewegungsrichtung erhalten bleibt - der zentrale
Exploration-vs-Exploitation-Regler dieses Verfahrens.

## Modell

Dieselbe kontinuierliche Standortwahl wie genetic-algorithm-demo/cma-es-demo/differential-evolution-demo/l-shade-demo:
Kosten eines Punkts $(x,y)$ im 100×100-km-Gebiet sind mehrere Gauß-Mulden unterschiedlicher Tiefe/Breite
(`K_WELLS=5`) - nur die tiefste ist das globale Optimum. **`pso_scenario.generate_real` reproduziert die
Vehikel-Erzeugung wortgleich** - bei Standard-Vehikel-Seed 35 bitidentisch zu den Vorgänger-Demos.

## Methodik

Klassisches gbest-PSO mit Trägheitsgewicht (`pso_algorithm.py`, kein Kern der Vorgänger-Demos kopiert - andere
Mechanik): $v_i \leftarrow w v_i + c_1 r_1 (pbest_i - x_i) + c_2 r_2 (gbest - x_i)$, $r_1, r_2 \sim U(0,1)$ je
Partikel und Dimension, Geschwindigkeit auf $[-V_{max}, V_{max}]$ gekappt ($V_{max}$ = 20 % der Gebietsbreite);
$x_i \leftarrow x_i + v_i$, auf die Box geklemmt (Geschwindigkeitskomponente bei Randberührung auf 0 gesetzt);
$pbest$/$gbest$ werden nie schlechter. Konstantes Trägheitsgewicht (kein linear abnehmendes Schema), $c_1=c_2=2{,}0$
(Kennedy & Eberhart/Shi & Eberhart Standardwerte).

**Kreuzprobe gegen `pyswarms`** (verbreitetes PSO-Referenzpaket): Geschwindigkeits-/Positions-Update per
Handrechnung geprüft (arithmetisch einfach); beide Implementierungen finden auf einer einfachen konvexen Funktion
(Kugel) mit demselben Budget verlässlich nahe an 0 - kein Generation-für-Generation-Gleichlauf behauptet
(unterschiedliche RNG-Nutzung). Zusätzlich Konvergenz auf Ellipsoid- und Rastrigin-Funktion (echt mehrgipflig)
geprüft.

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| Wie schlägt sich PSO gegen GA, CMA-ES und DE auf derselben mehrgipfligen Landschaft? | Mit Standardeinstellungen trifft PSO die globale Mulde in **90 %** der Läufe - bei BEIDEN Budgets. GA erreicht dort 55 %/95 %, CMA-ES nur 15 %/15 %, DE 100 %/100 %. | `test_comparison_experiment_headline_claims` |
| Zeigt das Trägheitsgewicht w den erwarteten U-förmigen Effekt (zu klein/zu groß beide schlecht)? | **Nein, nicht sauber.** Bei knappem Budget steigt die Trefferquote eher MIT w (33 % bei w=0,2 bis 60 % bei w=1,3), statt bei einem mittleren Wert am besten zu sein. | `test_w_experiment_headline_claims` |
| Folgt wenigstens die Diversität der Theorie? | Ja, klar: die Diversität am Ende wächst sauber monoton mit w (von praktisch 0 bei w=0,2 auf über 18 bei w=1,3) - ein größeres Trägheitsgewicht hält den Schwarm länger beweglich. | `test_w_experiment_headline_claims` |
| Wie stark hängt die Trefferquote von der Schwarmgröße N ab? | Klar: 0 % bei N=6 gegen 80 % ab N≥30 - eine zu kleine Schwarmgröße schadet deutlich. | `test_pop_sweep_headline_claims` |
| Stimmen Geschwindigkeits-/Positions-Update mit der Literatur überein? | Per Handrechnung geprüft; eigene Implementierung UND `pyswarms` konvergieren auf Kugel-Funktion verlässlich nahe 0. | `test_run_pso_and_pyswarms_reach_a_comparably_good_optimum_on_the_sphere` |

## Ehrliche Grenzen

- **Der klassische U-förmige Trägheitsgewicht-Effekt zeigt sich hier nicht sauber** - bei knappem Budget half ein
  größeres w eher, statt bei einem mittleren Wert am besten zu sein. Mit mehr Generationen könnte sich das erwartete
  Überschießen bei großem w noch zeigen - hier nicht geprüft, ehrlich als offene Frage stehen gelassen.
- **Kein linear abnehmendes Trägheitsgewicht** - viele PSO-Varianten lassen w über den Lauf von hoch auf niedrig
  sinken (Shi & Eberhart 1998), was Exploration und Exploitation zeitlich trennt. Diese Demo nutzt ein konstantes w.
- **Geschwindigkeits-Kappung $V_{max}$ ist fest** (20 % der Gebietsbreite, kein eigener Regler) - hat in der Praxis
  ebenfalls einen echten Effekt.
- **gbest-Topologie** (voll vernetzter Schwarm) - alle Partikel ziehen zum selben globalen Besten, anfällig für
  vorzeitige Konvergenz auf ein einzelnes Optimum. Ring-/lbest-Topologien mildern das, sind hier nicht umgesetzt.
- **Kein Beweis genereller Überlegenheit** gegenüber GA/CMA-ES - der Vorsprung gilt für DIESE Landschaft.
- **Kein Nachfolger in dieser Demo geplant** - PSO ist ein Geschwister von CMA-ES und der DE→L-SHADE-Kette (alle
  drei Kontrast-Kinder von GA für kontinuierliche Landschaften).

## Tests

61 Tests (`pytest tests/ -v`): Geschwindigkeits-/Positions-Update und Diversität per Handrechnung geprüft, Konvergenz
auf Kugel-, Ellipsoid- und Rastrigin-Funktion (eigene Implementierung UND `pyswarms` im Vergleich auf der
Kugel-Funktion), Szenario-Erzeugung bitidentisch zu den Vorgänger-Demos geprüft, AppTest-Rauchtests (jedes Preset,
Generation-Slider inkl. Abspielen, Permalink-Grenzen, beide Experimente + Sweep auf Abruf) und `test_claims.py`
(jede Zahl aus diesem README, mit CI-robusten Bändern für Einzellauf-Kennzahlen - siehe
`feedback_ci_platform_robust_tests.md`, von Anfang an angewendet). Ein echter Bug wurde beim Bauen gefunden und
behoben: das Permalink-Schrittweiten-Runden konnte einen Wert knapp über die Slider-Obergrenze runden (z. B.
1.5000000000000002 bei max=1.5) und den App-Lauf mit einem `StreamlitValueAboveMaxError` abbrechen lassen - jetzt
wird nach dem Runden zusätzlich hart auf die Grenzen geklemmt.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `pso_constants.py` | Regler-Grenzen, Vehikel-Konstanten, Presets |
| `pso_presets.py` | Permalink/Presets-Mechanik |
| `pso_scenario.py` | Vehikel-Erzeuger (Standortwahl, Gauß-Mulden-Landschaft), wortgleich zu den Vorgänger-Demos |
| `pso_algorithm.py` | PSO-Kern (Geschwindigkeit, Position, pbest/gbest) |
| `pso_evaluation.py` | Kennzahlen, Kopfexperiment, Trägheitsgewicht-Experiment, Sweep |
| `pso_visualization.py` | Plotly-Abbildungen (Landschaft mit Schwarm, Vergleiche) |

## Bewusst nicht umgesetzt

- Linear abnehmendes Trägheitsgewicht (Shi & Eberhart 1998) - konstantes w stattdessen.
- Eigener Regler für die Geschwindigkeits-Kappung $V_{max}$ (fest auf 20 % der Gebietsbreite).
- lbest-/Ring-Topologien oder andere Nachbarschaftsstrukturen jenseits von gbest.
- Ein PDF-Export - wie bei den anderen Konzepte-Demos dieses Portfolios nicht Teil der Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy.
