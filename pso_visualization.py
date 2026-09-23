"""Plotly-Abbildungen der PSO-Demo: Kostenlandschaft mit Schwarm (wachsendes Beispiel), Vergleichs- und
Sweep-Abbildungen. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go

import pso_constants as C

SWARM_COLOR = "#9ecae9"
GBEST_COLOR = "#54a24b"
PSO_COLOR = "#4c78a8"
GA_COLOR = "#f58518"
CMA_COLOR = "#e45756"
DE_COLOR = "#72b7b2"
REF_COLOR = "#7f7f7f"
CONTOUR_SCALE = "Blues_r"

LANDSCAPE_RES = 60


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.1), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _map_layout(fig, height=430):
    fig.update_xaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False)
    return _base(fig, height)


def _landscape_grid(inst, res=LANDSCAPE_RES):
    xs = np.linspace(0.0, C.AREA, res)
    gx, gy = np.meshgrid(xs, xs)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    z = inst.cost(grid).reshape(res, res)
    return xs, z


def build_growing_example(inst, generation, grid_xy):
    """Landschaft + aktuelle Partikel-Positionen (`pso_algorithm.Generation`) + gbest."""
    xs, z = _landscape_grid(inst)
    fig = go.Figure()
    fig.add_trace(go.Contour(x=xs, y=xs, z=z, colorscale=CONTOUR_SCALE, showscale=False, contours=dict(coloring="fill", showlines=False), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[grid_xy[0]], y=[grid_xy[1]], mode="markers", marker=dict(size=13, symbol="star", color="#f58518", line=dict(width=1, color="white")), name="global günstigste Lage"))
    fig.add_trace(go.Scatter(x=generation.positions[:, 0], y=generation.positions[:, 1], mode="markers",
                              marker=dict(size=7, color=SWARM_COLOR, line=dict(width=0.5, color=PSO_COLOR)), name="Schwarm"))
    fig.add_trace(go.Scatter(x=[generation.gbest[0]], y=[generation.gbest[1]], mode="markers",
                              marker=dict(size=11, symbol="diamond", color=GBEST_COLOR, line=dict(width=1, color="white")), name="gbest"))
    return _map_layout(fig)


def build_diversity_curve(diversity_history):
    xs = list(range(len(diversity_history)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=diversity_history, mode="lines", line=dict(color=PSO_COLOR, width=2.5), showlegend=False))
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="Diversität des Schwarms")
    return _base(fig, 260)


def build_best_curve(best_history, reference=None):
    xs = list(range(len(best_history)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=best_history, mode="lines", line=dict(color=PSO_COLOR, width=2.5), name="gbest"))
    if reference is not None:
        fig.add_hline(y=reference, line=dict(color=REF_COLOR, dash="dot"), annotation_text="globales Optimum (Gitter)", annotation_position="bottom right")
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="Kosten")
    return _base(fig, 260)


def build_comparison(report):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"PSO<br>{report['pso_small_evals']} Auswertungen"], y=[report["pso_small"]], marker_color=PSO_COLOR, showlegend=False, width=0.3))
    fig.add_trace(go.Bar(x=[f"PSO<br>{report['pso_large_evals']} Auswertungen"], y=[report["pso_large"]], marker_color=PSO_COLOR, showlegend=False, width=0.3, opacity=0.6))
    fig.add_trace(go.Bar(x=["GA<br>klein"], y=[report["ga_small"]], marker_color=GA_COLOR, showlegend=False, width=0.3))
    fig.add_trace(go.Bar(x=["GA<br>groß"], y=[report["ga_large"]], marker_color=GA_COLOR, showlegend=False, width=0.3, opacity=0.6))
    fig.add_trace(go.Bar(x=["CMA-ES<br>klein"], y=[report["cma_small"]], marker_color=CMA_COLOR, showlegend=False, width=0.3))
    fig.add_trace(go.Bar(x=["CMA-ES<br>groß"], y=[report["cma_large"]], marker_color=CMA_COLOR, showlegend=False, width=0.3, opacity=0.6))
    fig.add_trace(go.Bar(x=["DE<br>klein"], y=[report["de_small"]], marker_color=DE_COLOR, showlegend=False, width=0.3))
    fig.add_trace(go.Bar(x=["DE<br>groß"], y=[report["de_large"]], marker_color=DE_COLOR, showlegend=False, width=0.3, opacity=0.6))
    fig.update_yaxes(title_text="Trefferquote im globalen Trichter", range=[0, 1], tickformat=".0%")
    return _base(fig, 380)


def build_w_experiment(rows):
    xs = [r["w"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["share_global"] for r in rows], mode="lines+markers", line=dict(color=PSO_COLOR, width=2.5), showlegend=False))
    fig.update_xaxes(title_text="Trägheitsgewicht w")
    fig.update_yaxes(title_text="Trefferquote im globalen Trichter", range=[0, 1], tickformat=".0%")
    return _base(fig, 300)


def build_sweep(rows, param_label):
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["share_global"] for r in rows], mode="lines+markers", line=dict(color=PSO_COLOR, width=2.5), showlegend=False))
    fig.update_xaxes(title_text=param_label)
    fig.update_yaxes(title_text="Trefferquote im globalen Trichter", range=[0, 1], tickformat=".0%")
    return _base(fig, 300)
