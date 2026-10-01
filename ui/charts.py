"""
Plotly chart builders.

Colours follow a validated categorical palette in FIXED order (never cycled):
slot 1 blue, slot 2 orange, slot 3 aqua. Status colours are reserved for
good / warning / critical and always appear with a text label. Thin marks,
recessive grid, one y axis per chart, hover on every chart.
"""

import plotly.graph_objects as go

from utils.formatting import format_inr, format_lakh_crore

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]           # categorical slots 1-3
STATUS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}
INK_MUTED = "#898781"
GRID = "rgba(137,135,129,0.25)"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def _layout(fig: go.Figure, height: int = 320, y_title: str | None = None, show_legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=24, b=8), font=dict(family=FONT, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
        showlegend=show_legend, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=INK_MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=INK_MUTED), title=y_title)
    return fig


def cash_flow_bars(income: float, expenses: float, emi: float, investment: float, free_surplus: float) -> go.Figure:
    """Where the monthly income goes. One series, so one colour; values labelled."""
    labels = ["Living expenses", "EMIs", "Current SIP", "Free surplus"]
    values = [expenses, emi, investment, free_surplus]
    colors = [SERIES[0]] * 3 + [SERIES[0] if free_surplus >= 0 else STATUS["critical"]]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker=dict(color=colors, cornerradius=4),
        text=[format_inr(v) for v in values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    fig.add_vline(x=income, line=dict(color=INK_MUTED, dash="dot", width=1.5),
                  annotation_text=f"Income {format_inr(income)}", annotation_position="top")
    _layout(fig, 280, show_legend=False)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=True, gridcolor=GRID, tickprefix="₹")
    return fig


def sip_comparison(required: float, current: float) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=["Required SIP", "Current SIP"], y=[required, current],
        marker=dict(color=[SERIES[0], SERIES[1]], cornerradius=4), width=0.45,
        text=[format_inr(required), format_inr(current)], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{text}<extra></extra>",
    ))
    _layout(fig, 300, "Per month (₹)", show_legend=False)
    fig.update_layout(hovermode="closest")
    return fig


def projection_lines(series: dict[str, list[dict]], target: float | None = None) -> go.Figure:
    """series: {name: [{year, value}]}; up to 3 lines in fixed colour order, target as a dotted reference."""
    fig = go.Figure()
    for i, (name, rows) in enumerate(series.items()):
        fig.add_trace(go.Scatter(
            x=[r["year"] for r in rows], y=[r["value"] for r in rows], name=name, mode="lines",
            line=dict(color=SERIES[i % 3], width=2),
            customdata=[format_lakh_crore(r["value"]) for r in rows],
            hovertemplate=f"{name}: %{{customdata}}<extra></extra>",
        ))
    if target:
        fig.add_hline(y=target, line=dict(color=INK_MUTED, dash="dot", width=1.5),
                      annotation_text=f"Target {format_lakh_crore(target)}", annotation_position="top left")
    _layout(fig, 340, "Projected value")
    fig.update_xaxes(title="Years from now", dtick=1 if len(next(iter(series.values()))) <= 16 else None)
    top = max([r["value"] for rows in series.values() for r in rows] + [target or 0])
    ticks = inr_ticks(top)
    fig.update_yaxes(tickvals=ticks, ticktext=[format_lakh_crore(t) if t else "₹0" for t in ticks])
    return fig


def inr_ticks(top: float, n: int = 5) -> list[float]:
    """About n 'nice' tick values from 0 to top, so the axis reads in lakh / crore, not 3M."""
    if top <= 0:
        return [0.0]
    raw = top / n
    mag = 10 ** len(str(int(raw))) / 10
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    return [i * step for i in range(int(top // step) + 2)]


def allocation_donut(equity: float, debt: float, gold: float) -> go.Figure:
    labels, values = ["Equity", "Debt", "Gold"], [equity, debt, gold]
    keep = [(l, v, c) for l, v, c in zip(labels, values, SERIES) if v > 0]
    fig = go.Figure(go.Pie(
        labels=[k[0] for k in keep], values=[k[1] for k in keep], hole=0.62, sort=False,
        marker=dict(colors=[k[2] for k in keep], line=dict(color="rgba(255,255,255,0.9)", width=2)),
        textinfo="label+percent", hovertemplate="%{label}: %{value:.0f}%<extra></extra>",
    ))
    _layout(fig, 320, show_legend=False)  # slices are labelled directly
    fig.update_layout(hovermode="closest")
    return fig


def risk_meter(score: int, category: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(suffix=" / 21"),
        title=dict(text=category, font=dict(size=16)),
        gauge=dict(
            axis=dict(range=[7, 21], tickvals=[7, 11.5, 16.5, 21], ticktext=["7", "Conservative | Moderate", "Moderate | Aggressive", "21"],
                      tickfont=dict(size=10, color=INK_MUTED)),
            bar=dict(color="#0d366b", thickness=0.3),
            steps=[dict(range=[7, 11.5], color="#cde2fb"), dict(range=[11.5, 16.5], color="#86b6ef"),
                   dict(range=[16.5, 21], color="#3987e5")],
        ),
    ))
    fig.update_layout(height=240, margin=dict(l=24, r=24, t=48, b=8), font=dict(family=FONT),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig
