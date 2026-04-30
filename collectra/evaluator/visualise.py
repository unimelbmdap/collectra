__all__ = ["generate_evaluation_html"]

from collections import defaultdict
from pathlib import Path
from typing import Any

import plotly.colors

from ..logger import get_logger
from .base import EvaluationReport

logger = get_logger(__name__)

# ── Colour palette ────────────────────────────────────────────────────────────
_PALETTE = plotly.colors.qualitative.Pastel


def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


# ── Layout config ─────────────────────────────────────────────────────────────
_EXCLUDED_TYPES = {"Image"}  # data types to omit from the report
_ROW_HEIGHT_PX = 600  # pixels per subplot row
_VERTICAL_SPACING = 0.06  # fraction of figure height between rows

# ── Scatter (Precision vs Recall) config ──────────────────────────────────────
_BUBBLE_MIN = 6  # minimum bubble diameter (px)
_BUBBLE_MAX = 24  # maximum bubble diameter (px)
_BUBBLE_OPACITY = 0.85
_SCATTER_RANGE = (0, 1.05)  # axis range for both P and R

# ── Box plot config ───────────────────────────────────────────────────────────
_SCORE_RANGE = (0.0, 1.05)  # y-axis range for match score

# ── Legend config ─────────────────────────────────────────────────────────────
_LEGEND_X = 1.02  # x anchor (fraction of figure width)
_LEGEND_BGCOLOR = "rgba(30,30,30,0.8)"
_LEGEND_BORDER = "rgba(255,255,255,0.15)"

# ── Page chrome ───────────────────────────────────────────────────────────────
_PAPER_BG = "#121212"
_PLOT_BG = "#121212"
_PAGE_BG = "#000000"


# ── Data helpers ──────────────────────────────────────────────────────────────


def _label_colors(report: EvaluationReport, data_types: list[str]) -> dict[str, str]:
    """Assign a stable palette color to every label across all data types."""
    ordered: list[str] = []
    for dt in data_types:
        for lbl in report.aggregate[dt].get("per_label", {}):
            if lbl not in ordered:
                ordered.append(lbl)
    return {lbl: _color(i) for i, lbl in enumerate(ordered)}


def _collect_scores(
    report: EvaluationReport,
    data_types: list[str],
) -> dict[tuple[str, str], list[float]]:
    """Collect per-match scores keyed by (data_type, label)."""
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    allowed = set(data_types)
    for fr in report.file_results:
        for lbl, lm in fr.label_metrics.items():
            for dt in lm.get("types", []):
                if dt in allowed:
                    for m in lm.get("matches", []):
                        scores[(dt, lbl)].append(m["score"])
    return scores


# ── Trace builders ────────────────────────────────────────────────────────────


def _scatter_traces(
    per_label: dict,
    legend_ref: str,
    dt: str,
    colors: dict[str, str],
) -> list:
    """One bubble trace per label for the Precision vs Recall plot."""
    import plotly.graph_objects as go

    supports = [v.get("support", 1) for v in per_label.values()]
    max_support = max(supports, default=1)

    traces = []
    for lbl, metrics in per_label.items():
        support = metrics.get("support", 1)
        size = _BUBBLE_MIN + (_BUBBLE_MAX - _BUBBLE_MIN) * support / max_support
        traces.append(
            go.Scatter(
                x=[metrics.get("recall", 0.0)],
                y=[metrics.get("precision", 0.0)],
                mode="markers",
                name=lbl,
                legendgroup=f"{dt}:{lbl}",
                legend=legend_ref,
                text=[f"{lbl}<br>support={support}"],
                hovertemplate="%{text}<br>P=%{y:.3f}  R=%{x:.3f}<extra></extra>",
                marker=dict(size=size, opacity=_BUBBLE_OPACITY, color=colors[lbl]),
            )
        )
    return traces


def _box_traces(
    labels: list[str],
    scores: dict[tuple[str, str], list[float]],
    legend_ref: str,
    dt: str,
    colors: dict[str, str],
) -> list:
    """One box trace per label for the Match Score distribution plot."""
    import plotly.graph_objects as go

    traces = []
    for lbl in labels:
        lbl_scores = scores.get((dt, lbl), [])
        if lbl_scores:
            traces.append(
                go.Box(
                    y=lbl_scores,
                    name=lbl,
                    legendgroup=f"{dt}:{lbl}",
                    legend=legend_ref,
                    showlegend=False,
                    marker_color=colors[lbl],
                    boxmean=True,
                )
            )
    return traces


# ── Main entry point ──────────────────────────────────────────────────────────


def generate_evaluation_html(report: EvaluationReport, output_path: Path) -> None:
    """Build a dark-themed Plotly HTML report.

    For each included data type, two subplot rows are produced:
      - Precision vs Recall scatter bubble (per label)
      - Match score box plot (per label)
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error(
            "plotly is required for HTML reports. Install with: pip install plotly"
        )
        return

    data_types = [dt for dt in report.aggregate if dt not in _EXCLUDED_TYPES]

    if not data_types:
        Path(output_path).write_text(
            "<html><body><p>No evaluation data available.</p></body></html>",
            encoding="utf-8",
        )
        return

    colors = _label_colors(report, data_types)
    scores = _collect_scores(report, data_types)

    n_rows = 2 * len(data_types)
    row_height = (1.0 - (n_rows - 1) * _VERTICAL_SPACING) / n_rows

    subplot_titles = [
        title
        for dt in data_types
        for title in (f"Precision vs Recall — {dt}", f"Match Score by Label — {dt}")
    ]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=_VERTICAL_SPACING,
    )
    fig.update_layout(
        template="plotly_dark",
        title_text="Collectra Evaluation Report",
        height=_ROW_HEIGHT_PX * n_rows,
        colorway=_PALETTE,
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
    )

    for dt_idx, dt in enumerate(data_types):
        scatter_row = dt_idx * 2 + 1
        box_row = dt_idx * 2 + 2
        legend_ref = "legend" if dt_idx == 0 else f"legend{dt_idx + 1}"
        per_label = report.aggregate[dt].get("per_label", {})
        labels = list(per_label.keys())

        # Legend — anchored to the top of this data type's section
        legend_y = 1.0 - dt_idx * (2 * row_height + 2 * _VERTICAL_SPACING)
        _legend_update: dict[str, Any] = {
            legend_ref: dict(
                title=dict(text=f"<b>{dt}</b>", side="top center"),
                x=_LEGEND_X,
                y=legend_y,
                yanchor="top",
                bgcolor=_LEGEND_BGCOLOR,
                bordercolor=_LEGEND_BORDER,
                borderwidth=1,
                itemsizing="constant",
            )
        }
        fig.update_layout(**_legend_update)

        for trace in _scatter_traces(per_label, legend_ref, dt, colors):
            fig.add_trace(trace, row=scatter_row, col=1)
        fig.update_xaxes(
            title_text="Recall", range=list(_SCATTER_RANGE), row=scatter_row, col=1
        )
        fig.update_yaxes(
            title_text="Precision", range=list(_SCATTER_RANGE), row=scatter_row, col=1
        )

        for trace in _box_traces(labels, scores, legend_ref, dt, colors):
            fig.add_trace(trace, row=box_row, col=1)
        fig.update_yaxes(
            title_text="Match Score", range=list(_SCORE_RANGE), row=box_row, col=1
        )

    _include_plotlyjs: Any = "cdn"
    html_fig = pio.to_html(fig, include_plotlyjs=_include_plotlyjs, full_html=False)

    full_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Collectra Evaluation Report</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      margin: 0;
      padding: 20px;
      background: {_PAGE_BG};
      color: #e6e6e6;
    }}
    h1 {{
      margin: 0 0 20px 0;
      font-size: 20px;
      color: #e6e6e6;
      border-bottom: 1px solid #2a2a2a;
      padding-bottom: 12px;
    }}
    .report-container {{
      background: {_PAPER_BG};
      padding: 20px;
      border-radius: 8px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }}
  </style>
</head>
<body>
  <h1>Collectra Evaluation Report</h1>
  <div class="report-container">
    {html_fig}
  </div>
</body>
</html>"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(full_html, encoding="utf-8")
    logger.info("HTML evaluation report written to %s", output_path)
