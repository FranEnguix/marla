"""Drawing and exporting scenario figures.

Draws a :class:`~marla.visualization.scenario_layout.DiagramLayout` with
matplotlib's object API (no pyplot global state) and exports it as PDF
(TrueType fonts embedded, vector), SVG (vector, text kept as editable
``<text>``) and PNG. Exports are byte-for-byte reproducible: creation dates
are omitted and SVG element IDs use a fixed salt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

import marla
from marla.environment.scenario_realization import ScenarioRealization
from marla.visualization.scenario_layout import DiagramLayout, View, compute_layout
from marla.visualization.scenario_style import PAPER_WIDTHS_IN, ScenarioFigureStyle, TextMeasurer

FORMATS = ("pdf", "svg", "png")
DEFAULT_DPI = 300
MIN_DPI = 72

_EXPORT_RC = {
    "pdf.fonttype": 42,  # embed TrueType: selectable/editable text in the PDF
    "ps.fonttype": 42,
    "svg.fonttype": "none",  # keep text as <text> elements
    "svg.hashsalt": "marla-scenario-figure",
    "text.usetex": False,
    "path.simplify": False,
}


class FigureTooSmallError(ValueError):
    """Fitting the figure to the requested width would make text unreadably small."""


@dataclass(frozen=True)
class RenderedFigure:
    figure: Figure
    layout: DiagramLayout
    width_in: float
    height_in: float
    scale: float


def resolve_width(size: str | float | None) -> float | None:
    """``natural``/``None`` -> no fixed width; ``single``/``double`` -> the
    paper's column/text width; a number -> that many inches."""
    if size is None or size == "natural":
        return None
    if isinstance(size, str) and size in PAPER_WIDTHS_IN:
        return PAPER_WIDTHS_IN[size]
    try:
        width = float(size)
    except (TypeError, ValueError):
        raise ValueError(f"unknown figure size {size!r}; use natural, single, double or a width in inches") from None
    if width <= 0:
        raise ValueError(f"figure width must be positive, got {width}")
    return width


def render_layout(layout: DiagramLayout, style: ScenarioFigureStyle, width: float | None = None) -> RenderedFigure:
    """Draws ``layout``. With ``width``, the figure is exactly that wide: the
    content is scaled down to fit (never up) and centered; text may not
    shrink below ``style.min_font_pt``."""
    scale = 1.0
    fig_w = layout.width
    if width is not None:
        scale = min(1.0, width / layout.width)
        if layout.min_font_pt * scale < style.min_font_pt - 1e-9:
            raise FigureTooSmallError(
                f"fitting this {layout.width:.2f} in wide figure into {width:.2f} in would shrink text to "
                f"{layout.min_font_pt * scale:.1f} pt (< {style.min_font_pt} pt); use a wider size or 'natural'"
            )
        fig_w = width
    fig_h = layout.height * scale
    data_w = fig_w / scale
    pad_x = (data_w - layout.width) / 2

    fig = Figure(figsize=(fig_w, fig_h), facecolor=style.background)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(-pad_x, layout.width + pad_x)
    ax.set_ylim(0, layout.height)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(style.background)

    measurer = TextMeasurer(style.font_family)

    def lw(points: float) -> float:
        return points * scale

    for box in layout.boxes:
        r = box.rect
        if box.kind == "subnet":
            face, edge, width_pt, z = style.subnet_fill, style.subnet_edge, style.box_lw, 1
        elif box.kind == "internet":
            face, edge, width_pt, z = style.internet_fill, style.internet_edge, style.box_lw, 1
        else:
            face = style.card_fill
            edge = style.target_card_edge if box.target else style.card_edge
            width_pt = style.target_card_lw if box.target else style.card_lw
            z = 3
        rounding = min(style.box_rounding, r.w / 4, r.h / 4) if box.kind != "card" else min(0.03, r.h / 4)
        ax.add_patch(FancyBboxPatch((r.x0, r.y0), r.w, r.h, boxstyle=f"round,pad=0,rounding_size={rounding}",
                                    facecolor=face, edgecolor=edge, linewidth=lw(width_pt), zorder=z))

    for edge in layout.edges:
        xs = [p[0] for p in edge.points]
        ys = [p[1] for p in edge.points]
        if edge.arrow_end or edge.arrow_start:
            if len(edge.points) > 2:
                ax.plot(xs[:-1] if edge.arrow_end else xs[1:], ys[:-1] if edge.arrow_end else ys[1:],
                        color=style.link, linewidth=lw(style.link_lw), solid_capstyle="butt", zorder=2)
            start, end = (edge.points[-2], edge.points[-1]) if edge.arrow_end else (edge.points[1], edge.points[0])
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7 * scale, shrinkA=0, shrinkB=0,
                                         color=style.link, linewidth=lw(style.link_lw), zorder=2))
        else:
            ax.plot(xs, ys, color=style.link, linewidth=lw(style.link_lw), solid_capstyle="butt", zorder=2)

    for glyph in layout.glyphs:
        r = glyph.rect
        if glyph.kind == "host":
            ax.add_patch(Circle((r.cx, r.cy), r.w / 2, facecolor=style.host_fill, edgecolor=style.host_edge,
                                linewidth=lw(style.glyph_lw), zorder=4))
        elif glyph.kind == "target":
            ax.add_patch(Polygon([(r.cx, r.y1), (r.x1, r.cy), (r.cx, r.y0), (r.x0, r.cy)], closed=True,
                                 facecolor=style.target_fill, edgecolor=style.target_edge,
                                 linewidth=lw(style.glyph_lw), joinstyle="miter", zorder=4))
        elif glyph.kind == "firewall":
            ax.add_patch(FancyBboxPatch((r.x0, r.y0), r.w, r.h, boxstyle="round,pad=0,rounding_size=0.015",
                                        facecolor=style.firewall_fill, edgecolor=style.firewall_fill,
                                        linewidth=lw(0.3), zorder=5))
            ax.text(r.cx, r.cy, glyph.label, fontproperties=measurer.font(style.firewall_glyph_pt * scale, "bold"),
                    color=style.firewall_ink, ha="center", va="center_baseline", zorder=6)

    for t in layout.texts:
        ax.text(t.x, t.y, t.text, fontproperties=measurer.font(t.size * scale, t.weight), color=t.color,
                ha=t.ha, va="baseline", zorder=7)

    return RenderedFigure(fig, layout, fig_w, fig_h, scale)


def build_figure(
    realization: ScenarioRealization,
    view: View,
    *,
    size: str | float | None = "natural",
    caption: tuple[str, ...] = (),
    style: ScenarioFigureStyle | None = None,
) -> RenderedFigure:
    style = style or ScenarioFigureStyle()
    measurer = TextMeasurer(style.font_family)
    layout = compute_layout(realization, view, style, measurer, caption=caption)
    return render_layout(layout, style, resolve_width(size))


def export_figure(rendered: RenderedFigure, stem: Path, formats: tuple[str, ...] = FORMATS, dpi: int = DEFAULT_DPI) -> list[Path]:
    """Writes ``<stem>.<format>`` for every format; returns the paths."""
    unknown = [f for f in formats if f not in FORMATS]
    if unknown:
        raise ValueError(f"unsupported format(s) {unknown}; choose from {FORMATS}")
    if dpi < MIN_DPI:
        raise ValueError(f"dpi must be at least {MIN_DPI}, got {dpi}")
    creator = f"MARLA {marla.__version__} scenario plot (matplotlib {matplotlib.__version__})"
    metadata = {
        "pdf": {"Creator": creator, "Producer": creator, "CreationDate": None, "ModDate": None},
        "svg": {"Creator": creator, "Date": None},
        "png": {"Software": creator},
    }
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    with matplotlib.rc_context(_EXPORT_RC):
        for fmt in formats:
            path = stem.with_name(f"{stem.name}.{fmt}")
            rendered.figure.savefig(path, format=fmt, dpi=dpi, facecolor=rendered.figure.get_facecolor(),
                                    metadata=metadata[fmt])
            written.append(path)
    return written
