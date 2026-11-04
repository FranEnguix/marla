"""Visual constants for scenario figures (sizes are nominal: points for text
and lines, inches for geometry, at scale 1).

Restrained by design: neutral greys for structure, one accent (Okabe-Ito
vermillion, colorblind-safe) reserved for sensitive targets -- which are
*also* drawn as a diamond rather than a circle and tagged "target" on
verbose cards, so target status never depends on color alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from matplotlib import font_manager
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties

# Column/text widths of the AAMAS/ACM two-column format, in inches.
PAPER_WIDTHS_IN = {"single": 3.33, "double": 7.0}

# Ships with matplotlib, so it is available wherever MARLA is installed.
DEFAULT_FONT_FAMILY = "DejaVu Sans"


@dataclass(frozen=True)
class ScenarioFigureStyle:
    font_family: str = DEFAULT_FONT_FAMILY

    # Text sizes (pt).
    internet_pt: float = 7.5
    internet_sub_pt: float = 6.0
    subnet_title_pt: float = 7.5
    host_label_pt: float = 6.5
    card_title_pt: float = 6.5
    card_pt: float = 6.0
    firewall_glyph_pt: float = 5.0
    firewall_label_pt: float = 6.0
    legend_pt: float = 6.5
    caption_pt: float = 6.0
    min_font_pt: float = 5.0  # smallest size a scaled-down figure may use

    # Line widths (pt).
    link_lw: float = 0.9
    box_lw: float = 0.6
    card_lw: float = 0.5
    target_card_lw: float = 1.0
    glyph_lw: float = 0.7

    # Colors.
    ink: str = "#1A1A1A"
    muted_ink: str = "#5C5C5C"
    link: str = "#4A4A4A"
    subnet_fill: str = "#F2F2F0"
    subnet_edge: str = "#8C8C8C"
    internet_fill: str = "#FFFFFF"
    internet_edge: str = "#333333"
    host_fill: str = "#FFFFFF"
    host_edge: str = "#333333"
    target_fill: str = "#D55E00"
    target_edge: str = "#6E2E00"
    card_fill: str = "#FFFFFF"
    card_edge: str = "#A6A6A6"
    target_card_edge: str = "#D55E00"
    firewall_fill: str = "#333333"
    firewall_ink: str = "#FFFFFF"
    background: str = "#FFFFFF"

    # Geometry (in).
    margin: float = 0.06
    box_pad: float = 0.08
    box_rounding: float = 0.05
    title_gap: float = 0.07
    host_diameter: float = 0.13
    target_size: float = 0.17  # diamond diagonal
    host_label_gap: float = 0.035
    host_cell_min_w: float = 0.40
    host_row_gap: float = 0.06
    default_columns: int = 6
    card_pad: float = 0.045
    card_col_gap: float = 0.06
    card_gap: float = 0.06
    card_line_factor: float = 1.22  # line pitch / font size
    card_max_value_w: float = 1.7
    verbose_columns: int = 4
    max_layer_w: float = 15.0
    node_gap: float = 0.28
    layer_gap: float = 0.36
    port_spacing: float = 0.16
    firewall_w: float = 0.17
    firewall_h: float = 0.11
    legend_gap: float = 0.14
    legend_entry_gap: float = 0.18
    legend_glyph_w: float = 0.24
    legend_min_w: float = 2.6
    table_col_gap: float = 0.10
    table_gap: float = 0.14
    table_max_w: float = 7.0  # tables sit side by side up to this width


class TextMeasurer:
    """Measures rendered text extents (inches) with the Agg renderer's font
    metrics -- the same FreeType metrics the PDF/SVG/PNG backends use."""

    def __init__(self, family: str = DEFAULT_FONT_FAMILY) -> None:
        self.family = resolve_font_family(family)
        self._renderer = RendererAgg(16, 16, 72)
        self._cache: dict[tuple[str, float, str], tuple[float, float, float]] = {}

    def font(self, size: float, weight: str = "normal") -> FontProperties:
        return FontProperties(family=self.family, size=size, weight=weight)

    def measure(self, text: str, size: float, weight: str = "normal") -> tuple[float, float, float]:
        """``(width, height, descent)`` in inches; ``height`` includes ``descent``."""
        key = (text, size, weight)
        if key not in self._cache:
            w, h, d = self._renderer.get_text_width_height_descent(text, self.font(size, weight), ismath=False)
            self._cache[key] = (w / 72.0, h / 72.0, d / 72.0)
        return self._cache[key]

    def line_metrics(self, size: float, weight: str = "normal") -> tuple[float, float]:
        """Font-level ``(ascent, descent)`` in inches, independent of the
        actual characters, so stacked lines align regardless of content."""
        _, h, d = self.measure("Ag(|", size, weight)
        return h - d, d


@lru_cache(maxsize=None)
def resolve_font_family(family: str) -> str:
    try:
        font_manager.findfont(FontProperties(family=family), fallback_to_default=False)
        return family
    except ValueError:
        return DEFAULT_FONT_FAMILY
