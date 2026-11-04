"""Deterministic geometry for scenario figures (inches, y pointing up).

The layout is computed entirely here -- every box, glyph, edge and text
item with its measured extent -- so the renderer only draws, and tests can
check the geometry (no overlaps, nothing outside the canvas) directly.

**Hierarchical placement.** The internet is the root; every subnet sits on
the layer given by its breadth-first distance from the internet in the
scenario topology, so an attacker's path into the network reads top to
bottom. Within a layer, subnets are ordered by a few deterministic
barycenter sweeps (ties broken by YAML subnet index), which removes
avoidable edge crossings.

**Stable across seeds.** NASimEmu permutes subnet IDs per realization (see
:mod:`marla.environment.scenario_realization`), so placement is keyed by
each subnet's YAML index, never its realized ID, and horizontal slots are
sized for the subnet's *largest* possible host count: the same YAML subnet
lands in the same place in every realization of a scenario, whatever its
ID or sampled size in that seed.

**Edges are subnet-level.** NASimEmu has no host-to-host links: hosts in one
subnet reach each other, and subnets connect through the topology matrix,
each direction filtered by a subnet firewall. Edges therefore join subnet
boxes (or the internet node), never hosts; a link whose firewall does not
permit every scenario service carries a firewall mark.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal

from marla.environment.scenario_realization import RealizedHost, ScenarioRealization, SubnetConnection
from marla.visualization.scenario_style import ScenarioFigureStyle, TextMeasurer

View = Literal["default", "verbose"]
VIEWS: tuple[View, ...] = ("default", "verbose")

NO_PROCESSES = "n/a"
NO_PROCESSES_NOTE = "Processes n/a: the scenario defines no processes."

GROUND_TRUTH_NOTE = (
    "Host attributes are the sampled simulator ground truth for this seed; "
    "the agent does not observe them except through its own scans and exploits."
)


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def shifted(self, dx: float, dy: float) -> Rect:
        return Rect(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)

    def overlaps(self, other: Rect, tol: float = 1e-6) -> bool:
        return (min(self.x1, other.x1) - max(self.x0, other.x0) > tol
                and min(self.y1, other.y1) - max(self.y0, other.y0) > tol)

    def contains(self, other: Rect, tol: float = 1e-6) -> bool:
        return (other.x0 >= self.x0 - tol and other.x1 <= self.x1 + tol
                and other.y0 >= self.y0 - tol and other.y1 <= self.y1 + tol)


@dataclass(frozen=True)
class PlacedText:
    x: float
    y: float  # baseline
    text: str
    size: float
    weight: str
    color: str
    ha: str
    bbox: Rect
    role: str

    def shifted(self, dx: float, dy: float) -> PlacedText:
        return replace(self, x=self.x + dx, y=self.y + dy, bbox=self.bbox.shifted(dx, dy))


@dataclass(frozen=True)
class PlacedBox:
    kind: str  # "internet" | "subnet" | "card"
    rect: Rect
    subnet_id: int | None = None
    address: tuple[int, int] | None = None
    target: bool = False

    def shifted(self, dx: float, dy: float) -> PlacedBox:
        return replace(self, rect=self.rect.shifted(dx, dy))


@dataclass(frozen=True)
class PlacedGlyph:
    kind: str  # "host" | "target" | "firewall"
    rect: Rect
    address: tuple[int, int] | None = None
    role: str = "diagram"  # "diagram" | "legend"
    label: str = ""  # text drawn inside the glyph (firewall marks)

    def shifted(self, dx: float, dy: float) -> PlacedGlyph:
        return replace(self, rect=self.rect.shifted(dx, dy))


@dataclass(frozen=True)
class PlacedEdge:
    points: tuple[tuple[float, float], ...]
    arrow_start: bool = False
    arrow_end: bool = False
    connection: tuple[int, int] | None = None  # realized subnet IDs; None for legend samples

    def shifted(self, dx: float, dy: float) -> PlacedEdge:
        return replace(self, points=tuple((x + dx, y + dy) for x, y in self.points))


@dataclass
class DiagramLayout:
    view: View
    width: float = 0.0
    height: float = 0.0
    boxes: list[PlacedBox] = field(default_factory=list)
    glyphs: list[PlacedGlyph] = field(default_factory=list)
    texts: list[PlacedText] = field(default_factory=list)
    edges: list[PlacedEdge] = field(default_factory=list)
    # YAML subnet index -> (layer, order within layer): the seed-independent placement.
    subnet_slots: dict[int, tuple[int, int]] = field(default_factory=dict)

    @property
    def min_font_pt(self) -> float:
        return min((t.size for t in self.texts), default=0.0)

    def bounds(self) -> Rect:
        rects = [b.rect for b in self.boxes] + [g.rect for g in self.glyphs] + [t.bbox for t in self.texts]
        xs = [x for e in self.edges for x, _ in e.points]
        ys = [y for e in self.edges for _, y in e.points]
        return Rect(
            min([r.x0 for r in rects] + xs), min([r.y0 for r in rects] + ys),
            max([r.x1 for r in rects] + xs), max([r.y1 for r in rects] + ys),
        )

    def shifted(self, dx: float, dy: float) -> None:
        self.boxes = [b.shifted(dx, dy) for b in self.boxes]
        self.glyphs = [g.shifted(dx, dy) for g in self.glyphs]
        self.texts = [t.shifted(dx, dy) for t in self.texts]
        self.edges = [e.shifted(dx, dy) for e in self.edges]


class _Builder:
    def __init__(self, layout: DiagramLayout, style: ScenarioFigureStyle, measurer: TextMeasurer) -> None:
        self.layout = layout
        self.style = style
        self.m = measurer

    def text(self, x: float, y: float, text: str, size: float, *, weight: str = "normal",
             color: str | None = None, ha: str = "left", role: str = "label") -> PlacedText:
        w, h, d = self.m.measure(text, size, weight)
        x0 = {"left": x, "center": x - w / 2, "right": x - w}[ha]
        item = PlacedText(x, y, text, size, weight, color or self.style.ink, ha, Rect(x0, y - d, x0 + w, y - d + h), role)
        self.layout.texts.append(item)
        return item


def _wrap_tokens(text: str, max_w: float, size: float, measurer: TextMeasurer, weight: str = "normal") -> list[str]:
    """Greedy wrap at spaces, then at ``_`` inside an over-long word (service
    names like ``9200_windows_elasticsearch``); never drops characters."""
    if measurer.measure(text, size, weight)[0] <= max_w:
        return [text]
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = f"{current} {word}".strip()
        if measurer.measure(candidate, size, weight)[0] <= max_w:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = ""
        piece = ""
        for chunk in word.split("_"):
            joined = f"{piece}_{chunk}" if piece else chunk
            if piece and measurer.measure(joined, size, weight)[0] > max_w:
                lines.append(piece + "_")
                piece = chunk
            else:
                piece = joined
        current = piece
    if current:
        lines.append(current)
    return lines


# ---------------------------------------------------------------- host cards


@dataclass(frozen=True)
class CardContent:
    address: tuple[int, int]
    target: bool
    rows: tuple[tuple[str, tuple[str, ...]], ...]  # (label, value lines)


def _action_text(name: str, access: str, prob: float) -> str:
    return f"{name} ({access})" if prob >= 1.0 else f"{name} ({access}, p={prob:g})"


def card_contents(realization: ScenarioRealization) -> list[CardContent]:
    """Verbose card rows for every host: the sampled values, never the
    scenario's probabilities. Rows that only some scenarios have (discovery
    value, host firewall) are shown for every host of a scenario that has
    them, so cards keep one shape within a figure."""
    show_discovery = any(h.discovery_value for h in realization.hosts)
    show_host_fw = any(h.firewall_denied for h in realization.hosts)
    cards = []
    for host in realization.hosts:
        rows: list[tuple[str, tuple[str, ...]]] = [
            ("OS", (host.os if host.os is not None else "unknown",)),
            ("Services", host.services or ("none",)),
        ]
        if realization.defines_processes:
            rows.append(("Processes", host.processes or ("none",)))
        else:
            rows.append(("Processes", (NO_PROCESSES,)))
        rows.append(("Value", (f"{host.value:g}",)))
        if show_discovery:
            rows.append(("Discovery", (f"{host.discovery_value:g}",)))
        rows.append(("Exploits", tuple(_action_text(a.name, a.access, a.prob) for a in host.applicable_exploits) or ("none",)))
        rows.append(("Privesc", tuple(_action_text(a.name, a.access, a.prob) for a in host.applicable_privescs) or ("none",)))
        if show_host_fw:
            rows.append(("Host FW", tuple(
                f"denies {', '.join(svcs)} from ({src[0]},{src[1]})" for src, svcs in host.firewall_denied
            ) or ("none",)))
        cards.append(CardContent(host.address, host.sensitive, tuple(rows)))
    return cards


@dataclass(frozen=True)
class _CardMetrics:
    w: float
    label_w: float
    line_h: float
    title_h: float
    pad: float
    wrapped: dict  # (address, row index) -> wrapped lines
    lines: dict  # address -> total value lines

    def height(self, addresses) -> float:
        """Common height of a group of cards (one subnet's)."""
        return 2 * self.pad + self.title_h + max(self.lines[a] for a in addresses) * self.line_h


def _card_metrics(cards: list[CardContent], style: ScenarioFigureStyle, m: TextMeasurer) -> _CardMetrics:
    size = style.card_pt
    label_w = max(m.measure(label, size)[0] for c in cards for label, _ in c.rows)
    wrapped = {}
    value_w = 0.0
    line_counts = {}
    for c in cards:
        n_lines = 0
        for i, (_, values) in enumerate(c.rows):
            lines = [piece for v in values for piece in _wrap_tokens(v, style.card_max_value_w, size, m)]
            wrapped[(c.address, i)] = lines
            n_lines += len(lines)
            value_w = max([value_w] + [m.measure(line, size)[0] for line in lines])
        line_counts[c.address] = n_lines
    tag_w = m.measure("target", style.card_pt)[0] + style.target_size * 0.8 + 0.04
    title_w = max(m.measure(f"({c.address[0]},{c.address[1]})", style.card_title_pt, "bold")[0] for c in cards)
    ascent, descent = m.line_metrics(style.card_title_pt, "bold")
    line_h = size / 72.0 * style.card_line_factor
    title_h = ascent + descent + 0.03
    w = 2 * style.card_pad + max(label_w + style.card_col_gap + value_w, title_w + 0.08 + tag_w)
    return _CardMetrics(w, label_w, line_h, title_h, style.card_pad, wrapped, line_counts)


# ---------------------------------------------------------------- the layout


@dataclass
class _Node:
    source: int
    subnet_id: int
    w: float
    h: float
    slot_w: float
    columns: int
    rect: Rect | None = None


def _layers(realization: ScenarioRealization) -> tuple[list[list[int]], dict[int, set[int]]]:
    source_of = {realized: source for source, realized in enumerate(realization.subnet_permutation)}
    n = len(realization.subnet_permutation)
    adjacency: dict[int, set[int]] = {i: set() for i in range(n)}
    for c in realization.connections:
        i, j = source_of[c.a], source_of[c.b]
        adjacency[i].add(j)
        adjacency[j].add(i)
    depth = {0: 0}
    queue = [0]
    while queue:
        u = queue.pop(0)
        for v in sorted(adjacency[u]):
            if v not in depth:
                depth[v] = depth[u] + 1
                queue.append(v)
    unreached = max(depth.values()) + 1
    for i in range(n):
        depth.setdefault(i, unreached)
    layers = [sorted(i for i in range(n) if depth[i] == layer) for layer in range(max(depth.values()) + 1)]
    layers = [layer for layer in layers if layer]

    def reorder(nodes: list[int], reference: list[int]) -> None:
        ref_pos = {v: k for k, v in enumerate(reference)}
        current = {v: k for k, v in enumerate(nodes)}

        def key(v: int) -> tuple[float, int, int]:
            neighbours = [ref_pos[u] for u in adjacency[v] if u in ref_pos]
            return (sum(neighbours) / len(neighbours) if neighbours else current[v], current[v], v)

        nodes.sort(key=key)

    for _ in range(4):
        for k in range(1, len(layers)):
            reorder(layers[k], layers[k - 1])
        for k in range(len(layers) - 2, 0, -1):
            reorder(layers[k], layers[k + 1])

    def crossings() -> int:
        total = 0
        for upper, lower in zip(layers, layers[1:]):
            pu = {v: i for i, v in enumerate(upper)}
            pl = {v: i for i, v in enumerate(lower)}
            edges = [(pu[u], pl[v]) for u in upper for v in adjacency[u] if v in pl]
            total += sum((a - c) * (b - d) < 0 for i, (a, b) in enumerate(edges) for c, d in edges[i + 1:])
        return total

    # Adjacent swaps that strictly reduce crossings (barycenters are a heuristic).
    best = crossings()
    improved = True
    while improved and best:
        improved = False
        for layer in layers[1:]:
            for i in range(len(layer) - 1):
                layer[i], layer[i + 1] = layer[i + 1], layer[i]
                count = crossings()
                if count < best:
                    best, improved = count, True
                else:
                    layer[i], layer[i + 1] = layer[i + 1], layer[i]
    return layers, adjacency


def _ordered_least_squares(desired: list[float], offsets: list[float]) -> list[float]:
    """Centers ``c`` closest (least squares) to ``desired`` with every gap at
    least the packed one: ``c[i] - c[i-1] >= offsets[i] - offsets[i-1]``.
    Substituting ``y = c - offsets`` makes it isotonic regression, solved
    exactly by pool-adjacent-violators."""
    blocks: list[list[float]] = []  # [mean, weight]
    for target in (d - o for d, o in zip(desired, offsets)):
        blocks.append([target, 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            (m2, w2), (m1, w1) = blocks.pop(), blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2])
    y = [mean for mean, weight in blocks for _ in range(int(weight))]
    return [yi + o for yi, o in zip(y, offsets)]


def compute_layout(
    realization: ScenarioRealization,
    view: View,
    style: ScenarioFigureStyle | None = None,
    measurer: TextMeasurer | None = None,
    caption: tuple[str, ...] = (),
) -> DiagramLayout:
    """The complete, deterministic geometry of one figure (see module docstring)."""
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; expected one of {VIEWS}")
    style = style or ScenarioFigureStyle()
    m = measurer or TextMeasurer(style.font_family)
    layout = DiagramLayout(view=view)
    b = _Builder(layout, style, m)
    verbose = view == "verbose"

    layers, adjacency = _layers(realization)
    subnet_by_source = {s.source_index: s for s in realization.subnets}
    hosts_by_subnet = {s.subnet_id: realization.hosts_in(s.subnet_id) for s in realization.subnets}

    # -- node sizes
    title_asc, title_desc = m.line_metrics(style.subnet_title_pt, "bold")
    title_h = title_asc + title_desc

    def subnet_title(source: int) -> str:
        s = subnet_by_source[source]
        return f"Subnet {s.subnet_id} · {s.label}" if s.label else f"Subnet {s.subnet_id}"

    cards: dict[tuple[int, int], CardContent] = {}
    cm: _CardMetrics | None = None
    default_cell_h = 0.0
    if verbose:
        cards = {c.address: c for c in card_contents(realization)}
        cm = _card_metrics(list(cards.values()), style, m) if cards else None
        cell_w = cm.w if cm else style.host_cell_min_w
        cell_gap_x = cell_gap_y = style.card_gap
        column_cap = style.verbose_columns
    else:
        label_size = style.host_label_pt
        label_w = max([m.measure(f"({h.address[0]},{h.address[1]})", label_size)[0] for h in realization.hosts] or [0.0])
        la, ld = m.line_metrics(label_size)
        cell_w = max(style.host_cell_min_w, label_w + 0.1)
        default_cell_h = style.target_size + style.host_label_gap + la + ld
        cell_gap_x, cell_gap_y = 0.0, style.host_row_gap
        column_cap = style.default_columns

    def grid_w(columns: int) -> float:
        return columns * cell_w + max(columns - 1, 0) * cell_gap_x

    def cell_height(subnet_id: int) -> float:
        hosts = hosts_by_subnet[subnet_id]
        if not hosts:
            return 0.0
        return cm.height([h.address for h in hosts]) if verbose else default_cell_h

    def make_node(source: int, cap: int) -> _Node:
        s = subnet_by_source[source]
        n_hosts = len(hosts_by_subnet[s.subnet_id])
        cell_h = cell_height(s.subnet_id)
        title_w = m.measure(subnet_title(source), style.subnet_title_pt, "bold")[0]
        cols = max(1, min(n_hosts, cap))
        slot_cols = max(1, min(max(s.size_bounds[1], n_hosts), cap))
        rows = math.ceil(n_hosts / cols) if n_hosts else 0
        w = max(title_w, grid_w(cols)) + 2 * style.box_pad
        slot_w = max(title_w, grid_w(slot_cols)) + 2 * style.box_pad
        h = 2 * style.box_pad + title_h + (style.title_gap + rows * cell_h + max(rows - 1, 0) * cell_gap_y if rows else 0)
        return _Node(source, s.subnet_id, w, h, max(w, slot_w), cols)

    nodes: dict[int, _Node] = {}
    for layer in layers[1:]:
        cap = column_cap
        while True:
            for source in layer:
                nodes[source] = make_node(source, cap)
            total = sum(nodes[s].slot_w for s in layer) + style.node_gap * (len(layer) - 1)
            if total <= style.max_layer_w or cap == 1:
                break
            cap -= 1

    internet_lines = ("Internet", "(attacker)")
    ia, idsc = m.line_metrics(style.internet_pt, "bold")
    sa, sd = m.line_metrics(style.internet_sub_pt)
    internet_w = max(m.measure(internet_lines[0], style.internet_pt, "bold")[0],
                     m.measure(internet_lines[1], style.internet_sub_pt)[0]) + 2 * style.box_pad
    internet_h = 2 * 0.055 + ia + idsc + 0.01 + sa + sd

    source_of = {realized: source for source, realized in enumerate(realization.subnet_permutation)}
    layer_of = {source: k for k, layer in enumerate(layers) for source in layer}

    # -- vertical positions
    gaps = [style.layer_gap] * (len(layers) - 1)
    # Extra room under a layer for edges between non-adjacent subnets of that layer.
    same_layer = [c for c in realization.connections if layer_of[source_of[c.a]] == layer_of[source_of[c.b]]]
    routed: dict[int, list[SubnetConnection]] = {}
    for c in same_layer:
        layer = layers[layer_of[source_of[c.a]]]
        if abs(layer.index(source_of[c.a]) - layer.index(source_of[c.b])) > 1:
            routed.setdefault(layer_of[source_of[c.a]], []).append(c)
    for k, conns in routed.items():
        extra = 0.12 * len(conns) + 0.1
        if k < len(gaps):
            gaps[k] += extra
        else:
            gaps.append(extra)

    layer_heights = [internet_h] + [max(nodes[s].h for s in layer) for layer in layers[1:]]
    tops = [0.0]
    for k in range(1, len(layers)):
        tops.append(tops[k - 1] - layer_heights[k - 1] - gaps[k - 1])

    # -- horizontal positions (slots sized for the largest possible subnet)
    internet_rect = Rect(-internet_w / 2, -internet_h, internet_w / 2, 0.0)
    layout.boxes.append(PlacedBox("internet", internet_rect, subnet_id=0))
    b.text(0.0, -0.055 - ia, internet_lines[0], style.internet_pt, weight="bold", ha="center", role="internet")
    b.text(0.0, -0.055 - ia - idsc - 0.01 - sa, internet_lines[1], style.internet_sub_pt, color=style.muted_ink,
           ha="center", role="internet")
    rect_of: dict[int, Rect] = {0: internet_rect}
    slot_x: dict[int, float] = {0: 0.0}
    for k, layer in enumerate(layers[1:], start=1):
        # Wider gap between same-layer neighbours joined by a firewall-marked link.
        between = []
        for left, right in zip(layer, layer[1:]):
            gap = style.node_gap
            for c in same_layer:
                if {source_of[c.a], source_of[c.b]} == {left, right} and c.restricted:
                    gap = max(gap, _firewall_w(b, len(realization.connections), verbose) + 0.14)
            between.append(gap)
        # Offsets of each slot center from the first one when packed tightly.
        offsets = [0.0]
        for idx in range(1, len(layer)):
            offsets.append(offsets[-1] + nodes[layer[idx - 1]].slot_w / 2 + between[idx - 1] + nodes[layer[idx]].slot_w / 2)
        packed_left = -offsets[-1] / 2
        desired = []
        for idx, source in enumerate(layer):
            parents = [slot_x[u] for u in adjacency[source] if u in slot_x and layer_of[u] == k - 1]
            desired.append(sum(parents) / len(parents) if parents else packed_left + offsets[idx])
        centers = _ordered_least_squares(desired, offsets)
        for idx, source in enumerate(layer):
            node = nodes[source]
            slot_x[source] = centers[idx]
            node.rect = Rect(centers[idx] - node.w / 2, tops[k] - node.h, centers[idx] + node.w / 2, tops[k])
            rect_of[source] = node.rect
            layout.subnet_slots[source] = (k, idx)

    # -- subnet boxes, titles and hosts
    for source, node in sorted(nodes.items()):
        r = node.rect
        layout.boxes.append(PlacedBox("subnet", r, subnet_id=node.subnet_id))
        b.text(r.x0 + style.box_pad, r.y1 - style.box_pad - title_asc, subnet_title(source), style.subnet_title_pt,
               weight="bold", role="subnet_title")
        hosts = hosts_by_subnet[node.subnet_id]
        cell_h = cell_height(node.subnet_id)
        grid_top = r.y1 - style.box_pad - title_h - style.title_gap
        for row_start in range(0, len(hosts), node.columns):
            row = hosts[row_start:row_start + node.columns]
            row_index = row_start // node.columns
            x0 = r.cx - grid_w(len(row)) / 2
            y_top = grid_top - row_index * (cell_h + cell_gap_y)
            for col, host in enumerate(row):
                cell = Rect(x0 + col * (cell_w + cell_gap_x), y_top - cell_h, x0 + col * (cell_w + cell_gap_x) + cell_w, y_top)
                if verbose:
                    _place_card(b, cards[host.address], cell, cm, style)
                else:
                    _place_host_glyph(b, host, cell, style)

    # -- edges between subnets
    firewalls = _place_edges(b, realization, layers, layer_of, source_of, rect_of, routed, verbose)

    # -- legend, firewall rules and notes below the diagram
    diagram = layout.bounds()
    _place_legend(b, realization, diagram, verbose, caption, firewalls)

    bounds = layout.bounds()
    layout.shifted(style.margin - bounds.x0, style.margin - bounds.y0)
    layout.width = bounds.w + 2 * style.margin
    layout.height = bounds.h + 2 * style.margin
    return layout


def _place_host_glyph(b: _Builder, host: RealizedHost, cell: Rect, style: ScenarioFigureStyle) -> None:
    glyph_cy = cell.y1 - style.target_size / 2
    size = style.target_size if host.sensitive else style.host_diameter
    rect = Rect(cell.cx - size / 2, glyph_cy - size / 2, cell.cx + size / 2, glyph_cy + size / 2)
    b.layout.glyphs.append(PlacedGlyph("target" if host.sensitive else "host", rect, address=host.address))
    ascent, _ = b.m.line_metrics(style.host_label_pt)
    baseline = cell.y1 - style.target_size - style.host_label_gap - ascent
    b.text(cell.cx, baseline, f"({host.address[0]},{host.address[1]})", style.host_label_pt,
           weight="bold" if host.sensitive else "normal", ha="center", role="host_label")


def _place_card(b: _Builder, card: CardContent, cell: Rect, cm: _CardMetrics, style: ScenarioFigureStyle) -> None:
    b.layout.boxes.append(PlacedBox("card", cell, address=card.address, target=card.target))
    m = b.m
    ascent, _ = m.line_metrics(style.card_title_pt, "bold")
    title_base = cell.y1 - style.card_pad - ascent
    b.text(cell.x0 + style.card_pad, title_base, f"({card.address[0]},{card.address[1]})", style.card_title_pt,
           weight="bold", role="card_title")
    if card.target:
        tag = b.text(cell.x1 - style.card_pad, title_base, "target", style.card_pt, ha="right", role="card_tag")
        size = style.target_size * 0.8
        cy = title_base + m.line_metrics(style.card_pt)[0] * 0.38
        x1 = tag.bbox.x0 - 0.03
        b.layout.glyphs.append(PlacedGlyph("target", Rect(x1 - size, cy - size / 2, x1, cy + size / 2), address=card.address))
    line_ascent, _ = m.line_metrics(style.card_pt)
    y = cell.y1 - style.card_pad - cm.title_h - line_ascent
    for i, (label, _) in enumerate(card.rows):
        b.text(cell.x0 + style.card_pad, y, label, style.card_pt, color=style.muted_ink, role="card_label")
        for line in cm.wrapped[(card.address, i)]:
            b.text(cell.x0 + style.card_pad + cm.label_w + style.card_col_gap, y, line, style.card_pt, role="card_value")
            y -= cm.line_h


def _ports(rect: Rect, others: list[tuple[float, object]], style: ScenarioFigureStyle) -> dict:
    """Spread several edges leaving one box side, ordered by where they go."""
    others = sorted(others, key=lambda t: (t[0], str(t[1])))
    n = len(others)
    if n <= 1:
        return {key: rect.cx for _, key in others}
    spacing = min(style.port_spacing, 0.6 * rect.w / (n - 1))
    return {key: rect.cx + (k - (n - 1) / 2) * spacing for k, (_, key) in enumerate(others)}


def _firewall_w(b: _Builder, n_connections: int, verbose: bool) -> float:
    label = f"FW{n_connections}" if verbose else "FW"
    return max(b.style.firewall_w, b.m.measure(label, b.style.firewall_glyph_pt, "bold")[0] + 0.06)


def _subnet_name(subnet_id: int) -> str:
    return "Internet" if subnet_id == 0 else f"Subnet {subnet_id}"


def firewall_rule_text(c: SubnetConnection) -> str:
    """The services a restricted link's subnet firewall permits, per direction."""
    parts = []
    for src, dst, connected, services in ((c.a, c.b, c.a_to_b, c.services_a_to_b), (c.b, c.a, c.b_to_a, c.services_b_to_a)):
        if connected:
            parts.append(f"{_subnet_name(src)} → {_subnet_name(dst)}: {', '.join(services) if services else 'none'}")
    return ";  ".join(parts)


def _place_edges(b, realization, layers, layer_of, source_of, rect_of, routed, verbose) -> list[tuple[str, SubnetConnection]]:
    """Places every subnet link; returns the firewall marks' labels and links
    in reading order (top to bottom, left to right)."""
    style = b.style
    bottom_users: dict[int, list] = {}
    top_users: dict[int, list] = {}
    plans = []
    routed_ids = {(c.a, c.b) for conns in routed.values() for c in conns}
    for c in realization.connections:
        sa, sb = source_of[c.a], source_of[c.b]
        la, lb = layer_of[sa], layer_of[sb]
        if la != lb:
            upper, lower = (sa, sb) if la < lb else (sb, sa)
            bottom_users.setdefault(upper, []).append((rect_of[lower].cx, (c.a, c.b)))
            top_users.setdefault(lower, []).append((rect_of[upper].cx, (c.a, c.b)))
            plans.append(("vertical", c, upper, lower))
        elif (c.a, c.b) in routed_ids:
            bottom_users.setdefault(sa, []).append((rect_of[sb].cx, (c.a, c.b)))
            bottom_users.setdefault(sb, []).append((rect_of[sa].cx, (c.a, c.b)))
            plans.append(("routed", c, sa, sb))
        else:
            plans.append(("adjacent", c, sa, sb))
    bottom_ports = {s: _ports(rect_of[s], users, style) for s, users in bottom_users.items()}
    top_ports = {s: _ports(rect_of[s], users, style) for s, users in top_users.items()}
    layer_bottom = [min(rect_of[s].y0 for s in layer) for layer in layers]
    realized_of = realization.subnet_permutation

    def directions(c: SubnetConnection, first_id: int) -> tuple[bool, bool]:
        """(first -> other connected, other -> first connected)."""
        return (c.a_to_b, c.b_to_a) if first_id == c.a else (c.b_to_a, c.a_to_b)

    marks: list[tuple[tuple[float, float], SubnetConnection]] = []
    route_level: dict[int, int] = {}
    for kind, c, s1, s2 in plans:
        key = (c.a, c.b)
        if kind == "vertical":
            upper, lower = s1, s2
            px = bottom_ports[upper][key]
            start = [(px, rect_of[upper].y0)]
            # Leave the upper layer vertically first, so a diagonal never
            # clips a taller neighbour in that layer.
            bottom = layer_bottom[layer_of[upper]]
            if rect_of[upper].y0 - bottom > 1e-6:
                start.append((px, bottom))
            end = (top_ports[lower][key], rect_of[lower].y1)
            down, up = directions(c, realized_of[upper])
            b.layout.edges.append(PlacedEdge((*start, end), arrow_start=up and not down, arrow_end=down and not up,
                                             connection=key))
            mid = ((start[-1][0] + end[0]) / 2, (start[-1][1] + end[1]) / 2)
        elif kind == "adjacent":
            left, right = (s1, s2) if rect_of[s1].cx < rect_of[s2].cx else (s2, s1)
            y = min(rect_of[left].y1, rect_of[right].y1) - min(rect_of[left].h, rect_of[right].h) / 2
            p0, p1 = (rect_of[left].x1, y), (rect_of[right].x0, y)
            rightward, leftward = directions(c, realized_of[left])
            b.layout.edges.append(PlacedEdge((p0, p1), arrow_start=leftward and not rightward,
                                             arrow_end=rightward and not leftward, connection=key))
            mid = ((p0[0] + p1[0]) / 2, y)
        else:
            left, right = (s1, s2) if rect_of[s1].cx < rect_of[s2].cx else (s2, s1)
            layer = layer_of[left]
            level = route_level.get(layer, 0)
            route_level[layer] = level + 1
            y = layer_bottom[layer] - 0.1 - 0.12 * level
            pa = (bottom_ports[left][key], rect_of[left].y0)
            pb = (bottom_ports[right][key], rect_of[right].y0)
            rightward, leftward = directions(c, realized_of[left])
            b.layout.edges.append(PlacedEdge((pa, (pa[0], y), (pb[0], y), pb), arrow_start=leftward and not rightward,
                                             arrow_end=rightward and not leftward, connection=key))
            mid = ((pa[0] + pb[0]) / 2, y)
        if c.restricted:
            marks.append((mid, c))

    w = _firewall_w(b, len(realization.connections), verbose)
    labelled = []
    for n, ((cx, cy), c) in enumerate(sorted(marks, key=lambda t: (-round(t[0][1], 6), round(t[0][0], 6))), start=1):
        label = f"FW{n}" if verbose else "FW"
        rect = Rect(cx - w / 2, cy - style.firewall_h / 2, cx + w / 2, cy + style.firewall_h / 2)
        b.layout.glyphs.append(PlacedGlyph("firewall", rect, label=label))
        labelled.append((label, c))
    return labelled


def _place_legend(b: _Builder, realization: ScenarioRealization, diagram: Rect, verbose: bool, caption: tuple[str, ...],
                  firewalls: list[tuple[str, SubnetConnection]]) -> None:
    style = b.style
    m = b.m
    entries: list[tuple[str, str]] = [("host", "Host"), ("target", "Sensitive target")]
    one_way = any(not c.bidirectional for c in realization.connections)
    entries.append(("link", "Subnet link (bidirectional)" if not one_way else "Subnet link"))
    if one_way:
        entries.append(("arrow", "One-way subnet link"))
    if any(c.restricted for c in realization.connections):
        entries.append(("firewall", "Subnet firewall restricts services" + (" (rules below)" if verbose else "")))

    size = style.legend_pt
    ascent, descent = m.line_metrics(size)
    row_h = max(ascent + descent, style.target_size) + 0.04
    max_w = max(diagram.w, style.legend_min_w)
    rows: list[list[tuple[str, str, float]]] = [[]]
    row_w = 0.0
    for kind, label in entries:
        w = style.legend_glyph_w + 0.05 + m.measure(label, size)[0]
        if rows[-1] and row_w + style.legend_entry_gap + w > max_w:
            rows.append([])
            row_w = 0.0
        row_w += (style.legend_entry_gap if rows[-1] else 0.0) + w
        rows[-1].append((kind, label, w))

    y_top = diagram.y0 - style.legend_gap
    for row in rows:
        total = sum(w for _, _, w in row) + style.legend_entry_gap * (len(row) - 1)
        x = diagram.cx - total / 2
        baseline = y_top - row_h / 2 - (ascent - descent) / 2
        mid = baseline + (ascent - descent) / 2
        for kind, label, w in row:
            gx0, gx1 = x, x + style.legend_glyph_w
            gcx = (gx0 + gx1) / 2
            if kind == "host":
                s = style.host_diameter
                b.layout.glyphs.append(PlacedGlyph("host", Rect(gcx - s / 2, mid - s / 2, gcx + s / 2, mid + s / 2), role="legend"))
            elif kind == "target":
                s = style.target_size
                b.layout.glyphs.append(PlacedGlyph("target", Rect(gcx - s / 2, mid - s / 2, gcx + s / 2, mid + s / 2), role="legend"))
            elif kind == "link":
                b.layout.edges.append(PlacedEdge(((gx0, mid), (gx1, mid))))
            elif kind == "arrow":
                b.layout.edges.append(PlacedEdge(((gx0, mid), (gx1, mid)), arrow_end=True))
            elif kind == "firewall":
                fw = _firewall_w(b, 0, False)
                b.layout.edges.append(PlacedEdge(((gx0, mid), (gx1, mid))))
                b.layout.glyphs.append(PlacedGlyph("firewall", Rect(gcx - fw / 2, mid - style.firewall_h / 2,
                                                                    gcx + fw / 2, mid + style.firewall_h / 2),
                                                   role="legend", label="FW"))
            b.text(gx1 + 0.05, baseline, label, size, role="legend")
            x += w + style.legend_entry_gap
        y_top -= row_h

    if verbose and firewalls:
        size = style.firewall_label_pt
        fa, fd = m.line_metrics(size)
        label_w = max(m.measure(label, size, "bold")[0] for label, _ in firewalls)
        rule_w = max(m.measure(firewall_rule_text(c), size)[0] for _, c in firewalls)
        block_w = min(label_w + 0.08 + rule_w, max_w)
        x0 = diagram.cx - block_w / 2
        y_top -= 0.04
        for label, c in firewalls:
            baseline = y_top - fa
            b.text(x0, baseline, label, size, weight="bold", role="firewall_rule")
            for line in _wrap_tokens(firewall_rule_text(c), block_w - label_w - 0.08, size, m):
                b.text(x0 + label_w + 0.08, baseline, line, size, color=style.muted_ink, role="firewall_rule")
                baseline -= fa + fd + 0.015
            y_top = baseline + fa - 0.015

    notes = ([GROUND_TRUTH_NOTE] if verbose else []) + list(caption)
    if verbose and not realization.defines_processes:
        notes.insert(1, NO_PROCESSES_NOTE)
    na, nd = m.line_metrics(style.caption_pt)
    if notes:
        y_top -= 0.03
    for note in notes:
        for line in _wrap_tokens(note, max_w, style.caption_pt, m):
            b.text(diagram.cx, y_top - na, line, style.caption_pt, color=style.muted_ink, ha="center", role="note")
            y_top -= na + nd + 0.02


# ---------------------------------------------------------------- self-check


def _segment_hits_rect(p0: tuple[float, float], p1: tuple[float, float], r: Rect, shrink: float = 0.004) -> bool:
    """Whether segment p0-p1 passes through the interior of ``r`` (Liang-Barsky)."""
    x0, y0, x1, y1 = r.x0 + shrink, r.y0 + shrink, r.x1 - shrink, r.y1 - shrink
    if x0 >= x1 or y0 >= y1:
        return False
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, p0[0] - x0), (dx, x1 - p0[0]), (-dy, p0[1] - y0), (dy, y1 - p0[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def _segments_cross(a0, a1, b0, b1) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    d1, d2 = orient(b0, b1, a0), orient(b0, b1, a1)
    d3, d4 = orient(a0, a1, b0), orient(a0, a1, b1)
    return d1 * d2 < -1e-12 and d3 * d4 < -1e-12


def edge_crossings(layout: DiagramLayout) -> int:
    """Number of proper crossings between diagram edges (shared endpoints don't count)."""
    segments = [(p, q) for e in layout.edges if e.connection is not None for p, q in zip(e.points, e.points[1:])]
    return sum(_segments_cross(*segments[i], *segments[j])
               for i in range(len(segments)) for j in range(i + 1, len(segments)))


def check_layout(layout: DiagramLayout) -> list[str]:
    """Geometric problems that would make a figure unfit for publication:
    anything outside the canvas, overlapping text/glyphs/boxes, text crossing
    a box border, hosts outside their subnet, and edges running through text,
    host glyphs or unrelated boxes. Empty when the figure is clean."""
    issues: list[str] = []
    canvas = Rect(0.0, 0.0, layout.width, layout.height)
    for item, rect in ([(f"text {t.text!r}", t.bbox) for t in layout.texts]
                       + [(f"{g.kind} glyph", g.rect) for g in layout.glyphs]
                       + [(f"{b.kind} box", b.rect) for b in layout.boxes]):
        if not canvas.contains(rect):
            issues.append(f"{item} extends outside the canvas")

    texts = layout.texts
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if texts[i].bbox.overlaps(texts[j].bbox, tol=0.002):
                issues.append(f"text {texts[i].text!r} overlaps text {texts[j].text!r}")
    glyphs = layout.glyphs
    for g in glyphs:
        for t in texts:
            if g.rect.overlaps(t.bbox, tol=0.002):
                issues.append(f"{g.kind} glyph overlaps text {t.text!r}")
    for i in range(len(glyphs)):
        for j in range(i + 1, len(glyphs)):
            if glyphs[i].rect.overlaps(glyphs[j].rect, tol=0.002):
                issues.append(f"{glyphs[i].kind} glyph overlaps {glyphs[j].kind} glyph")

    containers = [b for b in layout.boxes if b.kind in ("internet", "subnet")]
    for i in range(len(containers)):
        for j in range(i + 1, len(containers)):
            if containers[i].rect.overlaps(containers[j].rect):
                issues.append(f"{containers[i].kind} box {containers[i].subnet_id} overlaps "
                              f"{containers[j].kind} box {containers[j].subnet_id}")
    subnet_boxes = {b.subnet_id: b.rect for b in containers}
    cards = [b for b in layout.boxes if b.kind == "card"]
    for card in cards:
        if not subnet_boxes.get(card.address[0], Rect(0, 0, 0, 0)).contains(card.rect):
            issues.append(f"card {card.address} is not inside subnet box {card.address[0]}")
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            if cards[i].rect.overlaps(cards[j].rect):
                issues.append(f"card {cards[i].address} overlaps card {cards[j].address}")
    for g in glyphs:
        if g.address is not None and g.role == "diagram" and not subnet_boxes[g.address[0]].contains(g.rect):
            issues.append(f"host glyph {g.address} is not inside subnet box {g.address[0]}")
    for t in texts:
        for box in layout.boxes:
            if t.bbox.overlaps(box.rect) and not box.rect.contains(t.bbox):
                issues.append(f"text {t.text!r} crosses the border of a {box.kind} box")

    for e in layout.edges:
        ends = set(e.connection) if e.connection is not None else set()
        for p, q in zip(e.points, e.points[1:]):
            for t in texts:
                if _segment_hits_rect(p, q, t.bbox, shrink=0.0):
                    issues.append(f"an edge runs through text {t.text!r}")
            for g in glyphs:
                if g.kind != "firewall" and _segment_hits_rect(p, q, g.rect):
                    issues.append(f"an edge runs through a {g.kind} glyph")
            if e.connection is None:
                continue
            for box in containers:
                if box.subnet_id not in ends and _segment_hits_rect(p, q, box.rect):
                    issues.append(f"edge {e.connection} runs through {box.kind} box {box.subnet_id}")
    return issues
