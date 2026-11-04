"""Deterministic geometry for scenario figures (inches, y pointing up).

The layout is computed entirely here -- every box, glyph, edge and text
item with its measured extent -- so the renderer only draws, and tests can
check the geometry (no overlaps, nothing outside the canvas) directly.

One topology engine serves both figure kinds; only what goes inside the
subnet boxes (and the legend, notes and tables explaining it) differs:
a seeded realization's hosts (:class:`_RealizationContent`) or a scenario
definition's host-count ranges and probabilities (:class:`_DefinitionContent`).

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
from marla.scenario.definition import SENSITIVE_SERVICE_ON_NON_TARGET, ScenarioDefinition, SubnetDefinition
from marla.visualization.scenario_style import ScenarioFigureStyle, TextMeasurer

View = Literal["simplified", "verbose"]
VIEWS: tuple[View, ...] = ("simplified", "verbose")

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
    model: ScenarioRealization | ScenarioDefinition,
    view: View,
    style: ScenarioFigureStyle | None = None,
    measurer: TextMeasurer | None = None,
    caption: tuple[str, ...] = (),
) -> DiagramLayout:
    """The complete, deterministic geometry of one figure (see module
    docstring) of a seeded realization or of a scenario definition."""
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; expected one of {VIEWS}")
    style = style or ScenarioFigureStyle()
    m = measurer or TextMeasurer(style.font_family)
    layout = DiagramLayout(view=view)
    b = _Builder(layout, style, m)
    verbose = view == "verbose"
    if isinstance(model, ScenarioDefinition):
        content: _Content = _DefinitionContent(model, verbose, style, m)
    else:
        content = _RealizationContent(model, verbose, style, m)

    layers, adjacency = _layers(model)
    subnet_by_source = {s.source_index: s for s in model.subnets}

    # -- node sizes
    title_asc, title_desc = m.line_metrics(style.subnet_title_pt, "bold")
    title_h = title_asc + title_desc

    def subnet_title(source: int) -> str:
        s = subnet_by_source[source]
        return f"Subnet {s.subnet_id} · {s.label}" if s.label else f"Subnet {s.subnet_id}"

    def make_node(source: int, cap: int) -> _Node:
        title_w = m.measure(subnet_title(source), style.subnet_title_pt, "bold")[0]
        inner_w, inner_h, slot_inner_w, cols = content.node(source, cap)
        w = max(title_w, inner_w) + 2 * style.box_pad
        slot_w = max(title_w, slot_inner_w) + 2 * style.box_pad
        h = 2 * style.box_pad + title_h + (style.title_gap + inner_h if inner_h else 0.0)
        return _Node(source, subnet_by_source[source].subnet_id, w, h, max(w, slot_w), cols)

    nodes: dict[int, _Node] = {}
    for layer in layers[1:]:
        cap = content.column_cap
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

    source_of = {realized: source for source, realized in enumerate(model.subnet_permutation)}
    layer_of = {source: k for k, layer in enumerate(layers) for source in layer}

    # -- vertical positions
    gaps = [style.layer_gap] * (len(layers) - 1)
    # Extra room under a layer for edges between non-adjacent subnets of that layer.
    same_layer = [c for c in model.connections if layer_of[source_of[c.a]] == layer_of[source_of[c.b]]]
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
                    gap = max(gap, _firewall_w(b, len(model.connections), verbose) + 0.14)
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

    # -- subnet boxes, titles and contents
    for source, node in sorted(nodes.items()):
        r = node.rect
        layout.boxes.append(PlacedBox("subnet", r, subnet_id=node.subnet_id))
        b.text(r.x0 + style.box_pad, r.y1 - style.box_pad - title_asc, subnet_title(source), style.subnet_title_pt,
               weight="bold", role="subnet_title")
        content.place(b, source, node, r.y1 - style.box_pad - title_h - style.title_gap)

    # -- edges between subnets
    firewalls = _place_edges(b, model, layers, layer_of, source_of, rect_of, routed, verbose)

    # -- legend, firewall rules, tables and notes below the diagram
    diagram = layout.bounds()
    _place_legend(b, model, diagram, verbose, caption, firewalls, content)

    bounds = layout.bounds()
    layout.shifted(style.margin - bounds.x0, style.margin - bounds.y0)
    layout.width = bounds.w + 2 * style.margin
    layout.height = bounds.h + 2 * style.margin
    return layout


class _Content:
    """What goes inside the subnet boxes, and the legend/notes/tables that
    explain it; the topology around it is shared by every figure type."""

    column_cap: int = 1

    def node(self, source: int, cap: int) -> tuple[float, float, float, int]:
        """(content width, content height, width of the subnet's largest
        possible content, columns) for subnet ``source``."""
        raise NotImplementedError

    def place(self, b: _Builder, source: int, node: _Node, top: float) -> None:
        raise NotImplementedError

    def legend_entries(self) -> list[tuple[str, str]]:
        raise NotImplementedError

    def notes(self) -> list[str]:
        return []

    def tables(self) -> list[_Table]:
        return []


class _RealizationContent(_Content):
    """Hosts of a seeded realization: glyphs (simplified) or cards (verbose)."""

    def __init__(self, r: ScenarioRealization, verbose: bool, style: ScenarioFigureStyle, m: TextMeasurer) -> None:
        self.r, self.verbose, self.style, self.m = r, verbose, style, m
        self.hosts = {s.source_index: r.hosts_in(s.subnet_id) for s in r.subnets}
        self.bounds = {s.source_index: s.size_bounds for s in r.subnets}
        self.cards: dict[tuple[int, int], CardContent] = {}
        self.cm: _CardMetrics | None = None
        if verbose:
            self.cards = {c.address: c for c in card_contents(r)}
            self.cm = _card_metrics(list(self.cards.values()), style, m) if self.cards else None
            self.cell_w = self.cm.w if self.cm else style.host_cell_min_w
            self.gap_x = self.gap_y = style.card_gap
            self.column_cap = style.verbose_columns
        else:
            label_w = max([m.measure(f"({h.address[0]},{h.address[1]})", style.host_label_pt)[0] for h in r.hosts] or [0.0])
            la, ld = m.line_metrics(style.host_label_pt)
            self.cell_w = max(style.host_cell_min_w, label_w + 0.1)
            self.glyph_cell_h = style.target_size + style.host_label_gap + la + ld
            self.gap_x, self.gap_y = 0.0, style.host_row_gap
            self.column_cap = style.default_columns

    def _grid_w(self, columns: int) -> float:
        return columns * self.cell_w + max(columns - 1, 0) * self.gap_x

    def _cell_h(self, source: int) -> float:
        hosts = self.hosts[source]
        if not hosts:
            return 0.0
        return self.cm.height([h.address for h in hosts]) if self.verbose else self.glyph_cell_h

    def node(self, source, cap):
        n_hosts = len(self.hosts[source])
        cols = max(1, min(n_hosts, cap))
        slot_cols = max(1, min(max(self.bounds[source][1], n_hosts), cap))
        rows = math.ceil(n_hosts / cols) if n_hosts else 0
        h = rows * self._cell_h(source) + max(rows - 1, 0) * self.gap_y
        return self._grid_w(cols) if n_hosts else 0.0, h, self._grid_w(slot_cols), cols

    def place(self, b, source, node, top):
        hosts = self.hosts[source]
        cell_h = self._cell_h(source)
        r = node.rect
        for row_start in range(0, len(hosts), node.columns):
            row = hosts[row_start:row_start + node.columns]
            x0 = r.cx - self._grid_w(len(row)) / 2
            y_top = top - (row_start // node.columns) * (cell_h + self.gap_y)
            for col, host in enumerate(row):
                cx0 = x0 + col * (self.cell_w + self.gap_x)
                cell = Rect(cx0, y_top - cell_h, cx0 + self.cell_w, y_top)
                if self.verbose:
                    _place_card(b, self.cards[host.address], cell, self.cm, self.style)
                else:
                    _place_host_glyph(b, host, cell, self.style)

    def legend_entries(self):
        return [("host", "Host"), ("target", "Sensitive target")]

    def notes(self):
        if not self.verbose:
            return []
        return [GROUND_TRUTH_NOTE] + ([] if self.r.defines_processes else [NO_PROCESSES_NOTE])


def _pct(p: float) -> str:
    """Probabilities as percentages, exact when they are round."""
    value = 100 * p
    if abs(value - round(value)) < 1e-9:
        return f"{round(value)}%"
    return f"{value:.1f}%"


def _counted(bounds: tuple[int | None, int | None], noun: str) -> str:
    text = _count_range(*bounds)
    return f"{text} {noun}" + ("" if text == "1" else "s")


def _count_range(lo: int | None, hi: int | None) -> str:
    if lo is None or hi is None:
        return "unspecified"
    return f"{lo}" if lo == hi else f"{lo}–{hi}"


class _DefinitionContent(_Content):
    """A scenario definition: host-count and target ranges (simplified), plus
    every generation probability (verbose)."""

    def __init__(self, d: ScenarioDefinition, verbose: bool, style: ScenarioFigureStyle, m: TextMeasurer) -> None:
        self.d, self.verbose, self.style, self.m = d, verbose, style, m
        self.subnets = {s.source_index: s for s in d.subnets}
        self.rows = {idx: self._rows(s) for idx, s in self.subnets.items()}
        size = style.card_pt if verbose else style.host_label_pt
        self.size = size
        ascent, descent = m.line_metrics(size)
        self.line_h = size / 72.0 * style.card_line_factor
        self.ascent = ascent
        self.marker = style.target_size * 0.75
        self.indent = self.marker + 0.05
        if verbose:
            self.label_w = max((m.measure(label, size)[0] for rows in self.rows.values() for label, _ in rows), default=0.0)
        self.wrapped = {idx: [(label, [piece for v in values for piece in _wrap_tokens(v, style.card_max_value_w, size, m)])
                              for label, values in rows] for idx, rows in self.rows.items()}

    def _target_values(self, s: SubnetDefinition) -> list[str]:
        if self.d.format == "v1":
            return [f"({a},{h}), value {v:g}" for (a, h), v in s.explicit_targets] or ["none"]
        p = s.target_probability
        if p is None:
            return ["unspecified"]
        if p <= 0.0:
            return ["none (0% per host)"] if self.verbose else ["none"]
        count = _count_range(s.targets_min, s.targets_max)
        if not self.verbose:
            return [count]
        if p >= 1.0:
            return ["every host (100%)", f"→ {count} per network"]
        return [f"each host {_pct(p)}, independently", f"→ {count} per network"]

    def _rows(self, s: SubnetDefinition) -> list[tuple[str, list[str]]]:
        hosts = _count_range(s.size_min, s.size_max)
        if not self.verbose:
            targets = self._target_values(s)[0]
            return [("Hosts", [hosts]), ("Targets", [targets])]
        host_values = [hosts]
        if s.size_min is not None and s.size_max is not None and s.size_max > s.size_min:
            host_values.append(f"uniform, {_pct(1 / (s.size_max - s.size_min + 1))} each")
        if s.size_note:
            host_values.append(s.size_note)
        rows = [("Hosts", host_values), ("Targets", self._target_values(s))]
        for service, presence in s.sensitive_service_presence:
            rows.append(("Sensitive svc", [f"{service}: {_pct(presence)} of hosts"]))
        for host in s.static_hosts:
            values = [host.os or "OS unspecified", "services: " + (", ".join(host.services) or "none")]
            if self.d.defines_processes:
                values.append("processes: " + (", ".join(host.processes) or "none"))
            values += [f"denies {', '.join(svcs)} from ({src[0]},{src[1]})" for src, svcs in host.firewall_denied]
            rows.append((f"({host.address[0]},{host.address[1]})", values))
        return rows

    def _has_targets(self, s: SubnetDefinition) -> bool:
        return bool(s.explicit_targets) or bool(s.target_probability and s.targets_max)

    def node(self, source, cap):
        rows = self.wrapped[source]
        n_lines = sum(len(values) for _, values in rows)
        if self.verbose:
            w = self.indent + self.label_w + self.style.card_col_gap + max(
                self.m.measure(v, self.size)[0] for _, values in rows for v in values)
        else:
            w = self.indent + max(self.m.measure(f"{label}: {values[0]}", self.size)[0] for label, values in rows)
        h = n_lines * self.line_h
        return w, h, w, 1

    def place(self, b, source, node, top):
        s = self.subnets[source]
        x0 = node.rect.x0 + self.style.box_pad
        y = top - self.ascent
        for label, values in self.wrapped[source]:
            if label == "Targets" and self._has_targets(s):
                cy = y + self.ascent * 0.38
                b.layout.glyphs.append(PlacedGlyph("target", Rect(x0, cy - self.marker / 2, x0 + self.marker, cy + self.marker / 2)))
            if self.verbose:
                b.text(x0 + self.indent, y, label, self.size, color=self.style.muted_ink, role="subnet_label")
                for value in values:
                    b.text(x0 + self.indent + self.label_w + self.style.card_col_gap, y, value, self.size, role="subnet_value")
                    y -= self.line_h
                continue
            b.text(x0 + self.indent, y, f"{label}: {values[0]}", self.size, role="subnet_value")
            y -= self.line_h

    def legend_entries(self):
        return [("target", "Subnet can hold sensitive targets")]

    def notes(self):
        d = self.d
        notes = []
        if self.verbose:
            what = ("ranges and probabilities of NASimEmu's generator, read from the YAML" if d.randomized_hosts or
                    d.format == "v2" else "a static scenario: every host is listed in the YAML")
            notes.append(f"Scenario definition, not a realization: {what}. Subnet numbers are YAML indices; every "
                         "realization randomly permutes the non-internet subnet IDs.")
            summary = (f"Per network: {_counted(d.host_count_range, 'host')}, "
                       f"{_counted((d.targets_min, d.targets_max), 'target')}")
            if d.p_zero_targets is not None and d.format == "v2":
                summary += f"; P(no target) = {_pct(d.p_zero_targets)}"
                if d.p_zero_targets > 0:
                    summary += " (MARLA regenerates such draws: its networks have at least 1 target)"
            notes.append(summary + ".")
        else:
            notes.append("Host and target counts are ranges over every network the scenario can generate.")
            if d.format == "v2" and d.p_zero_targets:
                notes.append(f"MARLA regenerates the {_pct(d.p_zero_targets)} of draws without a target.")
        notes += [f"Warning: {w}" for w in d.warnings]
        return notes

    def tables(self):
        if not self.verbose:
            return []
        tables = []
        d = self.d
        if d.randomized_hosts:
            rows: list[tuple[str, ...]] = []
            for i, model in enumerate(d.host_model):
                rows.append(("OS" if i == 0 else "", model.os, _pct(model.probability)))
            for model in d.host_model:
                n = len(model.service_candidates)
                k = _count_range(1, n) if n else "0"
                for j, service in enumerate(model.service_candidates):
                    label = f"Services | {model.os}" if j == 0 else (f"  {k} of {n}" if j == 1 else "")
                    rows.append((label, service, _pct(model.service_presence)))
                if n == 1:
                    rows.append((f"  {k} of {n}", "", ""))
            for service in d.sensitive_services:
                oses = [m.os for m in d.host_model if service in m.sensitive_services]
                share = {len(m.sensitive_services) for m in d.host_model if service in m.sensitive_services}
                per = 1 / share.pop() if len(share) == 1 else None
                p_target = _pct(per) if per is not None else "1/|sensitive services|"
                p_other = _pct(SENSITIVE_SERVICE_ON_NON_TARGET * per) if per is not None else "10%/|sensitive services|"
                rows.append(("Sensitive service", service, f"target {p_target}, other {p_other}"))
                if len(oses) < len(d.host_model):
                    rows.append(("", f"(on {', '.join(oses)} only)", ""))
            if d.defines_processes:
                for model in d.host_model:
                    for j, process in enumerate(model.process_candidates):
                        rows.append((f"Processes | {model.os}" if j == 0 else "", process, _pct(model.process_presence)))
            else:
                rows.append(("Processes", "none defined", "–"))
            rows.append(("Value", "target 100, other 0", ""))
            notes = []
            for model in d.host_model:
                if model.service_count_distribution:
                    dist = ", ".join(_pct(p) for _, p in model.service_count_distribution)
                    notes.append(f"{model.os}: k = {_count_range(1, len(model.service_candidates))} with P = {dist}")
            footnotes = [
                "Every host draws its OS uniformly; given the OS it gets k of the n non-sensitive services for that OS "
                "(a uniformly random subset, P(k) ∝ 0.5^k), so service presences are not independent: "
                + "; ".join(notes) + ". Percentages are P(present | OS).",
            ]
            tables.append(_Table("Host attributes (NASimEmu _random generator, every host)",
                                 ("Attribute", "Value", "Probability"), rows, ("l", "l", "r"), footnotes))
        rows = []
        for a in d.actions:
            requires = f"{a.requires}" if a.requires else "–"
            rows.append((a.name, requires, a.os or "any", f"{a.prob:g}", a.access, f"{a.cost:g}"))
        footnotes = []
        if d.scan_costs:
            footnotes.append("Scan costs: " + ", ".join(f"{label} {cost:g}" for label, cost in d.scan_costs) + ".")
        footnotes.append("Requires: the service (exploits) or process (privilege escalations) the target host must run "
                         "(–: none); p: success probability once the preconditions hold; Grants: access level gained.")
        tables.append(_Table("Actions", ("Action", "Requires", "OS", "p", "Grants", "Cost"), rows,
                             ("l", "l", "l", "r", "l", "r"), footnotes))
        return tables


@dataclass(frozen=True)
class _Table:
    title: str
    header: tuple[str, ...]
    rows: list[tuple[str, ...]]
    align: tuple[str, ...]
    footnotes: list[str]


def _table_geometry(b: _Builder, table: _Table, max_w: float) -> tuple[float, float, list[float]]:
    style, m = b.style, b.m
    size = style.card_pt
    widths = [max([m.measure(table.header[i], size, "bold")[0]] + [m.measure(row[i], size)[0] for row in table.rows])
              for i in range(len(table.header))]
    w = sum(widths) + style.table_col_gap * (len(widths) - 1)
    title_w = m.measure(table.title, style.legend_pt, "bold")[0]
    inner_w = max(w, title_w)
    line_h = size / 72.0 * style.card_line_factor
    foot_lines = sum(len(_wrap_tokens(f, min(inner_w, max_w), style.caption_pt, m)) for f in table.footnotes)
    ta, td = m.line_metrics(style.legend_pt, "bold")
    fa, fd = m.line_metrics(style.caption_pt)
    h = (ta + td + 0.05) + line_h * (len(table.rows) + 1) + 0.04 + foot_lines * (fa + fd + 0.015)
    return inner_w + 2 * style.card_pad, h + 2 * style.card_pad, widths


def _place_table(b: _Builder, table: _Table, x0: float, y_top: float, max_w: float) -> Rect:
    style, m = b.style, b.m
    w, h, widths = _table_geometry(b, table, max_w)
    rect = Rect(x0, y_top - h, x0 + w, y_top)
    b.layout.boxes.append(PlacedBox("panel", rect))
    size = style.card_pt
    line_h = size / 72.0 * style.card_line_factor
    ta, td = m.line_metrics(style.legend_pt, "bold")
    ascent, _ = m.line_metrics(size)
    x = x0 + style.card_pad
    y = y_top - style.card_pad - ta
    b.text(x, y, table.title, style.legend_pt, weight="bold", role="table_title")
    y -= td + 0.05 + ascent
    lefts = []
    cursor = x
    for width in widths:
        lefts.append(cursor)
        cursor += width + style.table_col_gap
    for row, weight, color in [(table.header, "bold", style.muted_ink)] + [(r, "normal", style.ink) for r in table.rows]:
        for i, cell in enumerate(row):
            if not cell:
                continue
            if table.align[i] == "r":
                b.text(lefts[i] + widths[i], y, cell, size, weight=weight, color=color, ha="right", role="table_cell")
            else:
                b.text(lefts[i], y, cell, size, weight=weight, color=color, role="table_cell")
        y -= line_h
    fa, fd = m.line_metrics(style.caption_pt)
    y += line_h - ascent - 0.04
    inner_w = w - 2 * style.card_pad
    for footnote in table.footnotes:
        for line in _wrap_tokens(footnote, min(inner_w, max_w), style.caption_pt, m):
            b.text(x, y - fa, line, style.caption_pt, color=style.muted_ink, role="table_note")
            y -= fa + fd + 0.015
    return rect


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


def _place_legend(b: _Builder, model, diagram: Rect, verbose: bool, caption: tuple[str, ...],
                  firewalls: list[tuple[str, SubnetConnection]], content: _Content) -> None:
    style = b.style
    m = b.m
    entries: list[tuple[str, str]] = content.legend_entries()
    one_way = any(not c.bidirectional for c in model.connections)
    entries.append(("link", "Subnet link (bidirectional)" if not one_way else "Subnet link"))
    if one_way:
        entries.append(("arrow", "One-way subnet link"))
    if any(c.restricted for c in model.connections):
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

    tables = content.tables()
    if tables:
        sizes = [_table_geometry(b, t, max(max_w, style.table_max_w))[:2] for t in tables]
        side_by_side = sum(w for w, _ in sizes) + style.table_gap * (len(sizes) - 1) <= max(max_w, style.table_max_w)
        y_top -= 0.08
        if side_by_side:
            x = diagram.cx - (sum(w for w, _ in sizes) + style.table_gap * (len(sizes) - 1)) / 2
            for t, (w, h) in zip(tables, sizes):
                _place_table(b, t, x, y_top, max(max_w, style.table_max_w))
                x += w + style.table_gap
            y_top -= max(h for _, h in sizes)
        else:
            for t, (w, h) in zip(tables, sizes):
                _place_table(b, t, diagram.cx - w / 2, y_top, max(max_w, style.table_max_w))
                y_top -= h + style.table_gap
            y_top += style.table_gap
        max_w = max(max_w, sum(w for w, _ in sizes) + style.table_gap * (len(sizes) - 1) if side_by_side
                    else max(w for w, _ in sizes))
        y_top -= 0.05

    notes = content.notes() + list(caption)
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

    containers = [b for b in layout.boxes if b.kind in ("internet", "subnet", "panel")]
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
