"""Scenario *definition* figures (``marla scenario plot SCENARIO`` without a
seed): read from the YAML only, host-count ranges, generation probabilities
that match NASimEmu's generator, and the simplified/verbose split."""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from marla.cli import app
from marla.environment.scenario_realization import ScenarioRealizer
from marla.scenario.definition import load_scenario_definition, skew_count_distribution
from marla.visualization.scenario_figures import MANIFEST_SCHEMA, load_manifest, plot_definitions
from marla.visualization.scenario_layout import check_layout, compute_layout
from marla.visualization.scenario_plot import build_figure, export_figure

REPO = Path(__file__).resolve().parent.parent
DIAGNOSTIC = REPO / "src" / "marla" / "scenarios" / "diagnostic"
SCENARIO_A = REPO / "tests" / "data" / "scenarios" / "sm_entry_dmz_one_target.v2.yaml"
SCENARIO_B = DIAGNOSTIC / "evidence_b_pivot_one_target.v2.yaml"
SCENARIO_B2 = DIAGNOSTIC / "evidence_b2_two_hosts.v2.yaml"
SCENARIO_C = DIAGNOSTIC / "evidence_c_multi_target.v2.yaml"
MEDIUM = REPO / "src" / "marla" / "scenarios" / "solvable" / "md_entry_user_three_subnets.solvable.v2.yaml"
STATIC_SMALL = REPO / "NASimEmu" / "src" / "nasimemu" / "nasim" / "scenarios" / "benchmark" / "small.yaml"
ALL_BUNDLED = sorted((REPO / "src" / "marla" / "scenarios").glob("*/*.yaml")) + [SCENARIO_A, STATIC_SMALL]

runner = CliRunner()


def _texts(layout, roles=None):
    return [t.text for t in layout.texts if roles is None or t.role in roles]


def _subnet_rows(layout, subnet_id):
    box = next(b for b in layout.boxes if b.kind == "subnet" and b.subnet_id == subnet_id)
    return [t.text for t in sorted((t for t in layout.texts if box.rect.contains(t.bbox) and t.role != "subnet_title"),
                                   key=lambda t: (-round(t.y, 6), t.x))]


def _write_scenario(tmp_path: Path, edit) -> Path:
    raw = yaml.safe_load(SCENARIO_C.read_text())
    edit(raw)
    path = tmp_path / "edited.v2.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


# ------------------------------------------------------ no generation at all


def test_definition_never_generates_a_network(monkeypatch, tmp_path):
    import nasimemu.nasim.scenarios.loader_v2 as loader_v2

    from marla.environment import nasimemu_adapter

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the definition path must not generate a network")

    monkeypatch.setattr(loader_v2.ScenarioLoaderV2, "load", forbidden)
    monkeypatch.setattr(nasimemu_adapter.NasimEmuAdapter, "reset", forbidden)
    random.seed(7)
    np.random.seed(7)
    expected = (random.random(), np.random.rand())
    random.seed(7)
    np.random.seed(7)
    for path in (SCENARIO_A, SCENARIO_C, STATIC_SMALL):
        d = load_scenario_definition(path)
        for view in ("simplified", "verbose"):
            export_figure(build_figure(d, view), tmp_path / f"{d.name}_{view}", ("png",), dpi=72)
    assert (random.random(), np.random.rand()) == expected


# ------------------------------------------------------------- host counts


@pytest.mark.parametrize("path, ranges", [
    (SCENARIO_A, [(1, 1), (2, 5)]),
    (SCENARIO_B, [(2, 2), (1, 1)]),
    (SCENARIO_B2, [(2, 2)]),
    (SCENARIO_C, [(2, 2), (2, 2)]),
    (MEDIUM, [(1, 1), (6, 10), (6, 10), (4, 6)]),
    (STATIC_SMALL, [(1, 1), (1, 1), (5, 5), (1, 1)]),
])
def test_host_count_ranges_come_from_the_yaml(path, ranges):
    d = load_scenario_definition(path)
    assert [(s.size_min, s.size_max) for s in d.subnets] == ranges
    layout = compute_layout(d, "simplified")
    for s, (lo, hi) in zip(d.subnets, ranges):
        expected = f"Hosts: {lo}" if lo == hi else f"Hosts: {lo}–{hi}"
        assert expected in _subnet_rows(layout, s.subnet_id)
    assert d.host_count_range == (sum(lo for lo, _ in ranges), sum(hi for _, hi in ranges))


def test_host_count_distribution_is_uniform_over_the_range():
    d = load_scenario_definition(SCENARIO_A)
    user = d.subnets[1]
    assert user.size_distribution == tuple((k, 0.25) for k in range(2, 6))
    assert "uniform, 25% each" in _subnet_rows(compute_layout(d, "verbose"), 2)


def test_connectivity_matches_the_yaml_topology():
    for path in (SCENARIO_A, MEDIUM, STATIC_SMALL):
        topology = yaml.safe_load(path.read_text())["topology"]
        d = load_scenario_definition(path)
        expected = {(i, j) for i in range(len(topology)) for j in range(i + 1, len(topology)) if topology[i][j] or topology[j][i]}
        assert {(c.a, c.b) for c in d.connections} == expected
        layout = compute_layout(d, "simplified")
        assert {e.connection for e in layout.edges if e.connection} == expected


# ----------------------------------------------------------------- targets


def test_sensitive_targets_are_represented_as_defined():
    a = load_scenario_definition(SCENARIO_A)
    assert [(s.target_probability, s.targets_min, s.targets_max) for s in a.subnets] == [(1.0, 1, 1), (0.0, 0, 0)]
    layout = compute_layout(a, "simplified")
    assert "Targets: 1" in _subnet_rows(layout, 1) and "Targets: none" in _subnet_rows(layout, 2)
    diamonds = [g for g in layout.glyphs if g.kind == "target" and g.role == "diagram"]
    box1 = next(b for b in layout.boxes if b.kind == "subnet" and b.subnet_id == 1)
    assert len(diamonds) == 1 and box1.rect.contains(diamonds[0].rect)

    c = load_scenario_definition(SCENARIO_C)
    assert [(s.targets_min, s.targets_max) for s in c.subnets] == [(0, 2), (2, 2)]
    assert (c.targets_min, c.targets_max) == (2, 4)
    assert "each host 50%, independently" in _subnet_rows(compute_layout(c, "verbose"), 1)

    small = load_scenario_definition(STATIC_SMALL)
    assert [t for s in small.subnets for t in s.explicit_targets] == [((2, 0), 100.0), ((4, 0), 100.0)]


def test_zero_target_probability_and_marla_conditioning_are_stated():
    b2 = load_scenario_definition(SCENARIO_B2)
    assert b2.p_zero_targets == pytest.approx(0.25)
    assert load_scenario_definition(SCENARIO_C).p_zero_targets == 0.0
    notes = " ".join(_texts(compute_layout(b2, "verbose"), {"note"}))
    assert "P(no target) = 25%" in notes and "MARLA regenerates" in notes
    # The observed regeneration frequency agrees (52 of the study's 200 test seeds).
    realizer = ScenarioRealizer(SCENARIO_B2)
    regenerated = sum(realizer.realize(s).regeneration_attempt > 0 for s in range(400))
    assert abs(regenerated / 400 - 0.25) < 4 * math.sqrt(0.25 * 0.75 / 400)


# ----------------------------------------------------------- probabilities


def test_skew_distribution_matches_nasimemu():
    assert [round(p, 4) for _, p in skew_count_distribution(3)] == [0.5714, 0.2857, 0.1429]
    assert [round(p, 4) for _, p in skew_count_distribution(2)] == [0.6667, 0.3333]
    assert skew_count_distribution(0) == ()


def test_verbose_shows_the_generation_probabilities():
    d = load_scenario_definition(SCENARIO_C)
    layout = compute_layout(d, "verbose")
    cells = _texts(layout, {"table_cell"})
    for expected in ("linux", "windows", "50%", "21_linux_proftpd", "52.4%", "9200_windows_elasticsearch", "66.7%",
                     "3306_any_mysql", "target 100%, other 10%", "Services | linux", "Services | windows", "0.8"):
        assert expected in cells, expected
    notes = " ".join(_texts(layout, {"table_note"}))
    assert "P(present | OS)" in notes and "not independent" in notes
    assert "3306_any_mysql: 55% of hosts" in _subnet_rows(layout, 1)
    assert "3306_any_mysql: 100% of hosts" in _subnet_rows(layout, 2)


def test_missing_probabilities_are_not_invented(tmp_path):
    def edit(raw):
        del raw["sensitive_hosts"][2]
        raw["subnets"][0] = "several"
    d = load_scenario_definition(_write_scenario(tmp_path, edit))
    dmz, internal = d.subnets
    assert (dmz.size_min, dmz.size_max) == (None, None) and dmz.size_note
    assert internal.target_probability is None and (internal.targets_min, internal.targets_max) == (None, None)
    assert any("no sensitive_hosts probability" in w for w in d.warnings)
    layout = compute_layout(d, "simplified")
    assert "Hosts: unspecified" in _subnet_rows(layout, 1)
    assert "Targets: unspecified" in _subnet_rows(layout, 2)
    verbose_cells = _texts(compute_layout(d, "verbose"), {"table_cell"})
    assert "none defined" in verbose_cells  # processes: ~ only -- no invented process probability


def test_simplified_figure_omits_attribute_distributions():
    for path in (SCENARIO_A, SCENARIO_C, MEDIUM):
        d = load_scenario_definition(path)
        text = " ".join(_texts(compute_layout(d, "simplified")))
        assert "%" not in text
        for word in d.os_names + d.services + ("Services", "Probability", "Actions"):
            assert word not in text


def test_probabilities_match_the_real_generator():
    """The closed-form probabilities against 600 realizations of the real
    generator (scenario C never triggers MARLA's zero-target regeneration)."""
    d = load_scenario_definition(SCENARIO_C)
    realizer = ScenarioRealizer(SCENARIO_C)
    os_counts, service_counts, mysql = Counter(), defaultdict(Counter), defaultdict(list)
    for seed in range(600):
        r = realizer.realize(seed)
        for s in r.subnets:
            for h in r.hosts_in(s.subnet_id):
                os_counts[h.os] += 1
                for svc in h.services:
                    service_counts[h.os][svc] += 1
                mysql[s.source_index].append("3306_any_mysql" in h.services)

    def close(observed: float, p: float, n: int) -> bool:
        return abs(observed - p) <= 4 * math.sqrt(p * (1 - p) / n) + 1e-9

    total = sum(os_counts.values())
    for model in d.host_model:
        assert close(os_counts[model.os] / total, model.probability, total)
        for svc in model.service_candidates:
            assert close(service_counts[model.os][svc] / os_counts[model.os], model.service_presence, os_counts[model.os])
    for s in d.subnets:
        (_, presence), = s.sensitive_service_presence
        assert close(sum(mysql[s.source_index]) / len(mysql[s.source_index]), presence, len(mysql[s.source_index]))


def test_static_scenario_lists_its_hosts_and_firewall_rules():
    d = load_scenario_definition(STATIC_SMALL)
    assert not d.randomized_hosts and not d.host_model
    layout = compute_layout(d, "verbose")
    rows = _subnet_rows(layout, 2)
    assert "(2,0)" in rows and "services: ftp, ssh" in rows and "processes: tomcat" in rows
    assert "(2,0), value 100" in rows
    rules = _texts(layout, {"firewall_rule"})
    assert "FW1" in rules and any("Internet → Subnet 1: http" in r for r in rules)


# ------------------------------------------------------------------ quality


@pytest.mark.parametrize("path", ALL_BUNDLED, ids=lambda p: p.name)
def test_definition_layouts_are_clean_and_deterministic(path):
    d = load_scenario_definition(path)
    for view in ("simplified", "verbose"):
        layout = compute_layout(d, view)
        assert check_layout(layout) == []
        assert layout.min_font_pt >= 5.0
        assert layout == compute_layout(load_scenario_definition(path), view)


def test_definition_and_realization_share_conventions():
    d = compute_layout(load_scenario_definition(SCENARIO_C), "simplified")
    r = compute_layout(ScenarioRealizer(SCENARIO_C).realize(3), "simplified")
    for layout in (d, r):
        assert [b.kind for b in layout.boxes].count("internet") == 1
        assert {t.text for t in layout.texts if t.role == "subnet_title"} <= {
            "Subnet 1 · DMZ", "Subnet 2 · DMZ", "Subnet 1 · internal", "Subnet 2 · internal"}
        assert any(g.kind == "target" for g in layout.glyphs)
    # The YAML subnets sit in the same slots in both figures.
    assert d.subnet_slots == r.subnet_slots


# -------------------------------------------------------- CLI and manifest


def test_cli_plot_without_seed_draws_both_definition_figures(tmp_path):
    result = runner.invoke(app, ["scenario", "plot", str(SCENARIO_C), "--output-dir", str(tmp_path), "--dpi", "72"])
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [f"evidence_c_multi_target{v}.{f}" for v in ("", "_verbose") for f in ("pdf", "svg", "png")] + ["manifest.json"])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    entry = manifest["figures"]["evidence_c_multi_target"]
    assert entry["kind"] == "definition" and set(entry["views"]) == {"simplified", "verbose"}
    assert (entry["hosts_min"], entry["hosts_max"], entry["targets_min"], entry["targets_max"]) == (4, 4, 2, 4)
    assert "seeds" not in entry
    for path in tmp_path.glob("*.*"):
        assert path.stat().st_size > 0


@pytest.mark.parametrize("mode, expected", [
    ("simplified", ["evidence_b2_two_hosts.png"]),
    ("verbose", ["evidence_b2_two_hosts_verbose.png"]),
    ("both", ["evidence_b2_two_hosts.png", "evidence_b2_two_hosts_verbose.png"]),
])
def test_cli_mode_and_format(tmp_path, mode, expected):
    result = runner.invoke(app, ["scenario", "plot", str(SCENARIO_B2), "--mode", mode, "--format", "png",
                                 "--output-dir", str(tmp_path), "--dpi", "72"])
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in tmp_path.glob("*.png")) == expected
    assert not list(tmp_path.glob("*.pdf"))


def test_cli_format_all_and_overwrite_policy(tmp_path):
    args = ["scenario", "plot", str(SCENARIO_A), "--format", "all", "--output-dir", str(tmp_path), "--dpi", "72"]
    assert runner.invoke(app, args).exit_code == 0
    assert {p.suffix for p in tmp_path.glob("sm_entry_dmz_one_target.*")} == {".pdf", ".svg", ".png"}
    again = runner.invoke(app, args)
    assert again.exit_code == 1 and "--overwrite" in again.output
    assert runner.invoke(app, args + ["--overwrite"]).exit_code == 0


def test_schema_1_manifests_are_migrated(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema": "marla.scenario_figures/1", "scenarios": {}, "unreconstructed": [],
        "figures": {"x_seed_1": {"id": "x_seed_1", "scenario": "x", "views": {"default": {"files": {}}}}},
    }))
    manifest = load_manifest(tmp_path)
    assert manifest["schema"] == MANIFEST_SCHEMA
    entry = manifest["figures"]["x_seed_1"]
    assert entry["kind"] == "realization" and set(entry["views"]) == {"simplified"}


def test_definitions_and_realizations_share_one_manifest(tmp_path):
    plot_definitions([SCENARIO_C], tmp_path, formats=("png",), dpi=72)
    result = runner.invoke(app, ["scenario", "plot", str(SCENARIO_C), "--seed", "2", "--format", "png",
                                 "--output-dir", str(tmp_path), "--dpi", "72"])
    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    kinds = {fid: e["kind"] for fid, e in manifest["figures"].items()}
    assert kinds == {"evidence_c_multi_target": "definition", "evidence_c_multi_target_seed_2": "realization"}
    assert manifest["figures"]["evidence_c_multi_target_seed_2"]["paper_representative"] is True
    assert "paper_representative" not in manifest["figures"]["evidence_c_multi_target"]
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["scenarios"]["evidence_c_multi_target"]["sha256"])
