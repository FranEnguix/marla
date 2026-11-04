"""Seeded scenario reconstruction and publication figures (``marla scenario
plot``/``plot-batch``): exact realizations, both views, rendering quality,
output naming/manifest, batch verification and CLI behaviour."""

from __future__ import annotations

import csv
import json
import random
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from marla.cli import app
from marla.environment.nasimemu_adapter import NasimEmuAdapter, _regeneration_seed
from marla.environment.scenario_realization import (
    MAX_SCENARIO_SEED,
    ScenarioRealizer,
    realize_scenario,
    scenario_display_name,
)
from marla.visualization.scenario_figures import (
    ScenarioFigureError,
    ScenarioRequest,
    figure_stem,
    parse_seed_list,
    plot_requests,
    read_census,
)
from marla.visualization.scenario_layout import NO_PROCESSES, card_contents, check_layout, compute_layout
from marla.visualization.scenario_plot import FigureTooSmallError, build_figure, export_figure

REPO = Path(__file__).resolve().parent.parent
DIAGNOSTIC = REPO / "src" / "marla" / "scenarios" / "diagnostic"
SCENARIO_A = REPO / "tests" / "data" / "scenarios" / "sm_entry_dmz_one_target.v2.yaml"
SCENARIO_B = DIAGNOSTIC / "evidence_b_pivot_one_target.v2.yaml"
SCENARIO_B2 = DIAGNOSTIC / "evidence_b2_two_hosts.v2.yaml"
SCENARIO_C = DIAGNOSTIC / "evidence_c_multi_target.v2.yaml"
STUDY_SCENARIOS = (SCENARIO_A, SCENARIO_B, SCENARIO_B2, SCENARIO_C)
LARGE = REPO / "src" / "marla" / "scenarios" / "solvable" / "md_entry_user_three_subnets.solvable.v2.yaml"
# NASimEmu's static benchmark: processes, restricted per-direction subnet firewalls.
STATIC_SMALL = REPO / "NASimEmu" / "src" / "nasimemu" / "nasim" / "scenarios" / "benchmark" / "small.yaml"
ALL_BUNDLED = sorted((REPO / "src" / "marla" / "scenarios").glob("*/*.yaml")) + [SCENARIO_A]

runner = CliRunner()


def _first_seed(path: Path, predicate, start: int = 0, limit: int = 400) -> int:
    realizer = ScenarioRealizer(path)
    for seed in range(start, start + limit):
        if predicate(realizer.realize(seed)):
            return seed
    raise AssertionError("no seed found")


def _direct_ground_truth(path: Path, seed: int) -> dict:
    """The simulator state after a plain NasimEmuAdapter.reset -- independent
    of the reconstruction code path (as the study's census script reads it)."""
    adapter = NasimEmuAdapter(str(path), max_episode_steps=1)
    adapter.reset(seed=seed)
    env = adapter._env.env
    sensitive = set(env.network.sensitive_addresses)
    out = {}
    for address in env.network.address_space:
        host = env.current_state.get_host(address)
        out[f"host-{address[0]}-{address[1]}"] = (
            next(name for name, on in host.os.items() if on),
            tuple(sorted(name for name, on in host.services.items() if on)),
            tuple(sorted(p for p, on in host.processes.items() if on and p is not None)),
            address in sensitive,
        )
    return out


def _texts(layout, role=None):
    return [t.text for t in layout.texts if role is None or t.role == role]


# ------------------------------------------------------- seed reproducibility


@pytest.mark.parametrize("path", STUDY_SCENARIOS, ids=lambda p: p.name)
def test_same_seed_reconstructs_identical_realization(path):
    first = realize_scenario(path, 3_000_000_005)
    second = ScenarioRealizer(path).realize(3_000_000_005)
    assert first.to_dict() == second.to_dict()


@pytest.mark.parametrize("path", STUDY_SCENARIOS + (STATIC_SMALL,), ids=lambda p: p.name)
def test_reconstruction_matches_the_simulators_generated_state(path):
    realizer = ScenarioRealizer(path)
    for seed in (0, 7, 3_000_000_000, 3_000_000_013):
        r = realizer.realize(seed)
        got = {h.key: (h.os, h.services, h.processes, h.sensitive) for h in r.hosts}
        assert got == _direct_ground_truth(path, seed)


def test_different_seeds_get_their_own_sampled_attributes():
    realizer = ScenarioRealizer(SCENARIO_C)
    realizations = [realizer.realize(s) for s in range(3_000_000_000, 3_000_000_010)]
    assert len({r.realization_fingerprint for r in realizations}) > 1
    for r in realizations:
        assert {h.key for h in r.hosts} == set(_direct_ground_truth(SCENARIO_C, r.requested_seed))


def test_zero_target_regeneration_is_reproduced():
    seed = _first_seed(SCENARIO_B2, lambda r: r.regeneration_attempt > 0, start=3_000_000_000)
    r = realize_scenario(SCENARIO_B2, seed)
    assert r.realization_seed == _regeneration_seed(seed, r.regeneration_attempt) != seed
    assert r.sensitive_hosts
    # Starting directly from the effective seed generates the identical network.
    direct = realize_scenario(SCENARIO_B2, r.realization_seed)
    assert direct.regeneration_attempt == 0
    assert direct.realization_fingerprint == r.realization_fingerprint
    adapter = NasimEmuAdapter(str(SCENARIO_B2), max_episode_steps=1)
    adapter.reset(seed=seed)
    assert (adapter.requested_seed, adapter.realization_seed, adapter.regeneration_attempt) == (
        seed, r.realization_seed, r.regeneration_attempt)


def test_subnet_permutation_is_recovered_so_yaml_subnets_keep_their_identity():
    realizer = ScenarioRealizer(SCENARIO_C)
    perms = set()
    for seed in range(20):
        r = realizer.realize(seed)
        perms.add(r.subnet_permutation)
        dmz, internal = (r.subnets[0], r.subnets[1])
        assert (dmz.source_index, dmz.label, internal.label) == (1, "DMZ", "internal")
        assert dmz.internet_facing and not internal.internet_facing
        # Every internal host is sensitive in this scenario (probability 1.0).
        assert all(h.sensitive for h in r.hosts_in(internal.subnet_id))
    assert perms == {(0, 1, 2), (0, 2, 1)}  # NASimEmu does permute subnet IDs


def test_reconstruction_and_rendering_do_not_perturb_generation(tmp_path):
    adapter = NasimEmuAdapter(str(SCENARIO_C), max_episode_steps=5)
    baseline = adapter.reset(seed=42).raw_observation.copy()
    random.seed(123)
    np.random.seed(123)
    expected = (random.random(), np.random.rand())
    random.seed(123)
    np.random.seed(123)
    r = realize_scenario(SCENARIO_C, 42)
    export_figure(build_figure(r, "verbose"), tmp_path / "x", ("png",), dpi=72)
    assert (random.random(), np.random.rand()) == expected  # caller's RNG streams untouched
    assert np.array_equal(adapter.reset(seed=42).raw_observation, baseline)
    from nasimemu.nasim.scenarios.scenario import Scenario
    assert Scenario._permute_subnets.__name__ == "_permute_subnets"  # recording wrapper removed


@pytest.mark.parametrize("seed", [-1, MAX_SCENARIO_SEED + 1, 1.5, True])
def test_invalid_seeds_are_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        realize_scenario(SCENARIO_C, seed)


def test_ground_truth_reader_is_never_imported_by_policy_code():
    """The realization reader exposes hidden simulator state; only the
    offline figure code and its CLI entry points may use it."""
    src = REPO / "src" / "marla"
    allowed = {src / "environment" / "scenario_realization.py", src / "cli.py", src / "scenario_cli.py"}
    allowed |= set((src / "visualization").glob("*.py"))
    pattern = re.compile(r"^\s*(from|import)\s+marla\.environment\.scenario_realization\b", re.MULTILINE)
    importers = {p for p in src.rglob("*.py") if pattern.search(p.read_text(encoding="utf-8"))}
    assert importers and importers <= allowed, sorted(str(p) for p in importers - allowed)


# ------------------------------------------------------------- default view


@pytest.mark.parametrize("path", STUDY_SCENARIOS, ids=lambda p: p.name)
def test_default_view_shows_topology_and_targets_only(path):
    r = realize_scenario(path, 3_000_000_001)
    layout = compute_layout(r, "simplified")
    subnets = [b for b in layout.boxes if b.kind == "subnet"]
    assert sorted(b.subnet_id for b in subnets) == sorted(s.subnet_id for s in r.subnets)
    assert [b for b in layout.boxes if b.kind == "internet"]
    assert not [b for b in layout.boxes if b.kind == "card"]
    glyphs = [g for g in layout.glyphs if g.role == "diagram" and g.kind in ("host", "target")]
    assert sorted(g.address for g in glyphs) == sorted(h.address for h in r.hosts)
    assert sorted(g.address for g in glyphs if g.kind == "target") == sorted(h.address for h in r.sensitive_hosts)
    assert sorted(e.connection for e in layout.edges if e.connection) == sorted((c.a, c.b) for c in r.connections)
    assert {c.b for c in r.connections if c.a == 0} == {s.subnet_id for s in r.subnets if s.internet_facing}
    text = " ".join(_texts(layout))
    for word in r.os_names + r.services + r.processes + ("Services", "Processes", "Exploits"):
        assert word not in text


def test_default_layout_keeps_yaml_subnets_in_place_across_seeds():
    realizer = ScenarioRealizer(SCENARIO_C)
    centers = {}
    for seed in range(12):
        r = realizer.realize(seed)
        layout = compute_layout(r, "simplified")
        boxes = {b.subnet_id: b.rect for b in layout.boxes if b.kind == "subnet"}
        for s in r.subnets:
            centers.setdefault(s.source_index, set()).add((round(boxes[s.subnet_id].cx, 6), round(boxes[s.subnet_id].y1, 6)))
        assert layout.subnet_slots == {1: (1, 0), 2: (2, 0)}
    assert all(len(v) == 1 for v in centers.values())


def test_layout_is_deterministic():
    r = realize_scenario(LARGE, 3)
    assert compute_layout(r, "verbose") == compute_layout(r, "verbose")


# ------------------------------------------------------------- verbose view


def _card_values(layout, address):
    card = next(b for b in layout.boxes if b.kind == "card" and b.address == address)
    rows, label = {}, None
    for t in sorted((t for t in layout.texts if card.rect.contains(t.bbox) and t.role in ("card_label", "card_value")),
                    key=lambda t: (-round(t.y, 6), t.x)):
        if t.role == "card_label":
            label = t.text
            rows[label] = []
        else:
            rows[label].append(t.text)
    return rows


@pytest.mark.parametrize("path", STUDY_SCENARIOS + (STATIC_SMALL,), ids=lambda p: p.name)
def test_verbose_cards_show_exactly_the_sampled_host_attributes(path):
    r = realize_scenario(path, 3_000_000_002)
    layout = compute_layout(r, "verbose")
    assert len([b for b in layout.boxes if b.kind == "card"]) == len(r.hosts)
    for host in r.hosts:
        rows = _card_values(layout, host.address)
        assert rows["OS"] == [host.os]
        assert rows["Services"] == list(host.services)
        if r.defines_processes:
            assert rows["Processes"] == (list(host.processes) or ["none"])
        else:
            assert rows["Processes"] == [NO_PROCESSES]
        assert rows["Value"] == [f"{host.value:g}"]
        expected_exploits = [a.name for a in host.applicable_exploits] or ["none"]
        assert [v.split(" ")[0] for v in rows["Exploits"]] == expected_exploits
        card = next(b for b in layout.boxes if b.kind == "card" and b.address == host.address)
        assert card.target == host.sensitive
        tags = [t for t in layout.texts if t.role == "card_tag" and card.rect.contains(t.bbox)]
        assert bool(tags) == host.sensitive


def test_verbose_view_never_shows_probabilities_as_observed_services():
    r = realize_scenario(SCENARIO_C, 3_000_000_003)
    for card in card_contents(r):
        services = dict(card.rows)["Services"]
        assert all(s in r.services for s in services)
        assert not any(re.search(r"\b0\.\d", s) for s in services)


def test_static_scenario_processes_and_firewall_rules_are_shown():
    r = realize_scenario(STATIC_SMALL, 7)
    assert r.defines_processes and any(h.processes for h in r.hosts)
    layout = compute_layout(r, "verbose")
    rules = [t.text for t in layout.texts if t.role == "firewall_rule"]
    restricted = [c for c in r.connections if c.restricted]
    assert len([t for t in rules if re.fullmatch(r"FW\d+", t)]) == len(restricted) > 0
    assert any("Internet → Subnet" in t for t in rules)
    assert sum(1 for g in layout.glyphs if g.kind == "firewall" and g.role == "diagram") == len(restricted)


def test_unknown_os_is_labelled_not_invented():
    r = realize_scenario(SCENARIO_C, 1)
    host = r.hosts[0]
    from dataclasses import replace
    r2 = replace(r, hosts=(replace(host, os=None),) + r.hosts[1:])
    assert dict(card_contents(r2)[0].rows)["OS"] == ("unknown",)


# --------------------------------------------------------- rendering quality


@pytest.mark.parametrize("path", ALL_BUNDLED + [STATIC_SMALL], ids=lambda p: p.name)
def test_layouts_have_no_overlaps_clipping_or_edges_through_boxes(path):
    realizer = ScenarioRealizer(path)
    for seed in (0, 1, 2):
        r = realizer.realize(seed)
        for view in ("simplified", "verbose"):
            layout = compute_layout(r, view)
            assert check_layout(layout) == []
            assert layout.min_font_pt >= 5.0


def test_rendered_text_matches_the_layout_and_stays_on_canvas():
    r = realize_scenario(STATIC_SMALL, 3)
    rendered = build_figure(r, "verbose")
    fig = rendered.figure
    renderer = fig.canvas.get_renderer()
    fig.canvas.draw()
    canvas = fig.bbox
    boxes = []
    for text in fig.axes[0].texts:
        bb = text.get_window_extent(renderer)
        assert canvas.x0 - 1 <= bb.x0 and bb.x1 <= canvas.x1 + 1 and canvas.y0 - 1 <= bb.y0 and bb.y1 <= canvas.y1 + 1
        boxes.append(bb)
    assert len(boxes) == len(rendered.layout.texts) + sum(g.kind == "firewall" for g in rendered.layout.glyphs)


def test_exports_are_vector_nonempty_and_byte_reproducible(tmp_path):
    r = realize_scenario(SCENARIO_C, 3_000_000_001)
    a = export_figure(build_figure(r, "verbose"), tmp_path / "a" / "fig", dpi=300)
    b = export_figure(build_figure(r, "verbose"), tmp_path / "b" / "fig", dpi=300)
    for pa, pb in zip(a, b):
        assert pa.stat().st_size > 1000
        assert pa.read_bytes() == pb.read_bytes()
    pdf = (tmp_path / "a" / "fig.pdf").read_bytes()
    assert b"FontFile2" in pdf and b"/Subtype /Image" not in pdf  # embedded TrueType text, no raster
    svg = (tmp_path / "a" / "fig.svg").read_text()
    assert "<text" in svg and "<image" not in svg
    from matplotlib.image import imread
    height, width = imread(tmp_path / "a" / "fig.png").shape[:2]
    assert width == pytest.approx(build_figure(r, "verbose").width_in * 300, abs=2)


def test_paper_widths_scale_down_but_refuse_unreadable_text():
    small = build_figure(realize_scenario(SCENARIO_C, 1), "simplified", size="single")
    assert small.width_in == pytest.approx(3.33)
    assert small.scale == 1.0  # never scaled up
    with pytest.raises(FigureTooSmallError):
        build_figure(realize_scenario(LARGE, 1), "verbose", size="single")
    large = build_figure(realize_scenario(LARGE, 1), "verbose", size="natural")
    assert large.layout.min_font_pt >= 6.0 and large.width_in > 7.0


# ------------------------------------------------------- naming and manifest


def _request(path: Path, seeds) -> dict:
    realizer = ScenarioRealizer(path)
    request = ScenarioRequest(realizer.path)
    for s in seeds:
        request.record(s).sources.add("test")
    return {realizer.sha256: request}


def test_flat_output_naming_and_manifest(tmp_path):
    seed = _first_seed(SCENARIO_B2, lambda r: r.regeneration_attempt > 0, start=3_000_000_000)
    effective = realize_scenario(SCENARIO_B2, seed).realization_seed
    result = plot_requests(_request(SCENARIO_B2, [seed, 5]), tmp_path, dpi=72)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert not [p for p in tmp_path.iterdir() if p.is_dir()]
    stem = f"evidence_b2_two_hosts_seed_{effective}"  # effective seed in the file name
    for suffix in ("", "_verbose"):
        for fmt in ("pdf", "svg", "png"):
            assert f"{stem}{suffix}.{fmt}" in files
    assert "manifest.json" in files and len(files) == 2 * 6 + 1
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    entry = manifest["figures"][stem]
    assert entry["seeds"][0]["requested_seed"] == seed
    assert entry["seeds"][0]["regeneration_attempt"] > 0
    assert entry["scenario_sha256"] == ScenarioRealizer(SCENARIO_B2).sha256
    assert entry["views"]["verbose"]["files"]["pdf"] == f"{stem}_verbose.pdf"
    assert sum(e["paper_representative"] for e in manifest["figures"].values()) == 1
    assert result.files_written == {"pdf": 4, "svg": 4, "png": 4}


def test_identical_realizations_are_rendered_once(tmp_path):
    realizer = ScenarioRealizer(SCENARIO_B2)
    seen, pair = {}, None
    for seed in range(3_000_000_000, 3_000_000_200):
        fp = realizer.realize(seed).realization_fingerprint
        if fp in seen:
            pair = (seen[fp], seed)
            break
        seen[fp] = seed
    assert pair, "scenario B2 has identical realizations among its test seeds"
    result = plot_requests(_request(SCENARIO_B2, pair), tmp_path, views=("simplified",), formats=("png",), dpi=72)
    assert len(result.rendered) == 1 and result.aliases == 1
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    (entry,) = manifest["figures"].values()
    assert sorted(s["requested_seed"] for s in entry["seeds"]) == sorted(pair)


def test_existing_files_are_never_silently_overwritten(tmp_path):
    plot_requests(_request(SCENARIO_C, [3]), tmp_path, views=("simplified",), formats=("png",), dpi=72)
    with pytest.raises(ScenarioFigureError, match="already exist"):
        plot_requests(_request(SCENARIO_C, [3]), tmp_path, views=("simplified",), formats=("png",), dpi=72)
    skipped = plot_requests(_request(SCENARIO_C, [3]), tmp_path, views=("simplified",), formats=("png",), dpi=72,
                            skip_existing=True)
    assert skipped.skipped_existing and not skipped.rendered
    redone = plot_requests(_request(SCENARIO_C, [3]), tmp_path, views=("simplified",), formats=("png",), dpi=72,
                           overwrite=True)
    assert redone.rendered


def test_same_name_different_scenario_gets_a_hash_suffix(tmp_path):
    other = tmp_path / "src" / SCENARIO_C.name
    other.parent.mkdir()
    other.write_text(SCENARIO_C.read_text().replace("prob: 0.8", "prob: 0.7"))
    out = tmp_path / "out"
    plot_requests(_request(SCENARIO_C, [1]), out, views=("simplified",), formats=("png",), dpi=72)
    plot_requests(_request(other, [1]), out, views=("simplified",), formats=("png",), dpi=72)
    manifest = json.loads((out / "manifest.json").read_text())
    names = sorted(manifest["scenarios"])
    sha = ScenarioRealizer(other).sha256
    assert names == sorted(["evidence_c_multi_target", f"evidence_c_multi_target_{sha[:8]}"])
    assert (out / f"evidence_c_multi_target_{sha[:8]}_seed_1.png").is_file()


def test_seed_list_parsing():
    assert parse_seed_list("11, 12,20-22") == [11, 12, 20, 21, 22]
    for bad in ("", "a", "5-3", "-1"):
        with pytest.raises(ValueError):
            parse_seed_list(bad)


# ------------------------------------------------------ batch verification


def _census(path: Path, seeds, out: Path, corrupt: int | None = None) -> Path:
    realizer = ScenarioRealizer(path)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["split", "seed", "host", "os", "services", "sensitive"])
        writer.writeheader()
        for seed in seeds:
            for h in realizer.realize(seed).hosts:
                services = ";".join(h.services) if seed != corrupt else "80_linux_drupal"
                writer.writerow({"split": "test", "seed": seed, "host": h.key, "os": h.os, "services": services,
                                 "sensitive": h.sensitive})
    return out


def test_census_batch_verifies_hosts_and_reports_mismatches(tmp_path):
    census = _census(SCENARIO_C, [10, 11, 12], tmp_path / "census.csv", corrupt=11)
    result = runner.invoke(app, ["scenario", "plot-batch", str(SCENARIO_C), "--from-census", str(census),
                                 "--output-dir", str(tmp_path / "out"), "--format", "png", "--dpi", "72"])
    assert result.exit_code == 1
    assert "could NOT be reconstructed" in result.output and "seed 11" in result.output
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert sorted(e["effective_seed"] for e in manifest["figures"].values()) == [10, 12]
    assert manifest["unreconstructed"][0]["requested_seed"] == 11
    assert not list((tmp_path / "out").glob("*seed_11*"))
    seeds = [s for e in manifest["figures"].values() for s in e["seeds"]]
    assert all(s["verification"]["census"].startswith("match") for s in seeds)
    assert read_census(census).keys() == {10, 11, 12}


def _episodes(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "eval" / "episodes.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path.parent


def test_episode_batch_deduplicates_across_conditions_and_checks_target_counts(tmp_path):
    realizer = ScenarioRealizer(SCENARIO_C)
    rows = []
    for condition, training_seed in (("baseline", 11), ("history_only", 12)):
        for seed in (20, 21):
            rows.append({"seed": seed, "scenario": str(SCENARIO_C), "is_eval": "True",
                         "sensitive_targets_total": len(realizer.realize(seed).sensitive_hosts),
                         "condition": condition, "training_seed": training_seed, "split": "test"})
    rows.append({"seed": 22, "scenario": str(SCENARIO_C), "is_eval": "True", "sensitive_targets_total": 99,
                 "condition": "baseline", "training_seed": 11, "split": "test"})
    rows.append({"seed": 23, "scenario": str(SCENARIO_C), "is_eval": "False", "sensitive_targets_total": 2,
                 "condition": "baseline", "training_seed": 11, "split": "train"})
    source = _episodes(tmp_path, rows)
    out = tmp_path / "out"
    result = runner.invoke(app, ["scenario", "plot-batch", "--from-episodes", str(source), "--eval-only",
                                 "--output-dir", str(out), "--mode", "simplified", "--format", "png", "--dpi", "72"])
    assert result.exit_code == 1, result.output  # seed 22's recorded target count does not match
    manifest = json.loads((out / "manifest.json").read_text())
    assert sorted(e["effective_seed"] for e in manifest["figures"].values()) == [20, 21]
    seed20 = manifest["figures"]["evidence_c_multi_target_seed_20"]["seeds"][0]
    assert seed20["provenance"]["conditions"] == ["baseline", "history_only"]
    assert seed20["provenance"]["training_seeds"] == [11, 12] and seed20["provenance"]["episodes"] == 2
    assert seed20["verification"]["episodes"].startswith("match")
    assert [u["requested_seed"] for u in manifest["unreconstructed"]] == [22]
    assert not list(out.glob("*seed_23*"))


def test_episode_batch_refuses_a_scenario_changed_since_the_run(tmp_path):
    scenario = tmp_path / "scenario.v2.yaml"
    shutil.copy(SCENARIO_C, scenario)
    source = _episodes(tmp_path, [{"seed": 1, "scenario": str(scenario), "is_eval": "True"}])
    (source / "metadata.json").write_text(json.dumps({"scenario_validation": {"scenario_hash": "0" * 64}}))
    result = runner.invoke(app, ["scenario", "plot-batch", "--from-episodes", str(source), "--output-dir",
                                 str(tmp_path / "out")])
    assert result.exit_code == 1 and "changed since the run" in result.output


# --------------------------------------------------------------------- CLI


def test_cli_plot_seed_mode_format_and_output_dir(tmp_path):
    out = tmp_path / "figs"
    result = runner.invoke(app, ["scenario", "plot", str(SCENARIO_A), "--seed", "11", "--output-dir", str(out),
                                 "--dpi", "72"])
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [f"sm_entry_dmz_one_target_seed_11{v}.{f}" for v in ("", "_verbose") for f in ("pdf", "svg", "png")]
        + ["manifest.json"])  # both figures by default
    result = runner.invoke(app, ["scenario", "plot", str(SCENARIO_A), "--seeds", "11-12", "--mode", "verbose",
                                 "--output-dir", str(out), "--format", "png", "--dpi", "72", "--overwrite"])
    assert result.exit_code == 0, result.output
    assert (out / "sm_entry_dmz_one_target_seed_12_verbose.png").is_file()
    assert not (out / "sm_entry_dmz_one_target_seed_12.png").exists()  # --mode verbose: only the verbose figure
    assert not (out / "sm_entry_dmz_one_target_seed_12_verbose.pdf").exists()


def test_cli_plot_refuses_to_overwrite_without_flag(tmp_path):
    args = ["scenario", "plot", str(SCENARIO_C), "--seed", "4", "--output-dir", str(tmp_path), "--format", "png",
            "--dpi", "72"]
    assert runner.invoke(app, args).exit_code == 0
    again = runner.invoke(app, args)
    assert again.exit_code == 1 and "--overwrite" in again.output
    assert runner.invoke(app, args + ["--overwrite"]).exit_code == 0


@pytest.mark.parametrize("args, message", [
    (["does/not/exist.v2.yaml", "--seed", "1"], "not found"),
    ([str(SCENARIO_C), "--seed", "-3"], "seed"),
    ([str(SCENARIO_C), "--seeds", "abc"], "seed"),
    ([str(SCENARIO_C), "--seed", "1", "--format", "jpg"], "format"),
    ([str(SCENARIO_C), "--seed", "1", "--mode", "fancy"], "--mode"),
    ([str(REPO / "examples" / "baseline.yaml")], "not a NASimEmu scenario"),
    ([str(REPO / "pyproject.toml")], "not valid YAML"),
])
def test_cli_plot_reports_helpful_errors(tmp_path, args, message):
    result = runner.invoke(app, ["scenario", "plot", *args, "--output-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert message in result.output


def test_cli_plot_run_dir_uses_the_runs_scenario_and_plots_dir(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "metadata.json").write_text(json.dumps({"scenario": str(SCENARIO_C)}))
    result = runner.invoke(app, ["scenario", "plot", "--run-dir", str(run), "--seed", "9", "--format", "png",
                                 "--dpi", "72"])
    assert result.exit_code == 0, result.output
    assert (run / "plots" / "scenarios" / "evidence_c_multi_target_seed_9.png").is_file()


def test_summarize_renders_scenarios_only_when_asked(tmp_path, monkeypatch):
    # The metric plots need a full episodes.csv; only the scenario option is under test here.
    monkeypatch.setattr("marla.metrics.plots.generate_plots", lambda run_dir, plots_dir: [])
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({
        "episode_count": 2, "goal_success_rate": 0.5, "mean_benchmark_return": 1.0,
        "mean_episode_duration_seconds": 1.0, "total_consultations": 0, "mean_plan_maker_latency_ms": None,
        "schema_rejection_rate": 0.0, "advice_changed_top_action_rate": 0.0,
        "total_training_environment_steps": 10, "total_training_seconds": 1.0,
    }))
    _episodes(run, [{"seed": 5, "scenario": str(SCENARIO_C), "is_eval": "True"},
                    {"seed": 6, "scenario": str(SCENARIO_C), "is_eval": "False"}])
    shutil.move(run / "eval" / "episodes.csv", run / "episodes.csv")
    plain = runner.invoke(app, ["summarize", str(run)])
    assert plain.exit_code == 0, plain.output
    assert not (run / "plots" / "scenarios").exists()
    result = runner.invoke(app, ["summarize", str(run), "--scenarios"])
    assert result.exit_code == 0, result.output
    names = sorted(p.name for p in (run / "plots" / "scenarios").iterdir())
    assert "evidence_c_multi_target_seed_5.pdf" in names and "evidence_c_multi_target_seed_5_verbose.png" in names
    assert "evidence_c_multi_target.pdf" in names and "evidence_c_multi_target_verbose.svg" in names
    assert not [n for n in names if "seed_6" in n]


def test_scenario_display_name():
    assert scenario_display_name("x/evidence_c_multi_target.v2.yaml") == "evidence_c_multi_target"
    assert scenario_display_name("small.yaml") == "small"
    assert figure_stem("a", 7, "verbose") == "a_seed_7_verbose"
