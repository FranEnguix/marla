from pathlib import Path

from marla.config.templates import find_nasimemu_scenario, render_templates


def test_find_nasimemu_scenario_returns_none_when_absent(tmp_path):
    assert find_nasimemu_scenario(tmp_path) is None


def test_find_nasimemu_scenario_finds_it_in_start_directory(tmp_path):
    scenario = tmp_path / "NASimEmu" / "scenarios" / "sm_entry_dmz_one_subnet.v2.yaml"
    scenario.parent.mkdir(parents=True)
    scenario.write_text("placeholder", encoding="utf-8")

    assert find_nasimemu_scenario(tmp_path) == scenario


def test_find_nasimemu_scenario_walks_up_from_a_nested_start_directory(tmp_path):
    scenario = tmp_path / "NASimEmu" / "scenarios" / "sm_entry_dmz_one_subnet.v2.yaml"
    scenario.parent.mkdir(parents=True)
    scenario.write_text("placeholder", encoding="utf-8")

    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)

    assert find_nasimemu_scenario(nested) == scenario


def test_find_nasimemu_scenario_gives_up_beyond_max_levels(tmp_path):
    scenario = tmp_path / "NASimEmu" / "scenarios" / "sm_entry_dmz_one_subnet.v2.yaml"
    scenario.parent.mkdir(parents=True)
    scenario.write_text("placeholder", encoding="utf-8")

    nested = tmp_path.joinpath(*["deep"] * 10)
    nested.mkdir(parents=True)

    assert find_nasimemu_scenario(nested, max_levels=2) is None


def test_render_templates_substitutes_scenario_into_both_templates():
    rendered = render_templates("/some/path/scenario.yaml")
    assert set(rendered) == {"baseline.yaml", "assisted.yaml"}
    for content in rendered.values():
        assert "scenario: /some/path/scenario.yaml" in content
