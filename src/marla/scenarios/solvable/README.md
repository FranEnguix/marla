# MARLA-owned solvable scenarios

Every file in this directory has passed `marla scenario check` (universal
structural-solvability validation, `capture_target` objective) at the time
it was committed. See [`marla.scenario_solvability`](../../../../docs/scenario_solvability.rst)
for the full definition, and `marla.scenario.models.ScenarioSolvabilityResult`
for the exact result each was validated against.

`.solvable.v2.yaml` means: *every realization NASimEmu's loader can
legally produce from this file has a valid action sequence reaching ROOT
on every sensitive host.* It does not mean PPO will learn it, that it's
easy, or that a solution fits in any particular `max_episode_steps`
budget -- see the linked docs for the full distinction between structural
solvability and learnability.

## Provenance

| File | Repaired from | Repair applied |
|---|---|---|
| `sm_entry_user_three_subnets.solvable.v2.yaml` | `NASimEmu/scenarios/sm_entry_user_three_subnets.v2.yaml` | Added one artificial privilege escalation, `marla_repair_windows_root_privesc` (`process: null, os: windows, access: root`) -- the original scenario could legally generate a sensitive Windows host exploitable only to USER (via `e_wp_ninja`) with no Windows privilege escalation to ROOT. See `research/aamas2027/` for the investigation that found this, and the file's own diff in git history for the exact change. |
| `md_entry_user_three_subnets.solvable.v2.yaml` | `NASimEmu/scenarios/md_entry_user_three_subnets.v2.yaml` | Same repair class and reason as above (`marla_repair_windows_root_privesc`, same failure signature: `services=['3306_any_mysql', '80_windows_wp_ninja']`, USER attainable via `e_wp_ninja`, no Windows ROOT privilege escalation). Packaged as a scenario-shift (medium-network, same entry/topology family) out-of-distribution counterpart to `sm_entry_user_three_subnets.solvable.v2.yaml`, for generalization evaluation -- not used for training. |

The original NASimEmu scenarios are deliberately left unmodified in
`NASimEmu/scenarios/` -- they remain useful as provenance demonstrating
*why* each repair was necessary (see `marla scenario check
NASimEmu/scenarios/<name>.v2.yaml`).

## Regenerating

```bash
marla scenario repair NASimEmu/scenarios/sm_entry_user_three_subnets.v2.yaml \
    --output src/marla/scenarios/solvable/sm_entry_user_three_subnets.solvable.v2.yaml \
    --overwrite
marla scenario check src/marla/scenarios/solvable/sm_entry_user_three_subnets.solvable.v2.yaml

marla scenario repair NASimEmu/scenarios/md_entry_user_three_subnets.v2.yaml \
    --output src/marla/scenarios/solvable/md_entry_user_three_subnets.solvable.v2.yaml \
    --overwrite
marla scenario check src/marla/scenarios/solvable/md_entry_user_three_subnets.solvable.v2.yaml
```

Never repair directly into this directory without re-running `marla
scenario check` on the result and confirming `PROVEN_SOLVABLE` -- the
repair command already does this itself and refuses to write a file it
can't verify, but always confirm before committing.
