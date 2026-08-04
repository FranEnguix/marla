"""Stable action IDs and the ``FINISH`` wrapper action.

NASimEmu's own action space is combinatorial: an action is a
``((subnet, host), action_list_index)`` pair, where ``action_list_index``
indexes a per-scenario list of ``[ServiceScan, OSScan, SubnetScan,
ProcessScan, *named exploits, *named privescs]``. There is no built-in
"legal actions" enumerator; the environment simply lets an agent attempt any
combination and returns a failed :class:`ActionResult` at cost if the
action's preconditions aren't met.

MARLA's legal action set is therefore: every host address currently visible
in the observation, crossed with every entry of the scenario's action list,
plus the MARLA-level ``finish`` action. Illegal attempts (missing services,
unreachable targets, etc.) stay in the candidate set by design -- the Plan
Maker's knowledge base is expected to down-weight them via visible
prerequisites, and the environment itself charges their cost on failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FINISH_ACTION_ID = "finish"

_HOST_KEY_RE = re.compile(r"^host-(\d+)-(\d+)$")


def host_target_key(subnet: int, host: int) -> str:
    """Stable, human-readable key for a host address, e.g. ``host-1-2``."""
    return f"host-{subnet}-{host}"


def parse_host_target_key(target_key: str) -> tuple[int, int]:
    match = _HOST_KEY_RE.match(target_key)
    if match is None:
        raise ValueError(f"Not a valid host target key: {target_key!r}")
    return int(match.group(1)), int(match.group(2))


@dataclass(frozen=True)
class ActionDescriptor:
    """A single legal action, identified by a semantically stable ID.

    Advisory correlation with the Plan Maker must use ``action_id``, never
    positional/vector order -- the candidate set size and order can change
    from step to step as hosts are discovered.
    """

    action_id: str
    action_type: str
    target_key: str | None
    parameters: dict[str, object] = field(default_factory=dict)
    is_finish: bool = False


def finish_descriptor() -> ActionDescriptor:
    return ActionDescriptor(
        action_id=FINISH_ACTION_ID,
        action_type="finish",
        target_key=None,
        parameters={},
        is_finish=True,
    )


# (id_prefix, action_type) for each of the four fixed per-host scan actions,
# in the exact order NASimEmuEnv._create_action_lists() lays out action_list.
_SCAN_ACTION_KINDS = (
    ("service-scan", "service_scan"),
    ("os-scan", "os_scan"),
    ("subnet-scan", "subnet_scan"),
    ("process-scan", "process_scan"),
)


def build_legal_actions(nasim_env, host_addresses: list[tuple[int, int]]) -> list[ActionDescriptor]:
    """Enumerate legal action descriptors for the given visible host addresses.

    ``nasim_env`` is the underlying ``nasimemu.env.NASimEmuEnv`` instance
    (already reset, so ``exploit_list``/``privesc_list`` are populated for
    the current scenario).
    """
    descriptors: list[ActionDescriptor] = []

    for target in host_addresses:
        target_key = host_target_key(*target)

        for id_prefix, action_type in _SCAN_ACTION_KINDS:
            descriptors.append(
                ActionDescriptor(
                    action_id=f"{id_prefix}:{target_key}",
                    action_type=action_type,
                    target_key=target_key,
                    parameters={},
                )
            )

        for exploit_name, exploit_def in nasim_env.exploit_list:
            descriptors.append(
                ActionDescriptor(
                    action_id=f"exploit:{target_key}:{exploit_name}",
                    action_type="exploit",
                    target_key=target_key,
                    parameters={
                        "service": exploit_def.get("service"),
                        "os": exploit_def.get("os"),
                    },
                )
            )

        for privesc_name, privesc_def in nasim_env.privesc_list:
            descriptors.append(
                ActionDescriptor(
                    action_id=f"privilege-escalation:{target_key}:{privesc_name}",
                    action_type="privilege_escalation",
                    target_key=target_key,
                    parameters={
                        "process": privesc_def.get("process"),
                        "os": privesc_def.get("os"),
                    },
                )
            )

    descriptors.append(finish_descriptor())
    return descriptors


def resolve_action_target(nasim_env, action_id: str) -> tuple[tuple[int, int], int] | None:
    """Resolve a non-FINISH action ID back into NASimEmu's own action space.

    Returns ``(target_address, action_list_index)``, or ``None`` for
    ``finish`` (which never reaches the underlying simulator, see
    :mod:`marla.environment.finish`). Raises :class:`ValueError` if
    ``action_id`` does not correspond to any action NASimEmu currently
    knows about (e.g. an unknown exploit/privesc name).
    """
    if action_id == FINISH_ACTION_ID:
        return None

    parts = action_id.split(":")
    if len(parts) < 2:
        raise ValueError(f"Malformed action_id: {action_id!r}")
    kind, target_key = parts[0], parts[1]
    target = parse_host_target_key(target_key)

    num_scan_kinds = len(_SCAN_ACTION_KINDS)
    for scan_index, (id_prefix, _action_type) in enumerate(_SCAN_ACTION_KINDS):
        if kind == id_prefix:
            return target, scan_index

    if kind == "exploit":
        if len(parts) != 3:
            raise ValueError(f"Malformed exploit action_id: {action_id!r}")
        exploit_name = parts[2]
        for offset, (name, _def) in enumerate(nasim_env.exploit_list):
            if name == exploit_name:
                return target, num_scan_kinds + offset
        raise ValueError(f"Unknown exploit in action_id: {action_id!r}")

    if kind == "privilege-escalation":
        if len(parts) != 3:
            raise ValueError(f"Malformed privilege-escalation action_id: {action_id!r}")
        privesc_name = parts[2]
        num_exploits = len(nasim_env.exploit_list)
        for offset, (name, _def) in enumerate(nasim_env.privesc_list):
            if name == privesc_name:
                return target, num_scan_kinds + num_exploits + offset
        raise ValueError(f"Unknown privilege escalation in action_id: {action_id!r}")

    raise ValueError(f"Unrecognized action_id: {action_id!r}")
