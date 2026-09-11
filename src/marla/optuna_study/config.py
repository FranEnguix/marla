"""Pydantic schema for an Optuna study configuration YAML (``marla
optimize <study-config.yaml>``) -- a SEPARATE format from a normal MARLA
experiment config (``examples/baseline.yaml`` etc.): it names a base
experiment config to start from (every FIXED parameter, spec section 28),
plus study-level settings (sampler, storage, trial budget, tuning/holdout
seeds, horizons).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StudyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    study_name: str
    # Optuna RDBStorage URL, e.g. "sqlite:///research/optuna/<name>/study.db".
    # A relative sqlite:/// path is resolved relative to the STUDY CONFIG
    # FILE's own directory at load time (see load_study_config), not the
    # process's current working directory -- so `marla optimize` behaves
    # identically regardless of where it's invoked from.
    storage: str
    # TPE sampler seed -- deliberately NOT a paper seed (101/202/303) and
    # NOT a tuning/holdout seed (909/919/929); spec section 19.
    sampler_seed: int = Field(gt=0)
    n_startup_trials: int = Field(default=8, ge=0)
    # COMPLETED trials target (spec section 48) -- failed/pruned trials
    # do not count toward this.
    n_completed_trials_target: int = Field(gt=0)
    # The base experiment config (e.g.
    # research/configs/ppo_only_4x512_target_progress.yaml) supplying
    # every FIXED parameter (spec section 28) -- resolved relative to the
    # study config file's own directory, same rule as `storage` above.
    base_config: str
    tuning_seeds: list[int] = Field(min_length=1)
    holdout_seed: int
    tuning_total_environment_steps: int = Field(gt=0)
    finalist_total_environment_steps: int = Field(gt=0)
    # Where per-trial/finalist/holdout run directories are written --
    # resolved relative to the study config file's own directory.
    runs_root: str
    # Spec section 48: ">5 trials failing for the same infrastructure
    # reason" stops the study rather than endlessly retrying.
    max_consecutive_infrastructure_failures: int = Field(default=5, gt=0)

    @model_validator(mode="after")
    def _tuning_seeds_exclude_paper_and_holdout_seeds(self) -> "StudyConfig":
        paper_seeds = {101, 202, 303}
        overlap = paper_seeds & set(self.tuning_seeds)
        if overlap:
            raise ValueError(f"tuning_seeds must never include paper seeds, found: {sorted(overlap)}")
        if self.sampler_seed in paper_seeds:
            raise ValueError("sampler_seed must never be a paper seed (101/202/303)")
        if self.holdout_seed in paper_seeds:
            raise ValueError("holdout_seed must never be a paper seed (101/202/303)")
        if self.holdout_seed in self.tuning_seeds:
            raise ValueError("holdout_seed must be disjoint from tuning_seeds (it must never be used during Optuna)")
        return self


def load_study_config(path: str | Path) -> tuple[StudyConfig, Path]:
    """Returns ``(config, base_dir)`` -- ``base_dir`` is the study config
    file's own parent directory, the base for resolving ``storage``/
    ``base_config``/``runs_root``'s relative paths.
    """
    import yaml

    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return StudyConfig.model_validate(raw), path.resolve().parent


def resolve_storage_url(config: StudyConfig, base_dir: Path) -> str:
    """A ``sqlite:///relative/path`` URL is resolved relative to
    ``base_dir`` and its parent directory created if missing (SQLite
    itself does not create directories). Any other storage URL (e.g. a
    real database) is returned unchanged.
    """
    prefix = "sqlite:///"
    if not config.storage.startswith(prefix):
        return config.storage
    raw_path = config.storage[len(prefix) :]
    resolved = raw_path if Path(raw_path).is_absolute() else str((base_dir / raw_path).resolve())
    Path(resolved).parent.mkdir(parents=True, exist_ok=True)
    return f"{prefix}{resolved}"


def resolve_path(relative: str, base_dir: Path) -> Path:
    p = Path(relative)
    return p if p.is_absolute() else (base_dir / p).resolve()
