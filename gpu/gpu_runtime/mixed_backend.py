"""Opt-in 9/1 verified replay + human Mathlib joint policy/value training.

This profile deliberately has a separate contract and snapshot schema. Both
sources train policy and categorical value using the same verified loss loop;
that joint SFT choice is explicit, not a claim about undisclosed paper details.
The first integration starts from base. Old-profile releases are incompatible.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .mixed_objective import (
    BATCH_SIZE, OBJECTIVE_KIND, REPLAY_PROFILE, SAMPLE_WEIGHT, SFT_PROFILE,
    SOURCE_COUNTS, prepare_mixed_event,
)
from .verified_backend import VerifiedReplayBackend


class MixedReplayBackend(VerifiedReplayBackend):
    OBJECTIVE_KIND = OBJECTIVE_KIND
    SNAPSHOT_SCHEMA = "reap.gpu.mixed-replay-backend.v1"
    BACKEND_KIND = "mixed-replay"
    CONFIG_KEY = "mixed_config"

    def __init__(self, model_path: str, *, dataset_root: str | Path,
                 mathlib_dataset_root: str | Path, max_distance: int, **kwargs: Any):
        root = Path(mathlib_dataset_root).absolute()
        if not root.is_dir() or any(p.is_symlink() or
                (hasattr(p, "is_junction") and p.is_junction()) for p in (root, *root.parents)):
            raise ValueError("Mathlib dataset store must be an existing directory without links")
        self.mathlib_dataset_root = root
        super().__init__(model_path, dataset_root=dataset_root, max_distance=max_distance, **kwargs)

    def _config(self) -> dict:
        config = super()._config()
        config["support"] = {**config["support"],
            "return": "negative_integer_verified_action_longest_branch; human_profile_linear_remaining_actions"}
        config["max_batch_samples"] = BATCH_SIZE
        config["mixture"] = {"source_counts": dict(SOURCE_COUNTS), "sample_weight": SAMPLE_WEIGHT,
            "sampling_unit": "verified_action_row", "ratio_scope": "every_complete_batch",
            "source_profiles": {"replay": REPLAY_PROFILE, "mathlib_sft": SFT_PROFILE},
            "source_losses": {source: ["policy", "value"] for source in SOURCE_COUNTS}}
        config["source_loss_report_reduction"] = "contribution_to_full_batch_mean; includes_1_over_10_weight"
        return config

    def _load_mathlib_dataset(self, digest: str) -> dict:
        from cpu_runtime.mathlib_trajectory import load_mathlib_dataset
        # prepare_mixed_event validates each pin before using this fixed root.
        # Its 22-file loader is distinct from the generated 17-file loader.
        return load_mathlib_dataset(self.mathlib_dataset_root/digest, expected_sha256=digest)

    def _prepare_event(self, session_id: str, policy_version: int, event: dict) -> dict:
        return prepare_mixed_event(event, session_id=session_id, policy_version=policy_version,
            max_distance=self.max_distance, load_replay=self._load_dataset,
            load_mathlib_sft=self._load_mathlib_dataset)

    def _source_loss_totals(self, prepared: dict) -> dict:
        if prepared["source_counts"] != SOURCE_COUNTS or prepared["sample_weight"] != SAMPLE_WEIGHT:
            raise ValueError("mixed prepared batch source counts/weights mismatch")
        return {source: {"policy_loss": 0.0, "value_loss": 0.0, "kl": 0.0} for source in SOURCE_COUNTS}

    def _source_loss_detail(self, prepared: dict, source_totals: dict) -> dict:
        sources = {}
        for source, totals in source_totals.items():
            count = prepared["source_counts"][source]
            sources[source] = {**totals, "rows": count, "weight_sum": count*SAMPLE_WEIGHT,
                "loss": totals["policy_loss"]+self.kl_beta*totals["kl"]+self.value_coefficient*totals["value_loss"]}
        return {"source_counts": dict(prepared["source_counts"]), "sample_weight": SAMPLE_WEIGHT,
            "source_losses": sources, "source_loss_reduction": self._config()["source_loss_report_reduction"]}
