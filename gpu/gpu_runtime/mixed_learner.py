"""Explicit fixed-catalog 9/1 learner using the shared durable commit protocol.

No implicit migration of verified-only releases, catalog append, or sampler
reset. Both sources train policy and categorical value; the contract records
this engineering choice. A failed/unknown step retains its durable intent.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

from .learner import LearnerCoordinator
from .learner_release_store import content_sha256
from .mixed_learner_store import describe_mixed_dataset
from .mixed_objective import OBJECTIVE_KIND, SOURCE_COUNTS, SAMPLE_WEIGHT, make_mixed_sampler, next_mixed_batch


class MixedLearnerCoordinator(LearnerCoordinator):
    BACKEND_KIND = "mixed-replay"
    RUN_SCHEMA = "reap.learner.run.v2"
    DATA_SCHEMA = "reap.learner.data-receipt.v2"
    OBJECTIVE_KIND = OBJECTIVE_KIND

    def __init__(self, runtime, *, learner_id, replay_pins, mathlib_sft_pins,
                 sampler_seed, journal_root, scope=None, implementation=None):
        self._configure(runtime, learner_id, journal_root)
        backend = runtime.backend
        config, self.sampler_state = make_mixed_sampler(
            replay_pins=replay_pins, mathlib_sft_pins=mathlib_sft_pins,
            seed=sampler_seed, max_distance=backend.max_distance,
            load_replay=backend._load_dataset, load_mathlib_sft=backend._load_mathlib_dataset)
        catalog = [describe_mixed_dataset(backend, source, pin)
            for source, pins in (("replay", replay_pins), ("mathlib_sft", mathlib_sft_pins)) for pin in pins]
        self.run = {"schema_version": self.RUN_SCHEMA, "role": "learner",
            "learner_id": learner_id, "backend_session_id": learner_id,
            "initialization": {"kind": "base"},
            "contract": runtime.actor.submit(backend.experience_contract),
            "seed": int.from_bytes(hashlib.sha256(learner_id.encode()).digest()[:8], "big") % 2**63,
            "scope": scope or {"kind": "generalist"}, "catalog": catalog,
            "sampler": {"kind": config["kind"], "config": config,
                "config_sha256": content_sha256(config), "initial_state": deepcopy(self.sampler_state)},
            "implementation": implementation or {
                "coordinator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
        self._start_run(catalog)

    def _prepare_next(self, append_dataset_pins):
        if append_dataset_pins:
            raise ValueError("mixed learner has a fixed catalog; append requires an explicit new run/profile")
        batch, after = next_mixed_batch(self.run["sampler"]["config"], self.sampler_state)
        return deepcopy(self.catalog), batch, after

    def _data_extras(self):
        return {"sampler_config_sha256": self.run["sampler"]["config_sha256"],
                "source_counts": dict(SOURCE_COUNTS), "sample_weight": SAMPLE_WEIGHT}
