"""Explicit v3 continuous mixed learning; no migration of historical v2 runs.

Scope pins record purpose, not a claim that every row is a target variant.
Dataset admission still requires the full verified/human source validators.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

from .learner import LearnerCoordinator
from .learner_release_store import content_sha256
from .mixed_learner_store import describe_mixed_dataset
from .mixed_objective import make_mixed_sampler, SOURCE_COUNTS, SAMPLE_WEIGHT, OBJECTIVE_KIND
from .continual_mixed_store import ContinualMixedStore, SAMPLER_KIND, sampling_transition


class ContinualMixedLearner(LearnerCoordinator):
    BACKEND_KIND = "mixed-replay"
    RUN_SCHEMA = ContinualMixedStore.RUN_SCHEMA
    DATA_SCHEMA = ContinualMixedStore.DATA_SCHEMA
    OBJECTIVE_KIND = OBJECTIVE_KIND

    def __init__(self, runtime, *, learner_id, replay_pins, mathlib_sft_pins, sampler_seed,
                 journal_root, initial_model_release_sha256=None, scope=None, implementation=None):
        self._configure(runtime, learner_id, journal_root)
        backend = runtime.backend
        config, self.sampler_state = make_mixed_sampler(replay_pins=replay_pins, mathlib_sft_pins=mathlib_sft_pins,
            seed=sampler_seed, max_distance=backend.max_distance,
            load_replay=backend._load_dataset, load_mathlib_sft=backend._load_mathlib_dataset)
        catalog = [describe_mixed_dataset(backend, source, pin)
            for source, pins in (("replay", replay_pins), ("mathlib_sft", mathlib_sft_pins)) for pin in pins]
        self.run = {"schema_version": self.RUN_SCHEMA, "role": "learner", "learner_id": learner_id,
            "backend_session_id": learner_id,
            "initialization": ({"kind": "base"} if initial_model_release_sha256 is None else
                {"kind": "learner_release", "release_sha256": initial_model_release_sha256}),
            "contract": runtime.actor.submit(backend.experience_contract),
            "seed": int.from_bytes(hashlib.sha256(learner_id.encode()).digest()[:8], "big") % 2**63,
            "scope": {"kind": "generalist"} if scope is None else deepcopy(scope), "catalog": catalog,
            "sampler": {"kind": SAMPLER_KIND, "config": config, "config_sha256": content_sha256(config),
                "initial_state": deepcopy(self.sampler_state)},
            "implementation": implementation or {"coordinator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
        self._start_run(catalog, initial_model_release_sha256)

    def _prepare_next(self, append_dataset_pins):
        if not isinstance(append_dataset_pins, (list, tuple)):
            raise ValueError("append pins must be an explicit ordered list or tuple")
        catalog = deepcopy(self.catalog)
        seen = {item["dataset_sha256"] for item in catalog}
        for pin in append_dataset_pins:
            if pin in seen:
                raise ValueError("catalog append cannot duplicate or replace a dataset")
            catalog.append(describe_mixed_dataset(self.runtime.backend, "replay", pin))
            seen.add(pin)
        # Admit every row including rows not selected this step. Keep the flat
        # descriptor prefix intact even though the pure sampler groups sources.
        if append_dataset_pins:
            backend = self.runtime.backend
            make_mixed_sampler(replay_pins=[d["dataset_sha256"] for d in catalog if d["source"] == "replay"],
                mathlib_sft_pins=[d["dataset_sha256"] for d in catalog if d["source"] == "mathlib_sft"],
                seed=self.run["sampler"]["config"]["seed"], max_distance=backend.max_distance,
                load_replay=backend._load_dataset, load_mathlib_sft=backend._load_mathlib_dataset)
        config, transition, refs, after = sampling_transition(self.run, self.catalog, catalog, self.sampler_state)
        self._prepared_extras = {"sampler_config_sha256": content_sha256(config), "sampler_config": config,
            "sampler_transition": transition, "source_counts": dict(SOURCE_COUNTS), "sample_weight": SAMPLE_WEIGHT}
        return catalog, refs, after

    def _data_extras(self):
        return deepcopy(self._prepared_extras)
