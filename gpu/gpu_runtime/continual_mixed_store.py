"""Opt-in v3 mixed runs: immutable source releases and append-only replay.

Old v2 objects are read using their original validator, never upgraded. The
training contract and its exact 9/1 objective do not change with this schema.
"""
from copy import deepcopy

from .learner_release_store import LearnerReleaseStore, content_sha256, require, _keys
from .mixed_learner_store import MixedLearnerStore, sampler_catalog, _same
from .mixed_objective import next_mixed_batch

SAMPLER_KIND = "append_replay_seeded_cyclic_9_1_v1"


def sampling_transition(run, previous_catalog, catalog, before):
    """Keep cumulative cursors, explicitly bind them to the enlarged catalog.

    Changing the replay row count changes its modulo mapping. This is not a
    promise to keep the old replay order; the unchanged human stream continues.
    """
    MixedLearnerStore._validate_catalog(previous_catalog)
    MixedLearnerStore._validate_catalog(catalog)
    require(len(catalog) >= len(previous_catalog) and
            _same(catalog[:len(previous_catalog)], previous_catalog), "replay catalog must preserve the exact flat prefix")
    added = catalog[len(previous_catalog):]
    require(all(item["source"] == "replay" for item in added), "only verified replay may be appended; human catalog is fixed")
    previous_config = deepcopy(run["sampler"]["config"])
    previous_config["catalog"] = sampler_catalog(previous_catalog)
    # This checks the actual committed state against its original configuration.
    next_mixed_batch(previous_config, before)
    config = deepcopy(previous_config)
    config["catalog"] = sampler_catalog(catalog)
    effective = {**before, "config_sha256": content_sha256(config)}
    refs, after = next_mixed_batch(config, effective)
    transition = {"kind": "append_replay_v1", "previous_config_sha256": content_sha256(previous_config),
        "config_sha256": content_sha256(config), "added_dataset_pins": [item["dataset_sha256"] for item in added],
        "effective_sampler_before": effective, "effective_sampler_before_sha256": content_sha256(effective)}
    return config, transition, refs, after


class ContinualMixedStore(MixedLearnerStore):
    RUN_SCHEMA = "reap.learner.run.v3"
    DATA_SCHEMA = "reap.learner.data-receipt.v3"
    DATA_FIELDS = MixedLearnerStore.DATA_FIELDS | {"sampler_config", "sampler_transition"}

    def _legacy(self):
        return MixedLearnerStore(self.root)

    def create_run(self, run_record):
        require(run_record.get("schema_version") == self.RUN_SCHEMA, "v3 store only creates v3 runs")
        return super().create_run(run_record)

    def create_checkpoint(self, run_sha, logical_state, backend_state, data_receipt, sampler_state,
                          parent_checkpoint_sha256=None):
        require(self.load_run(run_sha)["schema_version"] == self.RUN_SCHEMA, "v2 runs are read-only through the v3 store")
        return super().create_checkpoint(run_sha, logical_state, backend_state, data_receipt, sampler_state,
                                         parent_checkpoint_sha256)

    def publish(self, checkpoint_pin, weights):
        require(self.load_checkpoint(checkpoint_pin)["run"]["schema_version"] == self.RUN_SCHEMA,
                "v2 checkpoints are read-only through the v3 store")
        return super().publish(checkpoint_pin, weights)

    def _validate_run(self, run):
        # Dispatch on the declared, content-pinned schema. Never retry failed
        # validation as another schema.
        if run.get("schema_version") == MixedLearnerStore.RUN_SCHEMA:
            return self._legacy()._validate_run(run)
        LearnerReleaseStore._validate_run(self, run)
        require(run["learner_id"] == run["backend_session_id"], "v3 learner and backend session identity must agree")
        sampler = run["sampler"]
        _keys(sampler, {"kind", "config", "config_sha256", "initial_state"}, "continual mixed sampler")
        require(sampler["kind"] == SAMPLER_KIND and sampler["config_sha256"] == content_sha256(sampler["config"]),
                "continual mixed sampler kind/config differs")
        next_mixed_batch(sampler["config"], sampler["initial_state"])
        require(_same(sampler["config"]["catalog"], sampler_catalog(run["catalog"])) and
                sampler["config"]["max_distance"] == run["contract"][self.CONFIG_KEY]["support"]["distance_max"],
                "initial sampler catalog/support differs")
        scope = run["scope"]
        require(scope.get("kind") in {"generalist", "specialist"}, "explicit generalist/specialist purpose required")
        if scope["kind"] == "specialist":
            from .experience_store import require_sha256
            _keys(scope, {"kind", "problem_sha256", "curriculum_sha256"}, "specialist purpose")
            require_sha256(scope["problem_sha256"], "specialist problem")
            require_sha256(scope["curriculum_sha256"], "specialist curriculum")
            require(run["initialization"]["kind"] == "learner_release", "specialist requires an explicit source release")
        else:
            _keys(scope, {"kind"}, "generalist purpose")

    def _check_initialization(self, run_record):
        if run_record["schema_version"] == MixedLearnerStore.RUN_SCHEMA:
            return self._legacy()._check_initialization(run_record)
        if run_record["initialization"]["kind"] == "learner_release":
            metadata, _ = self.load_release(run_record["initialization"]["release_sha256"])
            require(content_sha256(metadata["contract"]) == content_sha256(run_record["contract"]),
                    "initial release contract differs")
            require(metadata["source"]["learner_id"] != run_record["backend_session_id"],
                    "new learner identity must differ from source learner")

    def _validate_training_batch(self, run, data, detail, sampler, parent, step):
        before = run["sampler"]["initial_state"] if parent is None else parent["sampler_state"]
        previous_catalog = run["catalog"] if parent is None else parent["data_receipt"]["catalog"]
        config, transition, refs, expected_after = sampling_transition(run, previous_catalog, data["catalog"], before)
        require(_same(data["sampler_config"], config) and _same(data["sampler_transition"], transition),
                "catalog configuration transition differs")
        require(data["sampler_config_sha256"] == content_sha256(config) and
                data["sampler_before_sha256"] == content_sha256(before) and
                data["sampler_after_sha256"] == content_sha256(sampler), "actual parent/new sampler binding differs")
        self._validate_sampler(sampler, step)
        require(_same(sampler, expected_after) and _same(data["batch_refs"], refs) and
                _same(data["event"]["samples"], refs), "continual mixed batch/state differs from exact 9/1 schedule")
        self._validate_resolved_batch(run, data, detail, refs, data["catalog"], config)

    def _validate_checkpoint(self, run, run_sha, logical, backend, data, sampler, parent, *, backend_checked=False):
        if run["schema_version"] == MixedLearnerStore.RUN_SCHEMA:
            return self._legacy()._validate_checkpoint(run, run_sha, logical, backend, data, sampler, parent,
                                                       backend_checked=backend_checked)
        step = LearnerReleaseStore._validate_checkpoint(self, run, run_sha, logical, backend, data, sampler, parent,
                                                        backend_checked=backend_checked)
        initialization = run["initialization"]
        expected_lineage = {}
        if initialization["kind"] == "learner_release":
            source, _ = self.load_release(initialization["release_sha256"])
            expected_lineage = {"model_release_sha256": initialization["release_sha256"],
                "weights_sha256": source["weights_sha256"], "source": source["source"],
                "reset": ["optimizer", "rng", "buffer", "policy_version", "event_receipts"]}
        require(_same(logical["lineage"], expected_lineage), "new learner lineage differs from immutable source release")
        self._validate_mixed_counters(run, logical, data, step)
        return step
