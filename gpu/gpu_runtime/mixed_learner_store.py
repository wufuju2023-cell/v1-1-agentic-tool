"""Fixed-catalog mixed learner storage, separate from verified-only v1.

This stdlib layer checks envelopes, typed provenance and the exact deterministic
9/1 schedule. Production admission and training must still use the full Lean
dataset loaders. Opaque tensor bytes are checked by the backend, not this store.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

from .experience_store import require_sha256
from .identifiers import validate_identifier
from .learner_release_store import (
    LearnerReleaseStore, DATA_FIELDS, CONTRACT_FIELDS, canonical_bytes,
    content_sha256, require, _keys, _integer, _object,
)
from .mixed_objective import (
    OBJECTIVE_KIND, REPLAY_PROFILE, SFT_PROFILE, SOURCE_COUNTS, SAMPLE_WEIGHT,
    STATE_SCHEMA, TOKENIZATION, next_mixed_batch,
)
from .verified_objective import VALUE_SEMANTICS

MIXED_RETURN = "negative_integer_verified_action_longest_branch; human_profile_linear_remaining_actions"
SOURCE_REDUCTION = "contribution_to_full_batch_mean; includes_1_over_10_weight"
REPLAY_FIELDS = {"source", "dataset_sha256", "profile", "rows", "replay_receipt_sha256",
    "trace_sha256", "source_session_id", "source_tree_id", "theorem_sha256", "source_model_release_sha256"}
SFT_FIELDS = {"source", "dataset_sha256", "profile", "rows", "mathlib_source", "mathlib_source_sha256",
    "source_info_sha256", "original_receipt_sha256", "capture_receipt_sha256", "replay_receipt_sha256",
    "trace_sha256", "source_policy_version"}


def _same(left, right):
    return canonical_bytes(left) == canonical_bytes(right)


def _number(value, label, *, positive=False, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value), f"finite {label} required")
    require(not positive or value > 0, f"positive {label} required")
    require(not nonnegative or value >= 0, f"nonnegative {label} required")


def describe_mixed_dataset(backend, source: str, pin: str) -> dict:
    """Admit real evidence from operator roots; never invent human actor fields."""
    require_sha256(pin, "mixed dataset")
    require(isinstance(source, str) and source in SOURCE_COUNTS, "unknown mixed dataset source")
    if source == "replay":
        from cpu_runtime.verified_trajectory import load_verified_dataset
        from cpu_runtime.verified_dataset_store import read_bundle
        directory = backend.dataset_root / pin
        dataset = load_verified_dataset(directory, expected_sha256=pin)
        bundle = read_bundle(directory)
        require(hashlib.sha256(bundle["dataset.json"]).hexdigest() == pin
                and hashlib.sha256(bundle["session.json"]).hexdigest() == dataset["inputs_sha256"]["session.json"],
                "generated dataset changed during provenance admission")
        session = json.loads(bundle["session.json"])
        result = {"source": source, "dataset_sha256": pin, "profile": dataset["profile"],
            "rows": len(dataset["rows"]), "replay_receipt_sha256": dataset["replay_receipt_sha256"],
            "trace_sha256": dataset["trace_sha256"], "source_session_id": dataset["session_id"],
            "source_tree_id": dataset["tree_id"], "theorem_sha256": dataset["theorem_sha256"],
            "source_model_release_sha256": (session.get("lineage") or {}).get("model_release_sha256")}
    else:
        from cpu_runtime.mathlib_trajectory import load_mathlib_dataset
        dataset = load_mathlib_dataset(backend.mathlib_dataset_root / pin, expected_sha256=pin)
        inputs = dataset["inputs_sha256"]
        result = {"source": source, "dataset_sha256": pin, "profile": dataset["profile"],
            "rows": len(dataset["rows"]), "mathlib_source": deepcopy(dataset["source"]),
            "mathlib_source_sha256": content_sha256(dataset["source"]), "source_info_sha256": inputs["source.json"],
            "original_receipt_sha256": inputs["original.receipt.json"],
            "capture_receipt_sha256": inputs["capture.receipt.json"],
            "replay_receipt_sha256": inputs["replay.receipt.json"], "trace_sha256": inputs["replay-trace.json"],
            "source_policy_version": dataset["source_policy_version"]}
    MixedLearnerStore._validate_descriptor(result)
    return result


def sampler_catalog(catalog: list[dict]) -> dict:
    """Project typed evidence descriptors to the sampler's ordered row catalog."""
    return {source: [{key: item[key] for key in ("dataset_sha256", "profile", "rows")}
        for item in catalog if item["source"] == source] for source in SOURCE_COUNTS}


class MixedLearnerStore(LearnerReleaseStore):
    RUN_SCHEMA = "reap.learner.run.v2"
    DATA_SCHEMA = "reap.learner.data-receipt.v2"
    DATA_FIELDS = DATA_FIELDS | {"sampler_config_sha256", "source_counts", "sample_weight"}
    OBJECTIVE_KIND = OBJECTIVE_KIND
    CONFIG_KEY = "mixed_config"
    SNAPSHOT_SCHEMA = "reap.gpu.mixed-replay-backend.v1"
    ACCEPTANCE_KIND = "mixed-training"

    def _validate_run(self, run):
        super()._validate_run(run)
        require(run["initialization"] == {"kind": "base"}, "first mixed profile admits base initialization only")
        sampler = run["sampler"]
        _keys(sampler, {"kind", "config", "config_sha256", "initial_state"}, "mixed run sampler")
        require(sampler["kind"] == "seeded_cyclic_9_1_v1", "mixed sampler kind mismatch")
        require(sampler["config_sha256"] == content_sha256(sampler["config"]), "mixed sampler config digest mismatch")
        config = sampler["config"]
        # Validate the full pure sampler contract and the exact zero state.
        next_mixed_batch(config, sampler["initial_state"])
        require(config["max_distance"] == run["contract"][self.CONFIG_KEY]["support"]["distance_max"],
                "mixed sampler support differs from training contract")
        require(_same(config["catalog"], sampler_catalog(run["catalog"])), "mixed sampler catalog differs from evidence catalog")
        require(sampler["initial_state"]["config_sha256"] == sampler["config_sha256"], "mixed zero state config differs")

    def _check_initialization(self, run_record):
        require(run_record["initialization"] == {"kind": "base"}, "mixed initialization cannot import another contract")

    @staticmethod
    def _validate_contract(contract):
        _keys(contract, (CONTRACT_FIELDS - {"verified_config"}) | {"mixed_config"}, "mixed contract")
        require(contract["backend"] == "mixed-replay" and contract["objective"] == OBJECTIVE_KIND,
                "explicit mixed backend/objective required")
        require_sha256(contract["base_sha256"], "mixed base")
        _integer(contract["hidden_size"], "hidden size", 1)
        _integer(contract["lora_rank"], "LoRA rank", 1)
        _number(contract["lora_alpha"], "LoRA alpha", positive=True)
        _number(contract["lora_dropout"], "LoRA dropout", nonnegative=True)
        require(contract["lora_dropout"] < 1, "LoRA dropout must be below one")
        targets = contract["target_modules"]
        require(isinstance(targets, list) and targets and all(isinstance(x, str) and x for x in targets)
                and len(set(targets)) == len(targets), "unique adapter targets required")
        require(contract["value_head"] == "linear-silu-linear-categorical-v1", "mixed categorical head mismatch")
        config = contract["mixed_config"]
        _keys(config, {"objective", "value_semantics", "base_tokenizer_sha256", "lora", "eos_token_id", "support",
            "head", "hidden_size", "value_loss", "policy_loss", "tokenization", "kl_reduction", "max_sequence_tokens",
            "max_batch_samples", "learning_rate", "value_learning_rate", "value_coefficient", "kl_beta", "max_grad_norm",
            "max_post_update_kl", "mixture", "source_loss_report_reduction"}, "mixed training config")
        require(config["objective"] == OBJECTIVE_KIND and config["value_semantics"] == VALUE_SEMANTICS
                and config["base_tokenizer_sha256"] == contract["base_sha256"]
                and type(config["hidden_size"]) is int and config["hidden_size"] == contract["hidden_size"], "mixed config identity differs")
        require(_same(config["lora"], {"rank": contract["lora_rank"], "alpha": contract["lora_alpha"],
            "dropout": contract["lora_dropout"], "target_modules": targets}), "nested mixed adapter differs")
        support = config["support"]
        _keys(support, {"distance_min", "distance_max", "return", "overflow"}, "mixed categorical support")
        require(type(support["distance_min"]) is int and support["distance_min"] == 1
                and type(support["distance_max"]) is int and 2 <= support["distance_max"] <= 4096
                and support["return"] == MIXED_RETURN and support["overflow"] == "reject", "mixed return semantics differ")
        _integer(config["eos_token_id"], "EOS token")
        require(type(config["max_sequence_tokens"]) is int and 2 <= config["max_sequence_tokens"] <= 4096
                and type(config["max_batch_samples"]) is int and config["max_batch_samples"] == 10, "mixed batch/sequence bounds differ")
        require(config["head"] == "linear-silu-linear-categorical"
                and config["value_loss"] == "categorical_cross_entropy_exact_integer_class"
                and config["policy_loss"] == "mean_over_sampled_rows_of_joint_tactic_plus_one_EOS_negative_log_probability"
                and config["tokenization"] == TOKENIZATION
                and config["kl_reduction"] == "mean_over_sampled_rows_of_prefix_token_sum_current_to_frozen_base",
                "mixed loss/tokenization/KL contract differs")
        for key in ("learning_rate", "value_learning_rate", "max_grad_norm"):
            _number(config[key], key, positive=True)
        for key in ("kl_beta", "value_coefficient"):
            _number(config[key], key, nonnegative=True)
        if config["max_post_update_kl"] is not None:
            _number(config["max_post_update_kl"], "KL maximum", positive=True)
        expected_mixture = {"source_counts": SOURCE_COUNTS, "sample_weight": SAMPLE_WEIGHT,
            "sampling_unit": "verified_action_row", "ratio_scope": "every_complete_batch",
            "source_profiles": {"replay": REPLAY_PROFILE, "mathlib_sft": SFT_PROFILE},
            "source_losses": {source: ["policy", "value"] for source in SOURCE_COUNTS}}
        require(_same(config["mixture"], expected_mixture) and config["source_loss_report_reduction"] == SOURCE_REDUCTION,
                "mixed source ratio/loss choices differ")

    @staticmethod
    def _validate_descriptor(item):
        require(isinstance(item, dict), "typed mixed catalog item required")
        source = item.get("source")
        require(isinstance(source, str) and source in SOURCE_COUNTS, "unknown mixed catalog source")
        _keys(item, REPLAY_FIELDS if source == "replay" else SFT_FIELDS, "typed mixed descriptor")
        require_sha256(item["dataset_sha256"], "dataset")
        _integer(item["rows"], "dataset rows", 1)
        require(item["profile"] == (REPLAY_PROFILE if source == "replay" else SFT_PROFILE), "mixed descriptor profile differs")
        for key in ("replay_receipt_sha256", "trace_sha256"):
            require_sha256(item[key], key)
        if source == "replay":
            validate_identifier(item["source_session_id"], kind="source_session_id")
            validate_identifier(item["source_tree_id"], kind="source_tree_id")
            require_sha256(item["theorem_sha256"], "source theorem")
            if item["source_model_release_sha256"] is not None:
                require_sha256(item["source_model_release_sha256"], "source release")
        else:
            for key in ("mathlib_source_sha256", "source_info_sha256", "original_receipt_sha256", "capture_receipt_sha256"):
                require_sha256(item[key], key)
            _object(item["mathlib_source"], "Mathlib original provenance")
            require(item["mathlib_source"].get("source_kind") == "mathlib_sft"
                    and item["mathlib_source"].get("generator_events") == "none"
                    and item["source_policy_version"] is None, "human evidence cannot claim actor behavior")
            require(item["mathlib_source_sha256"] == content_sha256(item["mathlib_source"]), "human source digest differs")

    @staticmethod
    def _validate_catalog(catalog):
        require(isinstance(catalog, list) and catalog, "ordered mixed catalog required")
        seen, sources = set(), set()
        for item in catalog:
            MixedLearnerStore._validate_descriptor(item)
            require(item["dataset_sha256"] not in seen, "duplicate or cross-source dataset pin")
            seen.add(item["dataset_sha256"]); sources.add(item["source"])
        require(sources == set(SOURCE_COUNTS), "both fixed mixed source catalogs are required")

    @staticmethod
    def _validate_sampler(state, step):
        _keys(state, {"schema_version", "config_sha256", "step", "replay_cursor", "mathlib_sft_cursor"}, "mixed sampler state")
        require(state["schema_version"] == STATE_SCHEMA, "mixed sampler state schema differs")
        require_sha256(state["config_sha256"], "sampler config")
        for key in ("step", "replay_cursor", "mathlib_sft_cursor"):
            _integer(state[key], key)
        require(state["step"] == step and state["replay_cursor"] == step*9
                and state["mathlib_sft_cursor"] == step, "mixed cursor/step mismatch")

    def _validate_training_batch(self, run, data, detail, sampler, parent, step):
        require(_same(data["catalog"], run["catalog"]), "mixed catalog is fixed; append/reorder requires a new run")
        before = run["sampler"]["initial_state"] if parent is None else parent["sampler_state"]
        config = run["sampler"]["config"]
        refs, expected_after = next_mixed_batch(config, before)
        require(_same(data["batch_refs"], refs) and _same(data["event"]["samples"], refs), "mixed event differs from exact deterministic 9/1 batch")
        self._validate_sampler(sampler, step)
        require(_same(sampler, expected_after), "mixed next sampler state differs")
        require(data["sampler_config_sha256"] == run["sampler"]["config_sha256"]
                and data["sampler_before_sha256"] == content_sha256(before)
                and data["sampler_after_sha256"] == content_sha256(sampler), "mixed sampler receipt binding differs")
        self._validate_resolved_batch(run, data, detail, refs, run["catalog"], config)

    def _validate_resolved_batch(self, run, data, detail, refs, catalog, config):
        require(_same(data["source_counts"], SOURCE_COUNTS) and _same(detail.get("source_counts"), SOURCE_COUNTS)
                and _same(data["sample_weight"], SAMPLE_WEIGHT) and _same(detail.get("sample_weight"), SAMPLE_WEIGHT), "mixed receipt ratio/weight differs")
        samples = detail.get("samples")
        require(isinstance(samples, list) and len(samples) == 10, "mixed receipt requires ten resolved samples")
        descriptors = {item["dataset_sha256"]: item for item in catalog}
        for ref, row in zip(refs, samples):
            require(isinstance(row, dict) and _same({key: row.get(key) for key in ref}, ref), "mixed training source/pin/row differs")
            descriptor = descriptors[ref["dataset_sha256"]]
            common = {"source", "dataset_sha256", "row", "return", "distance", "value_class", "source_policy_version"}
            if ref["source"] == "replay":
                _keys(row, common | {"source_session_id", "source_tree_id", "source_theorem_sha256", "node_index"}, "generated training row")
                require(row["source_session_id"] == descriptor["source_session_id"]
                        and row["source_tree_id"] == descriptor["source_tree_id"]
                        and row["source_theorem_sha256"] == descriptor["theorem_sha256"], "generated row provenance differs")
                _integer(row["source_policy_version"], "source behavior version")
                _integer(row["node_index"], "source node")
            else:
                _keys(row, common | {"mathlib_source"}, "human training row")
                require(row["source_policy_version"] is None and _same(row["mathlib_source"], descriptor["mathlib_source"]),
                        "human row provenance differs or invented actor version")
            distance = row["distance"]
            _integer(distance, "resolved distance", 1)
            require(distance <= config["max_distance"] and type(row["return"]) is int and row["return"] == -distance
                    and type(row["value_class"]) is int and row["value_class"] == distance-1, "resolved negative return/class differs")
        self._validate_source_losses(detail, run["contract"][self.CONFIG_KEY])

    def _validate_checkpoint(self, run, run_sha, logical, backend, data, sampler, parent, *, backend_checked=False):
        step = super()._validate_checkpoint(run, run_sha, logical, backend, data, sampler, parent,
                                          backend_checked=backend_checked)
        require(logical["lineage"] == {}, "base-initialized mixed learner cannot claim release lineage")
        self._validate_mixed_counters(run, logical, data, step)
        return step

    def _validate_mixed_counters(self, run, logical, data, step):
        detail = data["runtime_receipt"]["detail"]
        require(type(detail.get("examples_seen")) is int and detail["examples_seen"] == step*10,
                "mixed examples counter must equal ten rows per committed step")
        require(logical["adapter_metadata"].get("objective") == OBJECTIVE_KIND
                and type(logical["adapter_metadata"].get("examples_seen")) is int
                and logical["adapter_metadata"]["examples_seen"] == step*10
                and _same(logical["value_metadata"], run["contract"][self.CONFIG_KEY]),
                "logical mixed metadata differs from committed training contract/counters")

    @staticmethod
    def _validate_source_losses(detail, config):
        require(detail.get("source_loss_reduction") == SOURCE_REDUCTION, "mixed source loss reduction differs")
        sources = detail.get("source_losses")
        _keys(sources, SOURCE_COUNTS, "mixed source loss report")
        for source, count in SOURCE_COUNTS.items():
            values = sources[source]
            _keys(values, {"policy_loss", "value_loss", "kl", "loss", "rows", "weight_sum"}, "source loss contribution")
            require(type(values["rows"]) is int and values["rows"] == count
                    and _same(values["weight_sum"], count*SAMPLE_WEIGHT), "source loss row/weight differs")
            for key in ("policy_loss", "value_loss", "kl", "loss"):
                _number(values[key], key)
            expected = values["policy_loss"] + config["kl_beta"]*values["kl"] + config["value_coefficient"]*values["value_loss"]
            require(math.isclose(values["loss"], expected, rel_tol=1e-6, abs_tol=1e-6), "source total loss differs")
        for key in ("policy_loss", "value_loss", "kl", "loss"):
            _number(detail.get(key), "batch "+key)
            require(math.isclose(sum(values[key] for values in sources.values()), detail[key], rel_tol=1e-6, abs_tol=1e-6),
                    "source contributions do not sum to batch loss")
