"""Opt-in strict search objective; the legacy backend and server stay unchanged.

One event produces one transactional optimizer step through GpuRuntime. Runtime
owns rollback/quarantine. This subclass reuses named adapters, frozen base,
session RNG and tensor snapshots, but rejects cross-objective snapshot imports.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from .backend import BackendLearnResult
from .real_backend import RealProverBackend
from .search_objective import (
    DEFAULT_VALUE_FLOOR, MAX_CANDIDATES, OBJECTIVE_KIND, VALUE_SEMANTICS,
    finite_number, nonnegative_integer, prepare_search_event, validate_gamma,
    validate_value_floor, value_to_distance,
)

SNAPSHOT_SCHEMA = "reap.gpu.real-search-backend.v1"
TOKENIZATION = "canonical_tactic_separate_tokenization_no_implicit_EOS_joint_token_sum"
KL_REDUCTION = "visit_weighted_sum_of_prefix_next_token_KL_current_to_frozen_base"


class KLGuardExceeded(ValueError):
    """A tentative optimizer step must be rolled back by GpuRuntime."""

    def __init__(self, detail: dict[str, Any]) -> None:
        self.detail = detail
        super().__init__(f"post-update KL {detail['post_update_kl']:.17g} exceeds limit "
                         f"{detail['maximum']:.17g}; tentative update rejected; runtime rollback required")


class RealSearchBackend(RealProverBackend):
    # Also gives old tiny fixtures/instances an unchanged, disabled default.
    max_post_update_kl: float | None = None
    success_dataset_root: Path | None = None

    def __init__(self, model_path: str, *, gamma: float,
                 value_floor: float = DEFAULT_VALUE_FLOOR,
                 max_sequence_tokens: int = 4096, max_candidates: int = MAX_CANDIDATES,
                 max_post_update_kl: float | None = None,
                 success_dataset_root: Path | None = None,
                 **kwargs: Any) -> None:
        self.success_dataset_root = Path(success_dataset_root).absolute() if success_dataset_root is not None else None
        self.gamma = validate_gamma(gamma)  # no implicit search-discount default
        self.value_floor = validate_value_floor(value_floor)
        self.max_sequence_tokens = nonnegative_integer(max_sequence_tokens, "max_sequence_tokens")
        self.max_candidates = nonnegative_integer(max_candidates, "max_candidates")
        if self.max_sequence_tokens < 2 or not 1 <= self.max_candidates <= MAX_CANDIDATES:
            raise ValueError("invalid search training safety limits")
        if max_post_update_kl is not None:
            max_post_update_kl = finite_number(max_post_update_kl, "max_post_update_kl")
            if max_post_update_kl <= 0:
                raise ValueError("max_post_update_kl must be positive")
        self.max_post_update_kl = max_post_update_kl
        super().__init__(model_path, **kwargs)
        for name in ("learning_rate", "value_learning_rate", "max_grad_norm"):
            if finite_number(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("kl_beta", "value_coefficient"):
            if finite_number(getattr(self, name), name) < 0:
                raise ValueError(f"{name} must be nonnegative")

    def _search_config(self) -> dict[str, Any]:
        config = {"objective": OBJECTIVE_KIND, "value_semantics": VALUE_SEMANTICS,
                "gamma": self.gamma, "value_floor": self.value_floor,
                "max_distance": value_to_distance(0, self.gamma, value_floor=self.value_floor),
                "max_sequence_tokens": self.max_sequence_tokens, "max_candidates": self.max_candidates,
                "tokenization": TOKENIZATION, "policy_reduction": "visit_weighted_joint_token_sum",
                "kl_reduction": KL_REDUCTION,
                "kl_beta": self.kl_beta, "value_coefficient": self.value_coefficient,
                "learning_rate": self.learning_rate, "value_learning_rate": self.value_learning_rate,
                "max_grad_norm": self.max_grad_norm, "head": "sigmoid",
                "hidden_size": self.hidden_size, "lora_rank": self.lora_config.r}
        if self.max_post_update_kl is not None:
            config["kl_guard"] = {"maximum": self.max_post_update_kl, "reduction": KL_REDUCTION,
                                  "timing": "after_optimizer_step_before_commit",
                                  "reject_if": "strictly_greater",
                                  "numeric_policy": "reject_nonfinite_or_negative_without_clamping"}
        if self.success_dataset_root is not None:
            from .success_finalize_objective import CONTRACT
            config['success_finalization'] = dict(CONTRACT)
        return config

    def _value_metadata(self, **extra: Any) -> dict[str, Any]:
        return {"hidden_size": self.hidden_size, **self._search_config(), **extra}

    def create_session(self, session_id: str) -> dict[str, dict[str, Any]]:
        metadata = super().create_session(session_id)
        # Tanh has no parameters; replacing it preserves optimizer references,
        # initialization and private RNG exactly, while changing only this mode.
        self.sessions[session_id].value_head[-1] = self.torch.nn.Sigmoid()
        metadata["value"] = self._value_metadata()
        metadata["adapter"]["objective"] = OBJECTIVE_KIND
        return metadata

    def value(self, session_id: str, request: Any) -> float:
        probability = super().value(session_id, request)
        return value_to_distance(probability, self.gamma, value_floor=self.value_floor)

    def _search_value_loss(self, session: Any, hidden: Any,
                           example: dict[str, Any]) -> tuple[Any, Any, Any, dict[str, Any]]:
        """Return the legacy scalar value loss and its audit fields.

        The categorical search backend overrides this hook.  Keeping the
        scalar implementation here makes the default wire, target and loss
        byte-for-byte equivalent to the pre-existing real-search mode.
        """
        prediction = session.value_head(hidden).squeeze()
        target = self.torch.tensor(example["value_target"], device=self.device,
                                   dtype=self.torch.float32)
        loss = self.functional.mse_loss(prediction, target)
        return prediction, target, loss, {
            "prediction": float(prediction.detach().cpu()),
            "target": example["value_target"],
            "loss_kind": "scalar_mse_discounted_return",
        }

    def _success_value_loss(self, session: Any, hidden: Any,
                            row: dict[str, Any]) -> tuple[Any, Any, Any, dict[str, Any]]:
        prediction = session.value_head(hidden).squeeze()
        target = self.torch.tensor(row["value_target"], device=self.device,
                                   dtype=self.torch.float32)
        loss = self.functional.mse_loss(prediction, target)
        return prediction, target, loss, {
            "prediction": float(prediction.detach().cpu()),
            "target": row["value_target"],
            "distance": -row["return"],
            "loss_kind": "scalar_mse_discounted_return",
        }

    def export_session(self, session_id: str) -> dict[str, Any]:
        state = super().export_session(session_id)
        return {**state, "schema_version": SNAPSHOT_SCHEMA, "session_id": session_id,
                "search_config": self._search_config()}

    def import_session(self, session_id: str, state: dict[str, Any]) -> None:
        if (state.get("schema_version") != SNAPSHOT_SCHEMA or state.get("session_id") != session_id
                or state.get("search_config") != self._search_config()):
            raise ValueError("snapshot search objective/session/gamma/value semantics mismatch")
        legacy_wrapper = {**state, "schema_version": "reap.gpu.real-prover-backend.v1"}
        super().import_session(session_id, legacy_wrapper)

    def _assert_optimizer_scope(self, session_id: str, session: Any) -> list[Any]:
        adapter = self._adapter_parameters(session_id)
        parameters = adapter + list(session.value_head.parameters())
        allowed = {id(parameter) for parameter in parameters}
        optimized = [p for group in session.optimizer.param_groups for p in group["params"]]
        if len(optimized) != len({id(p) for p in optimized}) or {id(p) for p in optimized} != allowed:
            raise RuntimeError("optimizer scope differs from this session's adapter/value parameters")
        if any(p.requires_grad and id(p) not in allowed for p in self.model.parameters()):
            raise RuntimeError("base or another session's parameters unexpectedly require gradients")
        return parameters

    def experience_contract(self) -> dict[str, Any]:
        return {**super().experience_contract(), "backend": "real-search",
                "objective": OBJECTIVE_KIND, "value_head": "linear-silu-linear-sigmoid-v1",
                "search_config": self._search_config()}

    def _check_experience_snapshot(self, session_id: str, snapshot: dict[str, Any]) -> None:
        if (snapshot.get("schema_version") != SNAPSHOT_SCHEMA or snapshot.get("session_id") != session_id
                or snapshot.get("search_config") != self._search_config()
                or snapshot.get("experience_contract") != self.experience_contract()):
            raise ValueError("experience source identity/base/objective/gamma/value semantics mismatch")

    def initialize_from_experience(self, session_id: str, weights: dict[str, Any]) -> dict[str, Any]:
        metadata = super().initialize_from_experience(session_id, weights)
        metadata["adapter"]["objective"] = OBJECTIVE_KIND
        metadata["value"] = self._value_metadata(initialized_from_experience=True)
        return metadata

    def _tensor_manifest(self, value: Any) -> dict[str, Any]:
        """Hash every session tensor in 1 MiB CPU chunks, never copy/hash the base."""
        torch = self.torch
        tensors: dict[str, str] = {}
        metadata: dict[str, Any] = {}

        def walk(item: Any, path: str) -> None:
            if torch.is_tensor(item):
                digest = hashlib.sha256()
                digest.update(json.dumps({"shape": list(item.shape), "dtype": str(item.dtype)}, sort_keys=True).encode())
                raw = item.detach().contiguous().reshape(-1).view(torch.uint8)
                for start in range(0, raw.numel(), 1024 * 1024):
                    digest.update(raw[start:start + 1024 * 1024].cpu().numpy().tobytes())
                tensors[path] = digest.hexdigest()
            elif isinstance(item, dict):
                for key in sorted(item, key=str):
                    walk(item[key], path + "/" + str(key))
            elif isinstance(item, (tuple, list)):
                for index, part in enumerate(item):
                    walk(part, path + "/" + str(index))
            else:
                metadata[path] = item

        walk(value, "")
        encoded = json.dumps({"tensors": tensors, "metadata": metadata}, sort_keys=True, allow_nan=False).encode()
        return {"sha256": hashlib.sha256(encoded).hexdigest(), "tensors": tensors}

    def session_fingerprints(self, session_id: str) -> dict[str, Any]:
        """Explicit audit helper; tensor hashes cover only mutable session state."""
        session = self._activate(session_id)
        return {
            "adapter": self._tensor_manifest(self._adapter_state_dict(session_id)),
            "value_head": self._tensor_manifest(session.value_head.state_dict()),
            "optimizer": self._tensor_manifest(session.optimizer.state_dict()),
            "rng_counters": self._tensor_manifest({"cpu": session.cpu_rng_state,
                "device": session.device_rng_state, "seed": session.rng_seed,
                "optimizer_steps": session.optimizer_steps, "examples_seen": session.examples_seen}),
        }

    @staticmethod
    def _fingerprint_changes(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for group in before:
            old, new = before[group]["tensors"], after[group]["tensors"]
            changed = sorted(key for key in old.keys() & new.keys() if old[key] != new[key])
            added, removed = sorted(new.keys() - old.keys()), sorted(old.keys() - new.keys())
            # Compact receipt; all tensors were compared. Full manifests can be
            # obtained via session_fingerprints without 100s of browser polls.
            diffs = {key: [old.get(key), new.get(key)] for key in sorted(set(changed + added + removed))}
            result[group] = {"before_sha256": before[group]["sha256"], "after_sha256": after[group]["sha256"],
                "before_tensors": len(old), "after_tensors": len(new), "changed_tensors": len(changed),
                "added_tensors": len(added), "removed_tensors": len(removed),
                "changed_names_sample": changed[:4], "diff_manifest_sha256": hashlib.sha256(
                    json.dumps(diffs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
        return result

    def _measure_post_update_kl(self, prompt: dict[str, Any], tokenized: list[Any],
                                candidates: list[dict[str, Any]]) -> float:
        """Measure this event's prefix-distribution drift, not global policy KL.

        Match the loss metric: candidate visit weights, full-vocabulary
        KL(current || frozen base), sum over target-prefix token positions.
        No sampling, token averaging, length normalization, or implicit EOS.
        """
        torch = self.torch
        prompt_ids = prompt["input_ids"]
        result = 0.0
        with torch.no_grad():
            for candidate, target_ids in zip(candidates, tokenized):
                weight = candidate["target_probability"]
                if weight == 0:
                    continue
                ids = torch.cat((prompt_ids, target_ids), dim=1)
                attention = torch.cat((prompt.get("attention_mask", torch.ones_like(prompt_ids)),
                                       torch.ones_like(target_ids)), dim=1)
                length = target_ids.shape[1]
                current = self.model(input_ids=ids, attention_mask=attention, use_cache=False,
                                     logits_to_keep=length + 1, return_dict=True)
                logits = current.logits[:, -length - 1:-1, :].float()
                if logits.shape[1] != length:
                    raise RuntimeError("KL guard requires the exact target scoring positions")
                self._require_finite(logits, "post-update KL current logits")
                logs = self.functional.log_softmax(logits, dim=-1)
                expected_shape = logits.shape
                del current, logits
                with self.model.disable_adapter():
                    reference = self.model(input_ids=ids, attention_mask=attention, use_cache=False,
                                           logits_to_keep=length + 1, return_dict=True)
                    reference_logits = reference.logits[:, -length - 1:-1, :].float()
                    if reference_logits.shape != expected_shape:
                        raise RuntimeError("KL guard current/reference scoring shape mismatch")
                    self._require_finite(reference_logits, "post-update KL reference logits")
                    reference_logs = self.functional.log_softmax(reference_logits, dim=-1)
                    del reference, reference_logits
                kl = (logs.exp() * (logs - reference_logs)).sum()
                self._require_finite(kl, "post-update KL")
                result += weight * float(kl.cpu())
                del logs, reference_logs, kl
        return finite_number(result, "post-update KL")

    def learn(self, session_id: str, event: dict[str, Any]) -> BackendLearnResult:
        from .success_finalize_objective import OBJECTIVE_KIND as SUCCESS_KIND
        if event.get('kind') == SUCCESS_KIND:
            from .success_finalize_backend import learn_success
            return learn_success(self, session_id, event)
        session = self._activate(session_id)
        example = prepare_search_event(event, session_id=session_id, policy_version=session.optimizer_steps,
            gamma=self.gamma, value_floor=self.value_floor, max_candidates=self.max_candidates)
        torch = self.torch
        prompt = self._tokenize_prompt(example["prompt"])
        prompt_ids = prompt["input_ids"]
        if prompt_ids.shape[0] != 1 or prompt_ids.shape[1] < 1:
            raise ValueError("search training requires one nonempty prompt")
        tokenized = []
        for candidate in example["candidates"]:
            ids = self.tokenizer(candidate["tactic"], return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
            if ids.shape[0] != 1 or ids.shape[1] < 1 or prompt_ids.shape[1] + ids.shape[1] > self.max_sequence_tokens:
                raise ValueError("training sequence exceeds explicit limit or is empty; never truncate")
            tokenized.append(ids)
        parameters = self._assert_optimizer_scope(session_id, session)
        before = self.session_fingerprints(session_id)
        previous_use_cache = self.model.config.use_cache
        session.optimizer.zero_grad(set_to_none=True)
        policy_total = kl_total = 0.0
        candidate_trace = []
        guard_detail = None
        try:
            with self._session_rng(session):
                self.model.config.use_cache = False
                self.model.eval()  # gradients remain enabled; frozen-model dropout must not perturb reference KL
                session.value_head.train()
                for candidate, target_ids in zip(example["candidates"], tokenized):
                    weight = candidate["target_probability"]
                    trace = {**candidate, "target_tokens": target_ids.shape[1]}
                    if weight == 0:
                        trace["trained"] = False
                        candidate_trace.append(trace)
                        continue
                    ids = torch.cat((prompt_ids, target_ids), dim=1)
                    attention = torch.cat((prompt.get("attention_mask", torch.ones_like(prompt_ids)), torch.ones_like(target_ids)), dim=1)
                    length = target_ids.shape[1]
                    output = self.model(input_ids=ids, attention_mask=attention, use_cache=False,
                        logits_to_keep=length + 1, return_dict=True)
                    logits = output.logits[:, -length - 1:-1, :].float()
                    if logits.shape[1] != length:
                        raise RuntimeError("model did not return the exact target scoring positions")
                    self._require_finite(logits, "search policy logits")
                    log_probs = self.functional.log_softmax(logits, dim=-1)
                    nll = -log_probs.gather(-1, target_ids.unsqueeze(-1)).sum()
                    with torch.no_grad(), self.model.disable_adapter():
                        reference = self.model(input_ids=ids, attention_mask=attention, use_cache=False,
                            logits_to_keep=length + 1, return_dict=True)
                        reference_logits = reference.logits[:, -length - 1:-1, :].float()
                        self._require_finite(reference_logits, "search reference logits")
                        reference_logs = self.functional.log_softmax(reference_logits, dim=-1)
                    kl = (log_probs.exp() * (log_probs - reference_logs)).sum()
                    term = weight * (nll + self.kl_beta * kl)
                    self._require_finite({"nll": nll, "kl": kl, "weighted_loss": term}, "search candidate loss")
                    term.backward()
                    policy_total += weight * float(nll.detach().cpu())
                    kl_total += weight * float(kl.detach().cpu())
                    trace.update(trained=True, joint_token_nll=float(nll.detach().cpu()), prefix_kl_sum=float(kl.detach().cpu()))
                    candidate_trace.append(trace)
                    del output, logits, log_probs, reference, reference_logits, reference_logs, nll, kl, term
                # A single prompt-only value regression, not one per candidate.
                output = self.model(**prompt, output_hidden_states=True, logits_to_keep=1,
                                    use_cache=False, return_dict=True)
                prediction, target, value_loss, value_training = self._search_value_loss(
                    session, output.hidden_states[-1][:, -1, :].float(), example)
                self._require_finite({"prediction": prediction, "target": target, "loss": value_loss}, "search value")
                (self.value_coefficient * value_loss).backward()
                value_loss_number = float(value_loss.detach().cpu())
                prediction_number = float(prediction.detach().cpu())
                total = policy_total + self.kl_beta * kl_total + self.value_coefficient * value_loss_number
                self._require_finite(total, "total search loss")
                self._require_finite([p.grad for p in parameters if p.grad is not None], "search gradients")
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm, error_if_nonfinite=True)
                session.optimizer.step()
                self._require_finite(parameters, "search updated parameters")
                self._require_finite(session.optimizer.state_dict(), "search optimizer")
                if self.max_post_update_kl is not None:
                    # The value forward can retain all hidden-state outputs;
                    # release them before the extra, read-only guard forwards.
                    del output, prediction, target, value_loss
                    measurement_start = perf_counter()
                    post_kl = self._measure_post_update_kl(prompt, tokenized, example["candidates"])
                    post_kl = finite_number(post_kl, "post-update KL")
                    # No hidden tolerance changes the requested boundary. Even
                    # negative roundoff is rejected conservatively; this opt-in
                    # guard never reports a negative quantity as valid KL.
                    if post_kl < 0:
                        raise FloatingPointError("negative post-update KL; no clamping; runtime rollback required")
                    guard_detail = {"maximum": self.max_post_update_kl, "post_update_kl": post_kl,
                        "reduction": KL_REDUCTION, "timing": "after_optimizer_step_before_commit",
                        "scope": "current_event_candidate_prefixes_only", "reject_if": "strictly_greater",
                        "measurement_seconds": perf_counter() - measurement_start,
                        "measurement_clock": "host_perf_counter_wall_including_tensor_transfer",
                        "accepted": post_kl <= self.max_post_update_kl}
                    if not guard_detail["accepted"]:
                        raise KLGuardExceeded(guard_detail)
        finally:
            self.model.config.use_cache = previous_use_cache
            self.model.eval()
            session.value_head.eval()
            session.optimizer.zero_grad(set_to_none=True)
        session.optimizer_steps += 1
        session.examples_seen += 1
        after = self.session_fingerprints(session_id)
        detail = {"objective": OBJECTIVE_KIND, "loss": total, "policy_loss": policy_total,
            "kl": kl_total, "value_loss": value_loss_number, "value_before": prediction_number,
            "value_target": value_training["target"], "reward": example["reward"],
            "grad_norm": float(grad_norm.detach().cpu()), "optimizer_steps": session.optimizer_steps,
            "finite_loss": True, "finite_gradients": True, "finite_parameters": True,
            "finite_optimizer_state": True, "base_parameters_frozen": True,
            "base_content_hash_checked": False, "parameter_diffs": self._fingerprint_changes(before, after),
            "search_trace": {key: example[key] for key in ("tree_id", "step", "node_index", "policy_version", "gamma", "terminal_verified", "value_trace")},
            "candidate_targets": candidate_trace, "training_config": self._search_config(),
            "prompt_sha256": hashlib.sha256(example["prompt"].encode()).hexdigest()}
        if getattr(self, "_categorical_value_training", False):
            detail["value_training"] = value_training
        if guard_detail is not None:
            detail["kl_guard"] = guard_detail
        return BackendLearnResult(
            adapter_metadata={"kind": "lora", "rank": self.lora_config.r,
                "objective": OBJECTIVE_KIND, "examples_seen": session.examples_seen},
            value_metadata=self._value_metadata(last_loss=value_loss_number),
            optimizer_metadata={"kind": "AdamW", "steps": session.optimizer_steps}, detail=detail)
