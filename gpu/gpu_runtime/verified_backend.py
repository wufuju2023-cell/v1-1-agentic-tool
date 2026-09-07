"""Explicit successful-trajectory learner; not the online search-Q objective.

The finite categorical support and EOS convention are engineering choices,
recorded in each snapshot/experience contract. They do not claim to reproduce
AlphaProof's undisclosed support or its 3B encoder-decoder architecture.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .backend import BackendLearnResult
from .real_backend import RealProverBackend, TARGET_MODULES
from .search_backend import KLGuardExceeded, RealSearchBackend
from .search_objective import finite_number
from .verified_objective import OBJECTIVE_KIND, VALUE_SEMANTICS, prepare_verified_event, validate_support

SNAPSHOT_SCHEMA = "reap.gpu.verified-replay-backend.v1"


class VerifiedReplayBackend(RealProverBackend):
    OBJECTIVE_KIND = OBJECTIVE_KIND
    SNAPSHOT_SCHEMA = SNAPSHOT_SCHEMA
    BACKEND_KIND = "verified-replay"
    CONFIG_KEY = "verified_config"
    # Reuse the already tested state audit and optimizer scope checks. No search
    # target preparation, gamma conversion, or search learn method is inherited.
    _tensor_manifest = RealSearchBackend._tensor_manifest
    session_fingerprints = RealSearchBackend.session_fingerprints
    _fingerprint_changes = staticmethod(RealSearchBackend._fingerprint_changes)
    _assert_optimizer_scope = RealSearchBackend._assert_optimizer_scope
    _measure_prefix_kl = RealSearchBackend._measure_post_update_kl

    def __init__(self, model_path: str, *, dataset_root: str | Path,
                 max_distance: int, max_sequence_tokens: int = 4096,
                 max_post_update_kl: float | None = None, **kwargs: Any):
        self.max_distance = validate_support(max_distance)
        if type(max_sequence_tokens) is not int or not 2 <= max_sequence_tokens <= 4096:
            raise ValueError("verified sequence limit must be in 2..4096")
        self.max_sequence_tokens = max_sequence_tokens
        self.dataset_root = Path(dataset_root).absolute()
        if not self.dataset_root.is_dir() or any(p.is_symlink() or
                (hasattr(p, "is_junction") and p.is_junction()) for p in (self.dataset_root, *self.dataset_root.parents)):
            raise ValueError("verified dataset store must be an existing directory without links")
        if max_post_update_kl is not None:
            max_post_update_kl = finite_number(max_post_update_kl, "max_post_update_kl")
            if max_post_update_kl <= 0:
                raise ValueError("max_post_update_kl must be positive")
        self.max_post_update_kl = max_post_update_kl
        super().__init__(model_path, **kwargs)
        for key in ("learning_rate", "value_learning_rate", "max_grad_norm"):
            if finite_number(getattr(self, key), key) <= 0:
                raise ValueError(f"{key} must be positive")
        for key in ("kl_beta", "value_coefficient"):
            if finite_number(getattr(self, key), key) < 0:
                raise ValueError(f"{key} must be nonnegative")

    def _config(self) -> dict:
        return {"objective": self.OBJECTIVE_KIND, "value_semantics": VALUE_SEMANTICS,
            "base_tokenizer_sha256": RealProverBackend.experience_contract(self)["base_sha256"],
            "lora": {"rank": self.lora_config.r, "alpha": self.lora_config.lora_alpha,
                     "dropout": self.lora_config.lora_dropout, "target_modules": sorted(TARGET_MODULES)},
            "eos_token_id": self.tokenizer.eos_token_id,
            "support": {"distance_min": 1, "distance_max": self.max_distance,
                        "return": "negative_integer_longest_generated_action_branch", "overflow": "reject"},
            "head": "linear-silu-linear-categorical", "hidden_size": self.hidden_size,
            "value_loss": "categorical_cross_entropy_exact_integer_class",
            "policy_loss": "mean_over_sampled_rows_of_joint_tactic_plus_one_EOS_negative_log_probability",
            "tokenization": "canonical_tactic_separate_no_special_tokens_append_exactly_one_EOS",
            "kl_reduction": "mean_over_sampled_rows_of_prefix_token_sum_current_to_frozen_base",
            "max_sequence_tokens": self.max_sequence_tokens, "max_batch_samples": 32,
            "learning_rate": self.learning_rate, "value_learning_rate": self.value_learning_rate,
            "value_coefficient": self.value_coefficient, "kl_beta": self.kl_beta,
            "max_grad_norm": self.max_grad_norm, "max_post_update_kl": self.max_post_update_kl}

    def create_session(self, session_id: str) -> dict:
        metadata = super().create_session(session_id)
        try:
            session, torch = self.sessions[session_id], self.torch
            # This independent profile uses a distinct initialized head and a
            # fresh optimizer. Never replace either on an existing learner.
            with self._session_rng(session):
                session.value_head = torch.nn.Sequential(torch.nn.Linear(self.hidden_size, 256),
                    torch.nn.SiLU(), torch.nn.Linear(256, self.max_distance)).to(self.device, dtype=torch.float32)
                session.optimizer = torch.optim.AdamW([
                    {"params": self._adapter_parameters(session_id), "lr": self.learning_rate},
                    {"params": list(session.value_head.parameters()), "lr": self.value_learning_rate}])
            metadata["adapter"]["objective"] = self.OBJECTIVE_KIND
            metadata["value"] = self._config()
            return metadata
        except BaseException:
            self.delete_session(session_id)
            raise

    def value(self, session_id: str, request: Any) -> float:
        session, torch = self._activate(session_id), self.torch
        self.model.eval(); session.value_head.eval()
        with self._session_rng(session), torch.no_grad():
            output = self.model(**self._encoded_prompt(request), output_hidden_states=True,
                                logits_to_keep=1, use_cache=False, return_dict=True)
            logits = session.value_head(output.hidden_states[-1][:, -1, :].float())
            self._require_finite(logits, "verified value logits")
            support = torch.arange(1, self.max_distance+1, device=logits.device, dtype=logits.dtype)
            distance = float((logits.softmax(-1)*support).sum(-1).item())
            self._require_finite(distance, "verified expected distance")
            # Convex-combination roundoff at the endpoints only. This is not
            # training-target clipping; unsupported labels were already refused.
            return min(float(self.max_distance), max(1.0, distance))

    def export_session(self, session_id: str) -> dict:
        return {**super().export_session(session_id), "schema_version": self.SNAPSHOT_SCHEMA,
                "session_id": session_id, self.CONFIG_KEY: self._config()}

    def import_session(self, session_id: str, state: dict) -> None:
        if (state.get("schema_version") != self.SNAPSHOT_SCHEMA or state.get("session_id") != session_id
                or state.get(self.CONFIG_KEY) != self._config()):
            raise ValueError("verified snapshot session/objective/support/optimizer contract mismatch")
        super().import_session(session_id, {**state, "schema_version": "reap.gpu.real-prover-backend.v1"})

    def experience_contract(self) -> dict:
        return {**super().experience_contract(), "backend": self.BACKEND_KIND, "objective": self.OBJECTIVE_KIND,
                "value_head": "linear-silu-linear-categorical-v1", self.CONFIG_KEY: self._config()}

    def _check_experience_snapshot(self, session_id: str, snapshot: dict) -> None:
        if (snapshot.get("schema_version") != self.SNAPSHOT_SCHEMA or snapshot.get("session_id") != session_id
                or snapshot.get(self.CONFIG_KEY) != self._config()
                or snapshot.get("experience_contract") != self.experience_contract()):
            raise ValueError("verified experience source/session/objective/support mismatch")

    def initialize_from_experience(self, session_id: str, weights: dict) -> dict:
        metadata = super().initialize_from_experience(session_id, weights)
        metadata["adapter"]["objective"] = self.OBJECTIVE_KIND
        metadata["value"] = {**self._config(), "initialized_from_experience": True}
        return metadata

    def extract_checkpoint_weights(self, logical_state: dict, snapshot: dict) -> dict:
        """Extract parameters from a learner checkpoint, never a proof candidate.

        The caller verifies the immutable checkpoint and its training receipts.
        No live source session is needed. Shapes are checked against the frozen
        model's bootstrap adapter and this profile's explicit categorical head.
        """
        import base64
        import io
        if (logical_state.get("role") != "learner" or logical_state.get("theorem_id") is not None
                or logical_state.get("completed") is not False
                or logical_state.get("buffer_metadata", {}).get("pending_event_ids")
                or type(logical_state.get("policy_version")) is not int or logical_state["policy_version"] < 1
                or snapshot.get("schema_version") != self.SNAPSHOT_SCHEMA
                or snapshot.get("session_id") != logical_state.get("session_id")
                or snapshot.get(self.CONFIG_KEY) != self._config()
                or snapshot.get("encoding") != "torch-save-base64"):
            raise ValueError("invalid learner checkpoint identity/objective/configuration")
        payload = self.torch.load(io.BytesIO(base64.b64decode(snapshot["payload"], validate=True)),
                                 map_location="cpu", weights_only=False)
        if (set(payload) != {"adapter", "value_head", "optimizer", "optimizer_steps", "examples_seen", "rng"}
                or type(payload["optimizer_steps"]) is not int
                or payload["optimizer_steps"] != logical_state["policy_version"]
                or type(payload["examples_seen"]) is not int or payload["examples_seen"] < 1):
            raise ValueError("learner checkpoint counters or private state mismatch")
        self._require_finite(payload, "learner checkpoint")
        weights = {key: payload[key] for key in ("adapter", "value_head")}
        expected = self._checkpoint_adapter_specs()
        if not isinstance(weights["adapter"], dict) or set(weights["adapter"]) != set(expected):
            raise ValueError("learner checkpoint adapter names mismatch")
        for name, (shape, dtype) in expected.items():
            tensor = weights["adapter"][name]
            if not self.torch.is_tensor(tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError("learner checkpoint adapter shape/dtype mismatch")
        head_shapes = {"0.weight": (256, self.hidden_size), "0.bias": (256,),
                       "2.weight": (self.max_distance, 256), "2.bias": (self.max_distance,)}
        if not isinstance(weights["value_head"], dict) or set(weights["value_head"]) != set(head_shapes):
            raise ValueError("learner checkpoint categorical head names mismatch")
        for name, shape in head_shapes.items():
            tensor = weights["value_head"][name]
            if not self.torch.is_tensor(tensor) or tuple(tensor.shape) != shape or tensor.dtype != self.torch.float32:
                raise ValueError("learner checkpoint categorical head shape/dtype mismatch")
        buffer = io.BytesIO(); self.torch.save(weights, buffer)
        return {"contract": self.experience_contract(), "encoding": "torch-save-base64",
                "payload": base64.b64encode(buffer.getvalue()).decode("ascii")}

    def _checkpoint_adapter_specs(self):
        """New-session LoRA uses each ordinary Linear base weight's dtype.

        PEFT 0.17.1 get_peft_model autocasts the bootstrap adapter to FP32,
        whereas add_adapter injects at the base layer dtype (BF16 here).
        Reading this mapping never activates or allocates another adapter.
        """
        specs = {}
        for name, tensor in self._adapter_state_dict("__bootstrap__").items():
            parts = name.rsplit(".", 2)
            if len(parts) != 3 or parts[1] not in {"lora_A", "lora_B"} or parts[2] != "weight":
                raise ValueError("unsupported checkpoint LoRA namespace")
            layer = self.model.get_submodule(parts[0]).get_base_layer()
            if type(layer) is not self.torch.nn.Linear or not layer.weight.is_floating_point():
                raise ValueError("checkpoint publication requires ordinary unquantized Linear base layers")
            specs[name] = (tuple(tensor.shape), layer.weight.dtype)
        return specs

    def _load_dataset(self, digest: str) -> dict:
        from cpu_runtime.verified_trajectory import load_verified_dataset
        # The digest is validated by prepare_verified_event before path use.
        # No cache: modified/corrupted bundle bytes must fail before any update.
        return load_verified_dataset(self.dataset_root/digest, expected_sha256=digest)

    def _prepare_event(self, session_id: str, policy_version: int, event: dict) -> dict:
        return prepare_verified_event(event, session_id=session_id, policy_version=policy_version,
            max_distance=self.max_distance, load_dataset=self._load_dataset)

    def _source_loss_totals(self, prepared: dict) -> dict:
        return {}

    def _source_loss_detail(self, prepared: dict, source_totals: dict) -> dict:
        return {}

    def learn(self, session_id: str, event: dict) -> BackendLearnResult:
        session = self._activate(session_id)
        prepared = self._prepare_event(session_id, session.optimizer_steps, event)
        return self._learn_prepared(session_id, session, prepared)

    def _learn_prepared(self, session_id: str, session: Any, prepared: dict) -> BackendLearnResult:
        torch = self.torch
        eos = self.tokenizer.eos_token_id
        if type(eos) is not int or eos < 0:
            raise ValueError("verified replay requires one explicit integer EOS token")
        tokenized = []
        for row in prepared["samples"]:
            prompt = self.tokenizer(row["prompt"], return_tensors="pt").to(self.device)
            target = self.tokenizer(row["tactic"], return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
            if target.shape[0] != 1 or not target.numel() or bool((target == eos).any()):
                raise ValueError("verified tactic must be nonempty and contain no preexisting EOS")
            target = torch.cat((target, torch.tensor([[eos]], device=self.device, dtype=target.dtype)), dim=1)
            if (prompt["input_ids"].shape[0] != 1 or prompt["input_ids"].shape[1] < 1
                    or prompt["input_ids"].shape[1]+target.shape[1] > self.max_sequence_tokens):
                raise ValueError("verified sequence exceeds explicit limit; never truncate")
            tokenized.append((row, prompt, target))
        parameters = self._assert_optimizer_scope(session_id, session)
        before = self.session_fingerprints(session_id)
        previous_cache = self.model.config.use_cache
        session.optimizer.zero_grad(set_to_none=True)
        totals = {"policy_loss": 0.0, "value_loss": 0.0, "kl": 0.0}
        source_totals = self._source_loss_totals(prepared)
        guard = None
        try:
            with self._session_rng(session):
                self.model.config.use_cache = False; self.model.eval(); session.value_head.train()
                for row, prompt, target in tokenized:
                    prompt_ids = prompt["input_ids"]; length = target.shape[1]
                    ids = torch.cat((prompt_ids, target), dim=1)
                    attention = torch.cat((prompt.get("attention_mask", torch.ones_like(prompt_ids)), torch.ones_like(target)), dim=1)
                    output = self.model(input_ids=ids, attention_mask=attention, use_cache=False,
                        logits_to_keep=length+1, output_hidden_states=True, return_dict=True)
                    raw = output.logits[:, -length-1:-1, :].float()
                    if raw.ndim != 3 or raw.shape[:2] != (1, length):
                        raise ValueError("verified current logits shape mismatch")
                    self._require_finite(raw, "verified policy logits")
                    logs = raw.log_softmax(-1)
                    nll = -logs.gather(-1, target.unsqueeze(-1)).sum()
                    with torch.no_grad(), self.model.disable_adapter():
                        reference = self.model(input_ids=ids, attention_mask=attention, use_cache=False,
                            logits_to_keep=length+1, return_dict=True)
                        ref_raw = reference.logits[:, -length-1:-1, :].float()
                        if ref_raw.shape != raw.shape:
                            raise ValueError("verified reference logits shape mismatch")
                        self._require_finite(ref_raw, "verified reference logits")
                        ref_logs = ref_raw.log_softmax(-1)
                    kl = (logs.exp()*(logs-ref_logs)).sum()
                    value_logits = session.value_head(output.hidden_states[-1][:, prompt_ids.shape[1]-1, :].float())
                    if value_logits.shape != (1, self.max_distance):
                        raise ValueError("verified categorical logits shape mismatch")
                    self._require_finite(value_logits, "verified categorical logits")
                    target_class = torch.tensor([row["value_class"]], device=self.device, dtype=torch.long)
                    value_loss = self.functional.cross_entropy(value_logits, target_class)
                    loss = (nll+self.kl_beta*kl+self.value_coefficient*value_loss)*prepared["sample_weight"]
                    self._require_finite({"policy": nll, "kl": kl, "value": value_loss, "loss": loss}, "verified loss")
                    loss.backward()
                    for key, value in (("policy_loss", nll), ("kl", kl), ("value_loss", value_loss)):
                        weighted = prepared["sample_weight"]*float(value.detach().cpu())
                        totals[key] += weighted
                        if source_totals:
                            source_totals[row["source"]][key] += weighted
                    del output, raw, logs, reference, ref_raw, ref_logs, nll, kl, value_logits, value_loss, loss
                self._require_finite([p.grad for p in parameters if p.grad is not None], "verified gradients")
                norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm, error_if_nonfinite=True)
                session.optimizer.step()
                self._require_finite(parameters, "verified updated parameters")
                self._require_finite(session.optimizer.state_dict(), "verified optimizer")
                if self.max_post_update_kl is not None:
                    measured = sum(self._measure_prefix_kl(prompt, [target], [{"target_probability": 1.0}])
                                   for _, prompt, target in tokenized)*prepared["sample_weight"]
                    measured = finite_number(measured, "verified post-update KL")
                    if measured < 0:
                        raise FloatingPointError("negative verified KL; no clamping")
                    guard = {"post_update_kl": measured, "maximum": self.max_post_update_kl,
                        "scope": "current_verified_batch_prefixes_only", "reduction": self._config()["kl_reduction"],
                        "timing": "after_optimizer_step_before_commit", "accepted": measured <= self.max_post_update_kl}
                    if not guard["accepted"]:
                        raise KLGuardExceeded(guard)
        finally:
            self.model.config.use_cache = previous_cache; self.model.eval(); session.value_head.eval()
            session.optimizer.zero_grad(set_to_none=True)
        session.optimizer_steps += 1; session.examples_seen += len(tokenized)
        after = self.session_fingerprints(session_id)
        detail = {**totals, "objective": self.OBJECTIVE_KIND, "training_config": self._config(),
            "loss": totals["policy_loss"]+self.kl_beta*totals["kl"]+self.value_coefficient*totals["value_loss"],
            "optimizer_steps": session.optimizer_steps, "examples_seen": session.examples_seen,
            "grad_norm": float(norm.detach().cpu()), "finite_loss": True, "finite_gradients": True,
            "finite_parameters": True, "finite_optimizer_state": True, "base_parameters_frozen": True,
            "base_content_hash_checked": False, "parameter_diffs": self._fingerprint_changes(before, after),
            "samples": [{k: v for k, v in row.items() if k not in ("prompt", "tactic")} for row in prepared["samples"]]}
        detail.update(self._source_loss_detail(prepared, source_totals))
        if guard is not None:
            detail["kl_guard"] = guard
        return BackendLearnResult(adapter_metadata={"kind": "lora", "objective": self.OBJECTIVE_KIND,
            "rank": self.lora_config.r, "examples_seen": session.examples_seen}, value_metadata=self._config(),
            optimizer_metadata={"kind": "AdamW", "steps": session.optimizer_steps}, detail=detail)
