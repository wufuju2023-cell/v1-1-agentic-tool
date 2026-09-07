"""Opt-in Qwen3.5 text TTT with a frozen, shared thought context per state.

This is a distinct policy: think once on first value/policy access to a state,
then sample Lean actions conditional on that exact token prefix. The thought
is not an action or a policy-loss target. Old states keep their first context
after updates; new states think with the current adapter. Contexts are private
session state, included in snapshots but never in cross-problem experience.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

from .real_backend import PolicyScoringConfig
from .search_backend import RealSearchBackend
from .search_objective import value_to_distance
from .lean_action_format import normalize_lean_action, SCHEMA as ACTION_NORMALIZER_SCHEMA


THINKING_MODE = "shared-state-context-v1"
INPUT_FORMAT = "lean-local-state-chat-v2"
STRICT_ACTION_FORMAT = "strict-single-wrapper-v1"
ACTION_TRACE_SCHEMA = "reap.qwen35.canonical-action-trace.v1"
MAX_ACTION_TRACE_BATCHES = 4096
REAP_PREFIX = (
    "User: Please generate a tactic in lean4 to solve the state.\n"
    "Here're some theorems that may be helpful:\n"
)
REAP_STATE_SEPARATOR = "\nSTATE:\n"
REAP_SUFFIX = "\nTACTIC:\n\nAssistant:"
SYSTEM_PROMPT = (
    "You are choosing the next tactic inside an already open Lean 4 proof. "
    "The user supplies available premises and the current local proof state, "
    "including local hypotheses and the goal. These local hypotheses already exist. "
    "Think about a useful next proof step, then output only the next Lean tactic "
    "or one short tactic sequence to apply directly to that state. "
    "The step may leave subgoals; you do not need to finish the entire proof at once. "
    "Do not restate the problem, redeclare its local hypotheses, write a complete "
    "theorem or example, open a new proof with by, or use a code fence or JSON. "
    "Do not use sorry or admit."
)
STRICT_SYSTEM_PROMPT = (
    "You are choosing exactly ONE next Lean 4 tactic inside an already open proof. "
    "The user supplies available premises and the current local proof state. "
    "Think about the next useful step, then output exactly ONE next tactic expression. "
    "Stop immediately after that tactic. Leave any remaining subgoals open: "
    "the search will supply the next state before you choose the next action. "
    "Do not append further tactics or additional top-level commands. "
    "Do not output a whole proof, theorem, example, code fence, JSON, an opening by, sorry, or admit. "
    "Do not restate the problem or redeclare hypotheses already in the local context. "
    "Text after ⊢ is the goal, not a list of available local hypotheses. "
    "Variables bound inside ∀ and assumptions to the left of → in that goal "
    "are not yet in the local context. Only names displayed before ⊢ are currently available."
    " The evaluator parses one Lean tactic expression, not a whole proof block. "
    "Output only that next tactic, then stop and wait for the next state."
)
CLOSE_THOUGHT = "</think>"
FORCED_CLOSE = "\n</think>\n\n"
LANGUAGE_TARGETS = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
})


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def local_state_chat_content(prompt: str) -> str:
    """Remove only the pinned Reap wrapper, without rewriting its math payload."""
    if not isinstance(prompt, str) or not prompt.startswith(REAP_PREFIX) or not prompt.endswith(REAP_SUFFIX):
        raise ValueError("lean-local-state-chat-v2 requires the exact Reap prompt wrapper")
    payload = prompt[len(REAP_PREFIX):-len(REAP_SUFFIX)]
    if payload.count(REAP_STATE_SEPARATOR) != 1:
        raise ValueError("Reap prompt requires one unambiguous STATE separator")
    premises, state = payload.split(REAP_STATE_SEPARATOR)
    if not state.strip():
        raise ValueError("Reap local proof state is empty")
    return "Available premises:\n" + premises + "\n\nCurrent local Lean proof state:\n" + state


class Qwen35SearchBackend(RealSearchBackend):
    """Separate explicit backend; no changes to REAL-Prover prompt defaults."""

    def __init__(self, model_path: str, *, gamma: float, thinking_budget: int,
                 thinking_mode: str = THINKING_MODE, max_contexts: int = 2048,
                 max_policy_tokens: int = 512, action_format: str = "raw-v1", **kwargs: Any) -> None:
        if thinking_mode != THINKING_MODE:
            raise ValueError("only explicit shared-state-context-v1 thinking is supported")
        if action_format not in {"raw-v1", STRICT_ACTION_FORMAT}:
            raise ValueError("unknown Qwen action format; wrapper handling must be explicit")
        self.action_format = action_format
        for name, value, maximum in (("thinking_budget", thinking_budget, 8192),
                                     ("max_contexts", max_contexts, 65536),
                                     ("max_policy_tokens", max_policy_tokens, 512)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [1, {maximum}]")
        if kwargs.get("expected_hidden_size", 4096) != 4096:
            raise ValueError("Qwen3.5 9B requires text hidden_size 4096")
        if kwargs.get("max_post_update_kl") is not None:
            raise ValueError("Qwen3.5 KL guard requires separate GPU acceptance")
        scoring = kwargs.get("policy_scoring") or PolicyScoringConfig()
        if not isinstance(scoring, PolicyScoringConfig) or scoring.mode != "tokenwise":
            raise ValueError("Qwen3.5 only supports its canonical action scoring path")
        self.thinking_budget = thinking_budget
        self.max_contexts = max_contexts
        self.max_policy_tokens = max_policy_tokens
        self._contexts: dict[str, dict[str, dict[str, Any]]] = {}
        self._active_context_session: str | None = None
        kwargs["expected_hidden_size"] = 4096
        super().__init__(model_path, gamma=gamma, **kwargs)
        close = self.tokenizer.encode(CLOSE_THOUGHT, add_special_tokens=False)
        if len(close) != 1 or self.tokenizer.decode(close, skip_special_tokens=False) != CLOSE_THOUGHT:
            raise ValueError("this backend requires a verified single-token </think> marker")
        self._close_token = close[0]
        self._template_sha256 = hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest()

    def _load_base_model(self, model_path: str) -> Any:
        config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
        if (config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]
                or config.get("model_type") != "qwen3_5"
                or config.get("text_config", {}).get("hidden_size") != 4096):
            raise ValueError("requires Qwen3_5ForConditionalGeneration with text hidden_size 4096")
        from transformers import Qwen3_5ForConditionalGeneration
        return Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=self.torch.bfloat16, low_cpu_mem_usage=True,
        ).to(self.device)

    def _base_hidden_size(self, base: Any) -> int:
        return int(base.config.text_config.hidden_size)

    def _lora_target_modules(self, base: Any) -> list[str]:
        # Exact names prevent suffix targeting from adapting the vision tower.
        targets = sorted(name for name, module in base.named_modules()
                         if "language_model" in name.split(".")
                         and name.rsplit(".", 1)[-1] in LANGUAGE_TARGETS
                         and isinstance(module, self.torch.nn.Linear))
        present = {name.rsplit(".", 1)[-1] for name in targets}
        if not LANGUAGE_TARGETS.issubset(present):
            raise ValueError(f"missing Qwen3.5 text LoRA modules: {sorted(LANGUAGE_TARGETS - present)}")
        self._language_targets = targets
        return targets

    def _search_config(self) -> dict[str, Any]:
        config = {**super()._search_config(), "input_format": INPUT_FORMAT, "thinking": {
            "mode": THINKING_MODE, "budget": self.thinking_budget,
            "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "template_sha256": self._template_sha256,
            "system_prompt_sha256": hashlib.sha256(self._system_prompt().encode()).hexdigest(),
            "force_close": FORCED_CLOSE, "eos_before_close": "reject",
            "state_key": "exact_joined_Reap_prompt", "old_context_after_update": "frozen",
            "training": "conditional_Lean_suffix_only_no_thought_loss",
            "candidate_scoring": "canonical_tactic_tokens_no_EOS_current_raw_logprob",
            "action_sampling": {"temperature": "request", "top_p": 1.0, "top_k": 50},
            "max_contexts": self.max_contexts, "max_policy_tokens": self.max_policy_tokens}}
        if self._strict_actions():
            config["action_format"] = {"mode": STRICT_ACTION_FORMAT, "normalizer_schema": ACTION_NORMALIZER_SCHEMA,
                "outer_whitespace": "strip_matches_existing_Qwen_policy", "layers": 1,
                "scoring": "returned_canonical_action_tokens_same_frozen_prefix_no_EOS",
                "audit_schema": ACTION_TRACE_SCHEMA, "audit_storage": "session_snapshot_context",
                "unscorable_candidate": "empty_Lean_action_with_preserved_rejection_evidence",
                "max_audit_batches": MAX_ACTION_TRACE_BATCHES}
        return config

    def _strict_actions(self) -> bool:
        return getattr(self, "action_format", "raw-v1") == STRICT_ACTION_FORMAT

    def _system_prompt(self) -> str:
        return STRICT_SYSTEM_PROMPT if self._strict_actions() else SYSTEM_PROMPT

    def experience_contract(self) -> dict[str, Any]:
        return {**super().experience_contract(), "backend": "qwen35-search",
                "target_modules": list(self._language_targets)}

    def create_session(self, session_id: str) -> dict[str, Any]:
        metadata = super().create_session(session_id)
        self._contexts[session_id] = {}
        return metadata

    def delete_session(self, session_id: str) -> None:
        super().delete_session(session_id)
        self._contexts.pop(session_id, None)
        if self._active_context_session == session_id:
            self._active_context_session = None

    def _activate(self, session_id: str) -> Any:
        session = super()._activate(session_id)
        self._active_context_session = session_id
        return session

    def _chat_ids(self, prompt: str) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self._system_prompt()},
             {"role": "user", "content": local_state_chat_content(prompt)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
        if not rendered.rstrip().endswith("<think>"):
            raise ValueError("Qwen template did not open the thinking segment")
        return self.tokenizer.encode(rendered, add_special_tokens=False)

    def _encoded_ids(self, ids: list[int]) -> dict[str, Any]:
        tensor = self.torch.tensor([ids], device=self.device, dtype=self.torch.long)
        return {"input_ids": tensor, "attention_mask": self.torch.ones_like(tensor)}

    def _generate_thought_tokens(self, ids: list[int]) -> list[int]:
        # This method is called inside the session RNG transaction.
        self.model.eval()
        with self.torch.no_grad():
            output = self.model.generate(
                **self._encoded_ids(ids), do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                num_return_sequences=1, max_new_tokens=self.thinking_budget,
                eos_token_id=sorted(self._eos_ids() | {self._close_token}),
                pad_token_id=self.tokenizer.pad_token_id, return_dict_in_generate=False,
                output_scores=False, output_logits=False)
        return output[0, len(ids):].tolist()

    def _eos_ids(self) -> set[int]:
        eos = self.tokenizer.eos_token_id
        return set(eos if isinstance(eos, (list, tuple)) else [eos])

    def _ensure_context(self, session_id: str, prompt: str) -> dict[str, Any]:
        session = self._activate(session_id)
        existing = self._contexts[session_id].get(prompt)
        if existing is not None:
            return existing
        if len(self._contexts[session_id]) >= self.max_contexts:
            raise ValueError("session thought-context limit reached; no implicit eviction")
        start = self._chat_ids(prompt)
        closure = self.tokenizer.encode(FORCED_CLOSE, add_special_tokens=False)
        if len(start) + self.thinking_budget + len(closure) + self.max_policy_tokens > self.max_sequence_tokens:
            raise ValueError("thought plus answer exceeds max_sequence_tokens; never truncate prompt")
        old_cpu = session.cpu_rng_state.clone()
        old_device = None if session.device_rng_state is None else session.device_rng_state.clone()
        try:
            with self._session_rng(session):
                started = perf_counter()
                thought = self._generate_thought_tokens(start)
                elapsed = perf_counter() - started
                if not thought or len(thought) > self.thinking_budget:
                    raise ValueError("invalid thought generation length")
                natural = thought[-1] == self._close_token
                if any(token in self._eos_ids() for token in thought):
                    raise ValueError("EOS before thinking closed; no automatic retry")
                if self._close_token in thought[:-1]:
                    raise ValueError("thought generator continued after close marker")
                if not natural and len(thought) != self.thinking_budget:
                    raise ValueError("thinking stopped before its budget without a close marker")
                suffix = self.tokenizer.encode("\n\n", add_special_tokens=False) if natural else closure
                prefix = start + thought + suffix
                record = {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                          "source_policy_version": session.optimizer_steps,
                          "prefix_ids": prefix, "prefix_sha256": _digest(prefix),
                          "thought_ids": thought, "natural_close": natural,
                          "controller_suffix_ids": suffix, "chat_prefix_tokens": len(start),
                          "thought_generation_seconds": elapsed}
            self._contexts[session_id][prompt] = record
            return record
        except BaseException:
            session.cpu_rng_state, session.device_rng_state = old_cpu, old_device
            self._contexts[session_id].pop(prompt, None)
            raise

    def _tokenize_prompt(self, prompt: str) -> dict[str, Any]:
        # Training is lookup-only: never invent a new behavior context during learn.
        session_id = self._active_context_session
        record = self._contexts.get(session_id, {}).get(prompt)
        if record is None:
            raise ValueError("missing frozen thought context for exact training/inference prompt")
        return self._encoded_ids(record["prefix_ids"])

    def context_receipts(self, session_id: str) -> list[dict[str, Any]]:
        self._session(session_id)
        return [{key: value for key, value in record.items()
                 if key not in {"prefix_ids", "thought_ids", "controller_suffix_ids", "policy_generations"}}
                | {"prefix_tokens": len(record["prefix_ids"]), "thought_tokens": len(record["thought_ids"])}
                | ({"action_audit_batches": len(record["policy_generations"]),
                    "action_audit_sha256": _digest(record["policy_generations"])} if "policy_generations" in record else {})
                for record in self._contexts[session_id].values()]

    def value(self, session_id: str, request: Any) -> float:
        self._ensure_context(session_id, request.prompt)
        session = self._activate(session_id)
        self.model.eval()
        with self.torch.no_grad():
            output = self.model(**self._tokenize_prompt(request.prompt), output_hidden_states=True,
                                logits_to_keep=1, use_cache=False, return_dict=True)
            hidden = output.hidden_states[-1][:, -1, :].float()
            score = session.value_head(hidden).squeeze().item()
        return value_to_distance(score, self.gamma, value_floor=self.value_floor)

    def initial_equivalence_error(self, session_id: str, prompt: str) -> float:
        self._ensure_context(session_id, prompt)
        encoded = self._tokenize_prompt(prompt)
        self.model.eval()
        with self.torch.no_grad():
            adapted = self.model(**encoded, logits_to_keep=1, return_dict=True).logits.float()
            with self.model.disable_adapter():
                reference = self.model(**encoded, logits_to_keep=1, return_dict=True).logits.float()
        return float((adapted - reference).abs().max().item())

    def _score_generated_tokens(self, encoded: dict[str, Any], token_ids: list[int]) -> list[dict[str, Any]]:
        # One bounded full-prefix forward avoids depending on Qwen2 cache/position
        # internals in a hybrid linear/full-attention Qwen3.5 model.
        if not token_ids:
            return []
        if len(token_ids) > self.max_policy_tokens:
            raise ValueError("canonical action exceeds explicit policy scoring budget")
        torch = self.torch
        target = torch.tensor([token_ids], device=self.device, dtype=torch.long)
        ids = torch.cat((encoded["input_ids"], target), dim=1)
        if ids.shape[1] > self.max_sequence_tokens:
            raise ValueError("policy scoring sequence exceeds limit")
        output = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                            logits_to_keep=len(token_ids) + 1, return_dict=True)
        logits = output.logits[:, -len(token_ids) - 1:-1, :].float()
        if logits.shape[1] != len(token_ids):
            raise RuntimeError("Qwen output does not preserve action scoring positions")
        scores = torch.log_softmax(logits, -1).gather(-1, target.unsqueeze(-1)).flatten()
        self._require_finite(scores, "Qwen action logprobs")
        return [{"token": self.tokenizer.decode([token], skip_special_tokens=False), "logprob": float(score)}
                for token, score in zip(token_ids, scores.cpu().tolist())]

    def policy(self, session_id: str, request: Any) -> tuple[list[str], list[list[dict[str, Any]]]]:
        if self._strict_actions():
            return self._policy_strict_actions(session_id, request)
        if request.max_tokens > self.max_policy_tokens:
            raise ValueError("Qwen answer budget exceeds max_policy_tokens")
        self._ensure_context(session_id, request.prompt)
        session = self._activate(session_id)
        encoded = self._tokenize_prompt(request.prompt)
        options = {"do_sample": request.temperature > 0,
                   "num_return_sequences": request.n if request.temperature > 0 else 1,
                   "max_new_tokens": request.max_tokens, "pad_token_id": self.tokenizer.pad_token_id,
                   "eos_token_id": self.tokenizer.eos_token_id, "return_dict_in_generate": False,
                   "output_scores": False, "output_logits": False}
        if request.temperature > 0:
            options["temperature"] = request.temperature
            options["top_p"], options["top_k"] = 1.0, 50
        contents, probabilities = [], []
        self.model.eval()
        with self._session_rng(session), self.torch.no_grad():
            sequences = self.model.generate(**encoded, **options)
            for sequence in sequences:
                tokens = sequence[encoded["input_ids"].shape[1]:].tolist()
                for index, token in enumerate(tokens):
                    if token in self._eos_ids():
                        if any(t != self.tokenizer.pad_token_id for t in tokens[index + 1:]):
                            raise ValueError("non-padding action tokens after EOS")
                        tokens = tokens[:index]
                        break
                if self._close_token in tokens:
                    raise ValueError("answer contains thinking marker; no silent extraction")
                action = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
                if "<think>" in action or CLOSE_THOUGHT in action:
                    raise ValueError("answer reopened thinking; no silent thought extraction")
                # Only whitespace is canonicalized. Fences/theorem/JSON are NOT
                # rewritten into a different tactic; Lean can reject them.
                canonical = self.tokenizer.encode(action, add_special_tokens=False)
                contents.append(action)
                probabilities.append(self._score_generated_tokens(encoded, canonical))
        if request.temperature <= 0 and request.n > 1:
            contents *= request.n
            probabilities *= request.n
        return contents, probabilities

    def _canonical_candidate(self, tokens: list[int], budget: int) -> dict[str, Any]:
        """Pure CPU candidate interpretation; malformed text cannot abort peers."""
        eos_position = next((i for i, token in enumerate(tokens) if token in self._eos_ids()), None)
        content_tokens = tokens if eos_position is None else tokens[:eos_position]
        finish = "eos" if eos_position is not None else "length" if len(tokens) == budget else "early_stop_without_eos"
        raw = self.tokenizer.decode(content_tokens, skip_special_tokens=False)
        normalization = normalize_lean_action(raw)
        action = normalization["canonical"].strip()
        canonical_ids = self.tokenizer.encode(action, add_special_tokens=False)
        rejected = None
        if len(tokens) > budget:
            rejected = "generation_exceeds_requested_budget"
        elif eos_position is not None and any(t != self.tokenizer.pad_token_id for t in tokens[eos_position + 1:]):
            rejected = "nonpadding_after_eos"
        elif "<think>" in action or CLOSE_THOUGHT in action or self._close_token in content_tokens:
            rejected = "thinking_marker_in_action"
        elif not action or not canonical_ids:
            rejected = "empty_action"
        elif len(canonical_ids) > self.max_policy_tokens:
            rejected = "canonical_action_exceeds_scoring_budget"
        return {"generated_token_ids": list(tokens), "generated_token_ids_sha256": _digest(tokens),
                "content_token_ids": list(content_tokens), "finish_reason": finish,
                "generated_tokens_before_padding": len(tokens) if eos_position is None else eos_position + 1,
                "normalization": normalization, "canonical_action": action,
                "canonical_action_sha256": hashlib.sha256(action.encode("utf-8", errors="surrogatepass")).hexdigest(),
                "canonical_token_ids": canonical_ids, "canonical_token_ids_sha256": _digest(canonical_ids),
                "rejection": rejected, "returned_action": action if rejected is None else ""}

    def _policy_strict_actions(self, session_id: str, request: Any) -> tuple[list[str], list[list[dict[str, Any]]]]:
        if request.max_tokens > self.max_policy_tokens:
            raise ValueError("Qwen answer budget exceeds max_policy_tokens")
        context = self._ensure_context(session_id, request.prompt)
        if sum(len(record.get("policy_generations", [])) for record in self._contexts[session_id].values()) >= MAX_ACTION_TRACE_BATCHES:
            raise ValueError("action audit capacity reached; no implicit trace eviction")
        session = self._activate(session_id)
        encoded = self._tokenize_prompt(request.prompt)
        options = {"do_sample": request.temperature > 0,
            "num_return_sequences": request.n if request.temperature > 0 else 1,
            "max_new_tokens": request.max_tokens, "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id, "return_dict_in_generate": False,
            "output_scores": False, "output_logits": False}
        if request.temperature > 0:
            options.update(temperature=request.temperature, top_p=1.0, top_k=50)
        old_cpu = session.cpu_rng_state.clone()
        old_device = None if session.device_rng_state is None else session.device_rng_state.clone()
        self.model.eval()
        try:
            with self._session_rng(session), self.torch.no_grad():
                started = perf_counter()
                sequences = self.model.generate(**encoded, **options)
                elapsed = perf_counter() - started
                if len(sequences) != options["num_return_sequences"]:
                    raise RuntimeError("model returned an unexpected action batch size")
                candidates = []
                for sequence in sequences:
                    candidate = self._canonical_candidate(sequence[encoded["input_ids"].shape[1]:].tolist(), request.max_tokens)
                    ids = candidate["canonical_token_ids"]
                    if candidate["rejection"] is None and encoded["input_ids"].shape[1] + len(ids) > self.max_sequence_tokens:
                        candidate.update(rejection="canonical_sequence_exceeds_limit", returned_action="")
                    scores = [] if candidate["rejection"] else self._score_generated_tokens(encoded, ids)
                    candidates.append({**candidate, "token_logprobs": scores})
                if request.temperature <= 0 and request.n > 1:
                    candidates = [deepcopy(candidates[0]) for _ in range(request.n)]
                trace = {"schema_version": ACTION_TRACE_SCHEMA,
                    "policy_version": session.optimizer_steps, "prefix_sha256": context["prefix_sha256"],
                    "prompt_sha256": context["prompt_sha256"],
                    "request": {"n": request.n, "temperature": request.temperature, "max_tokens": request.max_tokens},
                    "generation_seconds": elapsed, "candidates": candidates}
                trace["sha256"] = _digest(trace)
            # Context initialization and this complete action batch are separate
            # commits. Infrastructure failures retain the already frozen thought.
            context.setdefault("policy_generations", []).append(trace)
        except BaseException:
            session.cpu_rng_state, session.device_rng_state = old_cpu, old_device
            raise
        return ([item["returned_action"] for item in candidates], [item["token_logprobs"] for item in candidates])

    def _require_recorded_action(self, context: dict[str, Any], tactic: str, version: int,
                                 raw_logprob: float | None = None) -> None:
        if not self._strict_actions():
            return
        for trace in context.get("policy_generations", []):
            if trace["policy_version"] != version:
                continue
            for candidate in trace["candidates"]:
                if candidate["rejection"] is None and candidate["returned_action"] == tactic:
                    probability = math.fsum(row["logprob"] for row in candidate["token_logprobs"])
                    if raw_logprob is None or math.isclose(probability, raw_logprob, rel_tol=1e-8, abs_tol=1e-8):
                        return
        raise ValueError("training action/version/logprob has no recorded canonical policy generation")

    def learn(self, session_id: str, event: dict[str, Any]) -> Any:
        self._activate(session_id)
        if event.get("kind") == "search_visit_backup":
            record = self._contexts[session_id].get(event.get("prompt"))
            if record is None:
                raise ValueError("online learning requires the recorded thought context")
            for candidate in event.get("candidates", []):
                version = candidate.get("behavior_version")
                if type(version) is not int or version < record["source_policy_version"]:
                    raise ValueError("candidate predates its frozen thought context")
                self._require_recorded_action(record, candidate.get("tactic"), version, candidate.get("raw_logprob"))
        else:
            from .success_finalize_objective import OBJECTIVE_KIND, prepare_success_event
            if event.get("kind") == OBJECTIVE_KIND:
                example = prepare_success_event(event, session_id=session_id,
                    policy_version=self.sessions[session_id].optimizer_steps,
                    dataset_root=self.success_dataset_root, gamma=self.gamma, value_floor=self.value_floor)
                for row in example["samples"]:
                    record = self._contexts[session_id].get(row["prompt"])
                    if record is None or row["policy_version"] < record["source_policy_version"]:
                        raise ValueError("successful replay lacks its original frozen thought context")
                    self._require_recorded_action(record, row["tactic"], row["policy_version"])
        result = super().learn(session_id, event)
        return replace(result, detail={**result.detail, "thought_contexts": self.context_receipts(session_id)})

    def export_session(self, session_id: str) -> dict[str, Any]:
        state = super().export_session(session_id)
        return {**state, "qwen_contexts": {"mode": THINKING_MODE,
                "policy_version": self.sessions[session_id].optimizer_steps,
                "entries": deepcopy(self._contexts[session_id])}}

    def initialize_from_experience(self, session_id: str, weights: dict[str, Any]) -> dict[str, Any]:
        if self._contexts.get(session_id):
            raise ValueError("experience destination must have no prior thought contexts")
        return super().initialize_from_experience(session_id, weights)

    def _validate_contexts(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict) or raw.get("mode") != THINKING_MODE:
            raise ValueError("snapshot requires explicit Qwen thought contexts")
        version, entries = raw.get("policy_version"), raw.get("entries")
        if (type(version) is not int or version < 0 or not isinstance(entries, dict)
                or len(entries) > self.max_contexts):
            raise ValueError("invalid snapshot context version/count")
        for prompt, record in entries.items():
            if not isinstance(prompt, str) or not isinstance(record, dict):
                raise ValueError("invalid context record")
            source = record.get("source_policy_version")
            ids, thought, suffix = (record.get(key) for key in ("prefix_ids", "thought_ids", "controller_suffix_ids"))
            if (type(source) is not int or not 0 <= source <= version
                    or any(not isinstance(row, list) or not row or any(type(t) is not int or not 0 <= t < len(self.tokenizer)
                                                                   for t in row) for row in (ids, thought, suffix))
                    or len(ids) > self.max_sequence_tokens or len(thought) > self.thinking_budget):
                raise ValueError("invalid context token/version bounds")
            start = self._chat_ids(prompt)
            natural = thought[-1] == self._close_token
            expected_suffix = self.tokenizer.encode("\n\n" if natural else FORCED_CLOSE, add_special_tokens=False)
            if (record.get("prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest()
                    or record.get("prefix_sha256") != _digest(ids)
                    or record.get("chat_prefix_tokens") != len(start)
                    or type(record.get("natural_close")) is not bool or record["natural_close"] != natural
                    or suffix != expected_suffix or ids != start + thought + suffix
                    or self._close_token in thought[:-1] or any(t in self._eos_ids() for t in thought)
                    or (not natural and len(thought) != self.thinking_budget)):
                raise ValueError("snapshot thought context integrity mismatch")
            if self._strict_actions():
                self._validate_action_traces(record, version)
            elif "policy_generations" in record:
                raise ValueError("raw backend cannot import canonical-action traces")
        if sum(len(record.get("policy_generations", [])) for record in entries.values()) > MAX_ACTION_TRACE_BATCHES:
            raise ValueError("snapshot exceeds action audit capacity")
        return deepcopy(entries)

    def _validate_action_traces(self, context: dict[str, Any], version: int) -> None:
        traces = context.get("policy_generations", [])
        if not isinstance(traces, list):
            raise ValueError("invalid action audit list")
        for trace in traces:
            if not isinstance(trace, dict) or trace.get("sha256") != _digest({key: value for key, value in trace.items() if key != "sha256"}):
                raise ValueError("action trace hash mismatch")
            request = trace.get("request", {})
            if (trace.get("schema_version") != ACTION_TRACE_SCHEMA or type(trace.get("policy_version")) is not int
                    or not context["source_policy_version"] <= trace["policy_version"] <= version
                    or trace.get("prefix_sha256") != context["prefix_sha256"]
                    or trace.get("prompt_sha256") != context["prompt_sha256"]
                    or type(request.get("max_tokens")) is not int or not 1 <= request["max_tokens"] <= self.max_policy_tokens
                    or type(request.get("n")) is not int or not 1 <= request["n"] <= 64
                    or not isinstance(trace.get("candidates"), list) or len(trace["candidates"]) != request["n"]):
                raise ValueError("action trace context/version/request mismatch")
            for candidate in trace["candidates"]:
                tokens = candidate.get("generated_token_ids")
                if not isinstance(tokens, list) or any(type(t) is not int or not 0 <= t < len(self.tokenizer) for t in tokens):
                    raise ValueError("invalid raw action token IDs")
                expected = self._canonical_candidate(tokens, request["max_tokens"])
                if expected["rejection"] is None and len(context["prefix_ids"]) + len(expected["canonical_token_ids"]) > self.max_sequence_tokens:
                    expected.update(rejection="canonical_sequence_exceeds_limit", returned_action="")
                if {k: v for k, v in candidate.items() if k != "token_logprobs"} != expected:
                    raise ValueError("raw/canonical action binding mismatch")
                scores = candidate.get("token_logprobs")
                required_length = 0 if candidate["rejection"] else len(candidate["canonical_token_ids"])
                if (not isinstance(scores, list) or len(scores) != required_length
                        or any(type(row.get("logprob")) not in (int, float) or not math.isfinite(row["logprob"])
                               or row["logprob"] > 0 for row in scores)):
                    raise ValueError("canonical action logprob shape/value mismatch")

    def import_session(self, session_id: str, state: dict[str, Any]) -> None:
        entries = self._validate_contexts(state.get("qwen_contexts"))
        super().import_session(session_id, state)
        self._contexts[session_id] = entries

    def _validate_import_payload(self, state: dict[str, Any], payload: dict[str, Any]) -> None:
        if payload.get("optimizer_steps") != state["qwen_contexts"]["policy_version"]:
            raise ValueError("snapshot optimizer/context version mismatch")
