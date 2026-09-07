"""Lazy PyTorch/Transformers backend for REAL-Prover test-time training.

Heavy GPU libraries are imported only when ``RealProverBackend`` is created,
so the control plane and its unit tests stay runnable on Windows/CPU hosts.
All public methods are expected to run through :class:`GpuActor`.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import io
import math
import json
from typing import Any

from .backend import BackendLearnResult
from .errors import DuplicateSessionError, SessionNotFoundError
from .schemas import ChatRequest


TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


@dataclass(frozen=True)
class PolicyScoringConfig:
    """Opt-in inference implementation; does not change the raw log-P target.

    Hard limits bound concurrent KV rows and retained vocabulary logits. The
    4096-token limit counts physical prompt padding plus generation budget.
    Larger requests remain supported by the unchanged default tokenwise path.
    """
    mode: str = "tokenwise"
    candidate_batch_size: int = 2
    token_chunk_size: int = 8

    def __post_init__(self) -> None:
        if self.mode not in {"tokenwise", "candidate_chunks", "tokenwise_deferred"}:
            raise ValueError("unknown policy scoring mode")
        for name, maximum in (("candidate_batch_size", 4), ("token_chunk_size", 32)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [1, {maximum}]")


MAX_SCORING_CANDIDATES = 64
MAX_SCORING_SEQUENCE_TOKENS = 4096
MAX_DEFERRED_SCALARS = 4096


@dataclass
class _RealSession:
    value_head: Any
    optimizer: Any
    optimizer_steps: int = 0
    examples_seen: int = 0
    rng_seed: int = 0
    cpu_rng_state: Any = None
    device_rng_state: Any = None


class RealProverBackend:
    """One frozen 7B base with isolated named LoRA/value/optimizer states."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda:0",
        expected_hidden_size: int = 3584,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        learning_rate: float = 1e-4,
        value_learning_rate: float = 3e-4,
        kl_beta: float = 0.02,
        value_coefficient: float = 0.5,
        max_grad_norm: float = 1.0,
        policy_scoring: PolicyScoringConfig | None = None,
    ) -> None:
        if policy_scoring is not None and not isinstance(policy_scoring, PolicyScoringConfig):
            raise ValueError("policy_scoring must be an explicit PolicyScoringConfig")
        self.policy_scoring = policy_scoring or PolicyScoringConfig()
        try:
            import torch
            import torch.nn.functional as functional
            from peft import LoraConfig, get_peft_model
            from peft.utils import set_peft_model_state_dict
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised on the GPU image
            raise RuntimeError(
                "RealProverBackend requires torch, transformers, and peft from the GPU image"
            ) from exc

        self.torch = torch
        self.functional = functional
        self.set_peft_model_state_dict = set_peft_model_state_dict
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("ROCm/CUDA device requested but torch.cuda.is_available() is false")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        base = self._load_base_model(model_path)
        hidden_size = self._base_hidden_size(base)
        if hidden_size != expected_hidden_size:
            raise RuntimeError(
                f"REAL-Prover hidden_size mismatch: expected {expected_hidden_size}, got {hidden_size}"
            )
        base.config.use_cache = True
        self.hidden_size = hidden_size
        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            target_modules=self._lora_target_modules(base),
            bias="none",
            task_type="CAUSAL_LM",
            # PEFT's default initialization makes B zero and preserves the
            # frozen base policy. ``False`` would initialize random A and B.
            init_lora_weights=True,
        )
        self.model = get_peft_model(base, self.lora_config, adapter_name="__bootstrap__")
        self.model.eval()
        self.sessions: dict[str, _RealSession] = {}
        self.learning_rate = float(learning_rate)
        self.value_learning_rate = float(value_learning_rate)
        self.kl_beta = float(kl_beta)
        self.value_coefficient = float(value_coefficient)
        self.max_grad_norm = float(max_grad_norm)

    def _load_base_model(self, model_path: str) -> Any:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, torch_dtype=self.torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(self.device)

    def _base_hidden_size(self, base: Any) -> int:
        return int(base.config.hidden_size)

    def _lora_target_modules(self, base: Any) -> list[str]:
        return list(TARGET_MODULES)

    def _tokenize_prompt(self, prompt: str) -> dict[str, Any]:
        """Shared inference/training prompt path; legacy bytes remain unchanged."""
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def _session(self, session_id: str) -> _RealSession:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise SessionNotFoundError(f"backend session not found: {session_id}") from exc

    def _activate(self, session_id: str) -> _RealSession:
        session = self._session(session_id)
        self.model.set_adapter(session_id)
        return session

    def _adapter_parameters(self, session_id: str) -> list[Any]:
        parameters = [parameter for name, parameter in self.model.named_parameters()
                      if self._is_adapter_tensor(name, session_id)]
        if not parameters:
            raise RuntimeError(f"no LoRA parameters found for adapter {session_id}")
        for parameter in parameters:
            parameter.requires_grad_(True)
        return parameters

    @staticmethod
    def _is_adapter_tensor(name: str, session_id: str) -> bool:
        parts = name.split(".")
        return len(parts) >= 3 and parts[-3] in {"lora_A", "lora_B"} and parts[-2] == session_id

    def _adapter_state_dict(self, session_id: str) -> dict[str, Any]:
        """Exact namespace export for our bias-free Linear LoRA configuration.

        PEFT's generic getter filters with substring membership and removes all
        occurrences of the adapter name. Prefix-related IDs can therefore leak
        foreign tensors; IDs such as 'model' can damage base module names too.
        Strip only the selected adapter's structural path component. Values are
        references, as in torch.state_dict; callers serialize before mutation.
        """
        result = {}
        for name, tensor in self.model.state_dict().items():
            if self._is_adapter_tensor(name, session_id):
                parts = name.split(".")
                canonical = ".".join(parts[:-2] + parts[-1:])
                if canonical in result:
                    raise RuntimeError("duplicate canonical LoRA tensor")
                result[canonical] = tensor
        if not result:
            raise RuntimeError(f"no exact LoRA state for adapter {session_id}")
        return result

    def _new_rng(self, session_id: str) -> tuple[int, Any, Any]:
        # A session's initialization/sampling must not depend on request order
        # or consume another session's (or the host's) random stream.
        seed = int.from_bytes(hashlib.sha256(session_id.encode("utf-8")).digest()[:8], "big") % (2**63)
        cpu_state = self.torch.Generator(device="cpu").manual_seed(seed).get_state()
        device_state = None
        if self.device.type == "cuda":
            device_state = self.torch.Generator(device=self.device).manual_seed(seed).get_state()
        return seed, cpu_state, device_state

    @contextmanager
    def _session_rng(self, session: _RealSession):
        """Commit the private stream on success; failed calls consume no RNG."""
        torch = self.torch
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        if session.cpu_rng_state is None or (devices and session.device_rng_state is None):
            raise RuntimeError("session RNG state is missing")
        # GpuActor serializes this process-wide swap. fork_rng restores the
        # outside CPU/selected-device streams even when generation raises.
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(session.cpu_rng_state)
            if devices:
                torch.cuda.set_rng_state(session.device_rng_state, device=self.device)
            yield
            cpu_state = torch.get_rng_state().clone()
            device_state = torch.cuda.get_rng_state(self.device).clone() if devices else None
            session.cpu_rng_state = cpu_state
            session.device_rng_state = device_state

    def create_session(self, session_id: str) -> dict[str, dict[str, Any]]:
        # PEFT stores adapter IDs in ModuleDict, whose keys cannot contain dots.
        # Reserve the internal adapter too; reject before touching model state.
        if "." in session_id or session_id == "__bootstrap__":
            raise ValueError("real adapter session IDs cannot contain dots or use __bootstrap__")
        if session_id in self.sessions:
            raise DuplicateSessionError(f"backend session already exists: {session_id}")
        torch = self.torch
        seed, cpu_state, device_state = self._new_rng(session_id)
        session = _RealSession(None, None, rng_seed=seed, cpu_rng_state=cpu_state, device_rng_state=device_state)
        added_adapter = False
        try:
            with self._session_rng(session):
                self.model.add_adapter(session_id, self.lora_config)
                added_adapter = True
                self.model.set_adapter(session_id)
                session.value_head = torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_size, 256),
                    torch.nn.SiLU(),
                    torch.nn.Linear(256, 1),
                    torch.nn.Tanh(),
                ).to(device=self.device, dtype=torch.float32)
                adapter_parameters = self._adapter_parameters(session_id)
                session.optimizer = torch.optim.AdamW(
                    [
                        {"params": adapter_parameters, "lr": self.learning_rate},
                        {"params": list(session.value_head.parameters()), "lr": self.value_learning_rate},
                    ]
                )
        except BaseException:
            if added_adapter:
                self.model.delete_adapter(session_id)
            raise
        self.sessions[session_id] = session
        return {
            "adapter": {"kind": "lora", "rank": self.lora_config.r, "initial_equivalent_to_base": True},
            "value": {"hidden_size": self.hidden_size, "head": "3584-256-1-tanh"},
            "optimizer": {"kind": "AdamW", "steps": 0},
            "reference": {"kind": "frozen_base", "adapter_disabled": True},
            "rng": {"seed": seed, "scope": "session", "device_type": self.device.type},
        }

    def delete_session(self, session_id: str) -> None:
        self._session(session_id)
        del self.sessions[session_id]
        self.model.delete_adapter(session_id)

    def _encoded_prompt(self, request: ChatRequest) -> dict[str, Any]:
        return self._tokenize_prompt(request.prompt)

    def _score_generated_tokens(self, encoded: dict[str, Any], token_ids: list[int]) -> list[dict[str, Any]]:
        """Raw log P(token|prefix), one vocabulary vector and one row KV cache.

        Re-score rather than retain generate's processed scores (temperature,
        top-k, etc.) or an entire generated_length × vocabulary tensor. This
        costs extra forward passes but bounds score storage independently of
        completion length. Qwen2's logits_to_keep=1 also bounds prefill logits.
        """
        if not token_ids:
            return []
        torch = self.torch
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).clone()
        cache = None
        row: list[dict[str, Any]] = []
        for index, token_id in enumerate(token_ids):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            output = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids[:, -input_ids.shape[1]:],
                past_key_values=cache, use_cache=True, logits_to_keep=1,
                return_dict=True,
            )
            logits = output.logits[:, -1, :].float()
            self._require_finite(logits, "raw policy logits")
            score = torch.log_softmax(logits, dim=-1)[0, token_id]
            self._require_finite(score, "raw policy token logprob")
            row.append({
                "token": self.tokenizer.decode([token_id], skip_special_tokens=False),
                "logprob": float(score.item()),
            })
            if index + 1 < len(token_ids):
                cache = output.past_key_values
                if cache is None:
                    raise RuntimeError("raw policy scoring requires a KV cache")
                input_ids = torch.tensor([[token_id]], device=self.device, dtype=encoded["input_ids"].dtype)
                attention_mask = torch.cat((attention_mask, torch.ones_like(input_ids)), dim=1)
            del output, logits, score
        return row

    def _scoring_prompt(self, encoded: dict[str, Any], generated_tokens: int) -> tuple[Any, Any]:
        """Validate bounded left-padded single-prompt input before GPU work."""
        torch = self.torch
        ids = encoded.get("input_ids")
        if (not torch.is_tensor(ids) or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1
                or ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError("chunked policy scoring requires one nonempty integer-token prompt")
        if type(generated_tokens) is not int or generated_tokens < 0 or ids.shape[1] + generated_tokens > MAX_SCORING_SEQUENCE_TOKENS:
            raise ValueError("chunked policy scoring exceeds 4096 prompt-plus-generation tokens")
        mask = encoded.get("attention_mask", torch.ones_like(ids))
        if (not torch.is_tensor(mask) or mask.shape != ids.shape or mask.device != ids.device
                or not bool(((mask == 0) | (mask == 1)).all()) or not bool(mask[0, -1] == 1)
                or not bool((mask[:, 1:].long() >= mask[:, :-1].long()).all())):
            raise ValueError("chunked policy scoring requires a binary left-padding attention mask")
        return ids, mask

    def _score_generated_candidates_deferred(self, encoded: dict[str, Any], candidates: list[list[int]]) -> list[list[dict[str, Any]]]:
        """Legacy forward shapes/order, with score materialization deferred.

        No vocabulary vector survives its token iteration: only independent
        scalar clones (at most 4096 across the request) and one finite flag.
        Checks inside the loop stay on device. The final flag+scores transfer
        is the single host synchronization used to materialize scoring results.
        Input validation happens before the loop; no partial rows can escape.
        """
        if (not isinstance(candidates, list) or len(candidates) > MAX_SCORING_CANDIDATES
                or any(not isinstance(row, list) or any(type(token) is not int or token < 0 for token in row)
                       for row in candidates)):
            raise ValueError("deferred scoring requires at most 64 integer-token candidates")
        count = sum(map(len, candidates))
        if count > MAX_DEFERRED_SCALARS:
            raise ValueError("deferred scoring exceeds 4096 retained scalars")
        self._scoring_prompt(encoded, max(map(len, candidates), default=0))
        if not count:
            return [[] for _ in candidates]
        torch = self.torch
        finite = torch.ones((), dtype=torch.bool, device=self.device)
        scores = []
        for token_ids in candidates:
            if not token_ids:
                continue
            # One host-to-device token construction per candidate. Subsequent
            # singleton views retain legacy (1, 1) forward inputs and order.
            candidate_ids = torch.tensor([token_ids], device=self.device, dtype=encoded["input_ids"].dtype)
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).clone()
            cache = None
            for index, token_id in enumerate(token_ids):
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                output = self.model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids[:, -input_ids.shape[1]:],
                    past_key_values=cache, use_cache=True, logits_to_keep=1,
                    return_dict=True,
                )
                logits = output.logits[:, -1, :].float()
                finite.logical_and_(torch.isfinite(logits).all())
                score = torch.log_softmax(logits, dim=-1)[0, token_id]
                finite.logical_and_(torch.isfinite(score).all())
                # Indexing is a view into a vocabulary vector. clone is needed
                # before deleting score so the list retains only scalar bytes.
                scores.append(score.detach().clone())
                if index + 1 < len(token_ids):
                    cache = output.past_key_values
                    if cache is None:
                        raise RuntimeError("raw policy scoring requires a KV cache")
                    input_ids = candidate_ids.narrow(1, index, 1)
                    attention_mask = torch.cat((attention_mask, torch.ones_like(input_ids)), dim=1)
                del output, logits, score
            del cache, candidate_ids
        materialized = torch.cat((finite.float().reshape(1), torch.stack(scores))).cpu().tolist()
        if materialized[0] != 1.0 or not all(math.isfinite(value) for value in materialized[1:]):
            raise FloatingPointError("non-finite deferred raw policy logits or token logprobs")
        rows, cursor = [], 1
        for token_ids in candidates:
            rows.append([{"token": self.tokenizer.decode([token], skip_special_tokens=False),
                          "logprob": materialized[cursor + offset]}
                         for offset, token in enumerate(token_ids)])
            cursor += len(token_ids)
        return rows

    def _score_generated_candidates(self, encoded: dict[str, Any], candidates: list[list[int]],
                                    config: PolicyScoringConfig) -> list[list[dict[str, Any]]]:
        """Raw causal log-P in bounded candidate batches and token chunks.

        Candidates come from this same policy call, already trimmed at their
        first EOS. First prefill contains prompt + target[:chunk-1]; subsequent
        chunks feed target[start-1:end-1] with the previous KV cache. Therefore
        retained logits predict exactly target[start:end], with no extra EOS.
        Shorter rows pad masked inputs; their extra logits are never scored.
        No generated scores, temperature transform, RNG calls or weight writes.
        """
        if not isinstance(config, PolicyScoringConfig) or config.mode != "candidate_chunks":
            raise ValueError("candidate scoring requires explicit candidate_chunks config")
        if (not isinstance(candidates, list) or len(candidates) > MAX_SCORING_CANDIDATES
                or any(not isinstance(row, list) or any(type(token) is not int or token < 0 for token in row)
                       for row in candidates)):
            raise ValueError("chunked scoring requires at most 64 integer-token candidates")
        longest = max(map(len, candidates), default=0)
        prompt, prompt_mask = self._scoring_prompt(encoded, longest)
        torch = self.torch
        rows: list[list[dict[str, Any]]] = [[] for _ in candidates]
        nonempty = [(index, tokens) for index, tokens in enumerate(candidates) if tokens]
        for batch_start in range(0, len(nonempty), config.candidate_batch_size):
            batch = nonempty[batch_start:batch_start + config.candidate_batch_size]
            count, cache = len(batch), None
            maximum = max(len(tokens) for _, tokens in batch)
            attention_mask = prompt_mask.expand(count, -1).clone()
            for start in range(0, maximum, config.token_chunk_size):
                width = min(config.token_chunk_size, maximum - start)
                input_start = max(0, start - 1)
                input_width = width - 1 if start == 0 else width
                tail = torch.full((count, input_width), self.tokenizer.pad_token_id,
                                  device=prompt.device, dtype=prompt.dtype)
                tail_mask = torch.zeros((count, input_width), device=prompt.device, dtype=attention_mask.dtype)
                for index, (_, tokens) in enumerate(batch):
                    # Last target token need not enter KV: there is no next
                    # valid target in this row, even if other rows continue.
                    actual = tokens[input_start:min(input_start + input_width, len(tokens) - 1)]
                    if actual:
                        tail[index, :len(actual)] = torch.tensor(actual, device=prompt.device, dtype=prompt.dtype)
                        tail_mask[index, :len(actual)] = 1
                input_ids = torch.cat((prompt.expand(count, -1), tail), dim=1) if start == 0 else tail
                attention_mask = torch.cat((attention_mask, tail_mask), dim=1)
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids[:, -input_ids.shape[1]:], past_key_values=cache,
                    use_cache=True, logits_to_keep=width, return_dict=True)
                logits = output.logits
                if logits.ndim != 3 or logits.shape[:2] != (count, width):
                    raise RuntimeError("chunked policy scoring returned unexpected retained logit shape")
                for index, (destination, tokens) in enumerate(batch):
                    target = tokens[start:start + width]
                    if not target:
                        continue
                    raw = logits[index, :len(target), :].float()
                    self._require_finite(raw, "raw policy logits")
                    target_ids = torch.tensor(target, device=raw.device, dtype=torch.long)
                    scores = torch.log_softmax(raw, dim=-1).gather(-1, target_ids[:, None]).squeeze(-1)
                    self._require_finite(scores, "raw policy token logprob")
                    rows[destination].extend({"token": self.tokenizer.decode([token], skip_special_tokens=False),
                                              "logprob": float(score)} for token, score in zip(target, scores.tolist()))
                    del raw, scores
                if start + width < maximum:
                    cache = output.past_key_values
                    if cache is None:
                        raise RuntimeError("chunked policy scoring requires a KV cache")
                del output, logits
            del cache
        return rows

    def policy(self, session_id: str, request: ChatRequest) -> tuple[list[str], list[list[dict[str, Any]]]]:
        session = self._activate(session_id)
        torch = self.torch
        self.model.eval()
        encoded = self._encoded_prompt(request)
        scoring = getattr(self, "policy_scoring", PolicyScoringConfig())
        if not isinstance(scoring, PolicyScoringConfig):
            raise ValueError("policy_scoring must be an explicit PolicyScoringConfig")
        if scoring.mode != "tokenwise":
            if not 1 <= request.n <= MAX_SCORING_CANDIDATES:
                raise ValueError("opt-in policy scoring requires at most 64 candidates")
            self._scoring_prompt(encoded, request.max_tokens)
        do_sample = request.temperature > 0
        if scoring.mode == "tokenwise_deferred" and (request.n if do_sample else 1) * request.max_tokens > MAX_DEFERRED_SCALARS:
            raise ValueError("deferred generation budget exceeds 4096 retained scalars")
        generation_options: dict[str, Any] = {
            "do_sample": do_sample,
            "num_return_sequences": request.n if do_sample else 1,
            "max_new_tokens": request.max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": False,
            "output_scores": False,
            "output_logits": False,
        }
        if do_sample:
            generation_options["temperature"] = max(request.temperature, 1e-5)
        contents: list[str] = []
        logprobs: list[list[dict[str, Any]]] = []
        candidate_tokens: list[list[int]] = []
        with self._session_rng(session), torch.no_grad():
            sequences = self.model.generate(**encoded, **generation_options)
            prompt_length = encoded["input_ids"].shape[1]
            eos = self.tokenizer.eos_token_id
            eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
            for sequence in sequences:
                token_ids = sequence[prompt_length:].tolist()
                for index, token_id in enumerate(token_ids):
                    if token_id in eos_ids:
                        # Count the first EOS decision, never subsequent batch
                        # padding (including when pad_token_id == eos_token_id).
                        if any(item != self.tokenizer.pad_token_id for item in token_ids[index + 1:]):
                            raise ValueError("non-padding tokens after generated EOS")
                        token_ids = token_ids[:index + 1]
                        break
                contents.append(self.tokenizer.decode(token_ids, skip_special_tokens=True).strip())
                if scoring.mode == "tokenwise":
                    logprobs.append(self._score_generated_tokens(encoded, token_ids))
                else:
                    candidate_tokens.append(token_ids)
            if scoring.mode == "candidate_chunks":
                logprobs = self._score_generated_candidates(encoded, candidate_tokens, scoring)
            elif scoring.mode == "tokenwise_deferred":
                logprobs = self._score_generated_candidates_deferred(encoded, candidate_tokens)
        if not do_sample and request.n > 1:
            contents *= request.n
            logprobs *= request.n
        return contents, logprobs

    def value(self, session_id: str, request: ChatRequest) -> float:
        session = self._activate(session_id)
        torch = self.torch
        self.model.eval()
        encoded = self._encoded_prompt(request)
        with torch.no_grad():
            output = self.model(**encoded, output_hidden_states=True, return_dict=True)
            hidden = output.hidden_states[-1][:, -1, :].float()
            score = session.value_head(hidden).squeeze().item()
        return float(score)

    def initial_equivalence_error(self, session_id: str, prompt: str) -> float:
        """Maximum logit delta between a fresh adapter and the frozen base."""
        self._activate(session_id)
        torch = self.torch
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            adapted = self.model(**encoded, return_dict=True).logits.float()
            with self.model.disable_adapter():
                reference = self.model(**encoded, return_dict=True).logits.float()
        return float((adapted - reference).abs().max().cpu())

    def _require_finite(self, value: Any, label: str) -> None:
        """Check session tensors without copying the shared base model."""
        if self.torch.is_tensor(value):
            if not bool(self.torch.isfinite(value).all().item()):
                raise FloatingPointError(f"non-finite {label}")
        elif isinstance(value, dict):
            for key, item in value.items():
                self._require_finite(item, f"{label}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                self._require_finite(item, f"{label}[{index}]")
        elif isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError(f"non-finite {label}")

    def learn(self, session_id: str, event: dict[str, Any]) -> BackendLearnResult:
        session = self._activate(session_id)
        torch = self.torch
        reward = float(event.get("reward", 0.0))
        if not math.isfinite(reward) or not -1.0 <= reward <= 1.0:
            raise ValueError("reward must be finite and in [-1, 1]")
        if reward > 0 and not bool(event.get("terminal_verified", event.get("root_verified", False))):
            raise ValueError("positive reward requires terminal_verified=true")
        prompt = event.get("prompt")
        target = str(event.get("tactic") or "")
        if not isinstance(prompt, str) or not prompt.strip() or not target:
            raise ValueError("training event requires the original inference prompt and a non-empty tactic")
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        target_ids = self.tokenizer(target, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        if not prompt_ids.numel() or not target_ids.numel():
            raise ValueError("training prompt and tactic must each encode at least one token")
        input_ids = torch.cat((prompt_ids, target_ids), dim=1)
        labels = torch.full_like(input_ids, -100)
        labels[:, prompt_ids.shape[1]:] = target_ids

        session.optimizer.zero_grad(set_to_none=True)
        previous_use_cache = self.model.config.use_cache
        try:
            self.model.config.use_cache = False
            self.model.train()
            session.value_head.train()
            output = self.model(
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
                return_dict=True,
            )
            token_nll = output.loss.float()
            policy_loss = reward * token_nll
            current_logits = output.logits[:, prompt_ids.shape[1] - 1:-1, :].float()
            with torch.no_grad(), self.model.disable_adapter():
                reference = self.model(input_ids=input_ids, return_dict=True)
                reference_logits = reference.logits[:, prompt_ids.shape[1] - 1:-1, :].float()
            current_log_probs = self.functional.log_softmax(current_logits, dim=-1)
            reference_log_probs = self.functional.log_softmax(reference_logits, dim=-1)
            kl = self.functional.kl_div(
                reference_log_probs,
                current_log_probs.exp(),
                reduction="batchmean",
            ) / max(1, target_ids.numel())
            value_hidden = output.hidden_states[-1][:, prompt_ids.shape[1] - 1, :].float()
            value_prediction = session.value_head(value_hidden).squeeze()
            value_target = torch.tensor(reward, device=self.device, dtype=torch.float32)
            value_loss = self.functional.mse_loss(value_prediction, value_target)
            loss = policy_loss + self.kl_beta * kl + self.value_coefficient * value_loss
            self._require_finite({
                "total": loss, "policy": policy_loss, "value": value_loss, "kl": kl,
            }, "loss")
            loss.backward()
            parameters = self._adapter_parameters(session_id) + list(session.value_head.parameters())
            self._require_finite([parameter.grad for parameter in parameters if parameter.grad is not None], "gradient")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, self.max_grad_norm, error_if_nonfinite=True,
            )
            session.optimizer.step()
            # An optimizer can fail after modifying only some tensors. Runtime
            # owns the pre-step snapshot and rolls back any exception here.
            self._require_finite(parameters, "parameter")
            self._require_finite(session.optimizer.state_dict(), "optimizer")
        finally:
            self.model.config.use_cache = previous_use_cache
            self.model.eval()
            session.value_head.eval()
            session.optimizer.zero_grad(set_to_none=True)
        session.optimizer_steps += 1
        session.examples_seen += 1
        detail = {
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "kl": float(kl.detach().cpu()),
            "reward": reward,
            "grad_norm": float(grad_norm.detach().cpu()),
            "finite_loss": True,
            "finite_gradients": True,
            "finite_parameters": True,
            "finite_optimizer_state": True,
            "optimizer_steps": session.optimizer_steps,
        }
        return BackendLearnResult(
            adapter_metadata={"kind": "lora", "rank": self.lora_config.r, "examples_seen": session.examples_seen},
            value_metadata={"hidden_size": self.hidden_size, "last_loss": detail["value_loss"]},
            optimizer_metadata={"kind": "AdamW", "steps": session.optimizer_steps},
            detail=detail,
        )

    def export_session(self, session_id: str) -> dict[str, Any]:
        session = self._activate(session_id)
        payload = {
            "adapter": self._adapter_state_dict(session_id),
            "value_head": session.value_head.state_dict(),
            "optimizer": session.optimizer.state_dict(),
            "optimizer_steps": session.optimizer_steps,
            "examples_seen": session.examples_seen,
            "rng": {
                "seed": session.rng_seed, "device_type": self.device.type,
                "cpu": session.cpu_rng_state, "device": session.device_rng_state,
            },
        }
        buffer = io.BytesIO()
        self.torch.save(payload, buffer)
        return {
            "schema_version": "reap.gpu.real-prover-backend.v1",
            "encoding": "torch-save-base64",
            "payload": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }

    def import_session(self, session_id: str, state: dict[str, Any]) -> None:
        session = self._activate(session_id)
        if state.get("schema_version") != "reap.gpu.real-prover-backend.v1":
            raise ValueError("unsupported REAL-Prover backend snapshot schema")
        raw = base64.b64decode(str(state.get("payload", "")), validate=True)
        payload = self.torch.load(io.BytesIO(raw), map_location=self.device, weights_only=False)
        self._validate_import_payload(state, payload)
        rng = payload.get("rng")
        if not isinstance(rng, dict) or rng.get("device_type") != self.device.type:
            raise ValueError("snapshot requires matching per-session RNG state")
        cpu_state = rng["cpu"].cpu()
        # Validate RNG bytes using private generators before mutating any state.
        self.torch.Generator(device="cpu").set_state(cpu_state)
        device_state = rng.get("device")
        if self.device.type == "cuda":
            if device_state is None:
                raise ValueError("snapshot is missing device RNG state")
            device_state = device_state.cpu()
            self.torch.Generator(device=self.device).set_state(device_state)
        self._require_finite(payload, "snapshot")
        self.set_peft_model_state_dict(self.model, payload["adapter"], adapter_name=session_id)
        session.value_head.load_state_dict(payload["value_head"])
        session.optimizer.load_state_dict(payload["optimizer"])
        session.optimizer_steps = int(payload["optimizer_steps"])
        session.examples_seen = int(payload["examples_seen"])
        session.rng_seed = int(rng["seed"])
        session.cpu_rng_state = cpu_state.clone()
        session.device_rng_state = None if device_state is None else device_state.clone()

    def _validate_import_payload(self, state: dict[str, Any], payload: dict[str, Any]) -> None:
        """Subclass metadata checks run before restoring any mutable tensors."""

    def experience_contract(self) -> dict[str, Any]:
        """Hash actual loaded frozen tensors once, only on explicit opt-in.

        No paths, claimed model names, or mutable adapter names define identity.
        Transfers are chunked to avoid a second full CPU copy of the 7B model.
        """
        if not hasattr(self, "_experience_base_digest"):
            digest = hashlib.sha256()
            for name, tensor in sorted(list(self.model.named_parameters()) + list(self.model.named_buffers())):
                if "lora_" in name:
                    continue
                if tensor.requires_grad:
                    raise ValueError("experience identity requires a frozen base")
                digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode())
                raw = tensor.detach().contiguous().reshape(-1).view(self.torch.uint8)
                for start in range(0, raw.numel(), 1024 * 1024):
                    digest.update(raw[start:start + 1024 * 1024].cpu().numpy().tobytes())
            config = self.model.config.to_dict()
            config.pop("_name_or_path", None)
            digest.update(json.dumps(config, sort_keys=True, default=str).encode())
            digest.update(self.tokenizer.backend_tokenizer.to_str().encode())
            digest.update(json.dumps({key: getattr(self.tokenizer, key, None) for key in
                ("eos_token_id", "pad_token_id", "padding_side", "special_tokens_map")},
                sort_keys=True, default=str).encode())
            self._experience_base_digest = digest.hexdigest()
        return {"backend": "real", "base_sha256": self._experience_base_digest,
                "objective": "legacy-terminal-v1", "hidden_size": self.hidden_size,
                "lora_rank": self.lora_config.r, "lora_alpha": self.lora_config.lora_alpha,
                "lora_dropout": self.lora_config.lora_dropout, "target_modules": sorted(TARGET_MODULES),
                "value_head": "linear-silu-linear-tanh-v1"}

    def _check_experience_snapshot(self, session_id: str, snapshot: dict[str, Any]) -> None:
        if snapshot.get("schema_version") != "reap.gpu.real-prover-backend.v1":
            raise ValueError("unsupported experience source backend")
        if snapshot.get("experience_contract") != self.experience_contract():
            raise ValueError("experience source base/model/objective mismatch")

    def experience_weights(self, session_id: str, snapshot: dict[str, Any]) -> dict[str, Any]:
        self._check_experience_snapshot(session_id, snapshot)
        raw = base64.b64decode(snapshot["payload"], validate=True)
        payload = self.torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        weights = {key: payload[key] for key in ("adapter", "value_head")}
        self._require_finite(weights, "experience source")
        buffer = io.BytesIO()
        self.torch.save(weights, buffer)
        return {"contract": self.experience_contract(), "encoding": "torch-save-base64",
                "payload": base64.b64encode(buffer.getvalue()).decode("ascii")}

    def initialize_from_experience(self, session_id: str, weights: dict[str, Any]) -> dict[str, Any]:
        if (set(weights) != {"contract", "encoding", "payload"}
                or weights["contract"] != self.experience_contract()
                or weights["encoding"] != "torch-save-base64"):
            raise ValueError("experience base/model/objective/encoding mismatch")
        session = self._activate(session_id)
        if session.optimizer_steps or session.examples_seen or session.optimizer.state:
            raise ValueError("experience initialization requires a fresh optimizer and counters")
        raw = base64.b64decode(weights["payload"], validate=True)
        payload = self.torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        if set(payload) != {"adapter", "value_head"}:
            raise ValueError("experience may contain only adapter/value parameters")
        expected = {"adapter": self._adapter_state_dict(session_id),
                    "value_head": session.value_head.state_dict()}
        for group, tensors in expected.items():
            if not isinstance(payload[group], dict) or set(payload[group]) != set(tensors):
                raise ValueError(f"experience {group} tensor names mismatch")
            for name, target in tensors.items():
                tensor = payload[group][name]
                if (not self.torch.is_tensor(tensor) or tensor.shape != target.shape or tensor.dtype != target.dtype):
                    raise ValueError(f"experience {group} shape/dtype mismatch")
        self._require_finite(payload, "experience")
        self.set_peft_model_state_dict(self.model, payload["adapter"], adapter_name=session_id)
        session.value_head.load_state_dict(payload["value_head"], strict=True)
        # PEFT may report unrelated missing base keys; verify copied values directly.
        copied = {"adapter": self._adapter_state_dict(session_id),
                  "value_head": session.value_head.state_dict()}
        for group in copied:
            for name, tensor in copied[group].items():
                if not self.torch.equal(tensor.detach().cpu(), payload[group][name]):
                    raise ValueError("experience parameter copy verification failed")
        return {"adapter": {"kind": "lora", "rank": self.lora_config.r,
                            "initial_equivalence_checked": False, "initialized_from_experience": True},
                "value": {"hidden_size": self.hidden_size, "initialized_from_experience": True}}
