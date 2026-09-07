#!/usr/bin/env python3
"""HTTP control/data plane for the session-isolated GPU runtime."""

from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

from .errors import (
    DuplicateSessionError,
    EventConflictError,
    GpuRuntimeError,
    InvalidIdentifierError,
    SessionNotFoundError,
    VersionConflictError,
)
from .real_backend import PolicyScoringConfig, RealProverBackend
from .search_backend import RealSearchBackend
from .runtime import GpuRuntime
from .snapshot_store import _json_bytes
from .toy_backend import ToyBackend


_SESSION_PATH = re.compile(r"^/sessions/([^/]+)(?:/(.*))?$")


def _read_initialization_contract(runtime: GpuRuntime) -> tuple[dict[str, Any], str]:
    """Read the actual immutable contract without allocating a session.

    The backend's initialization contract is fixed for this runtime; existing
    runtime APIs do not mutate it. Read it on the actor, where backend access
    belongs, and freeze the exact JSON bytes used for the receipt/hash.
    """
    def read_contract() -> tuple[dict[str, Any], str]:
        accessor = getattr(runtime.backend, "experience_contract", None)
        if not callable(accessor):
            raise ValueError("backend has no initialization contract")
        contract = accessor()
        if not isinstance(contract, dict) or not contract:
            raise ValueError("backend initialization contract must be a nonempty JSON object")
        try:
            raw = _json_bytes(contract)
        except (TypeError, ValueError) as exc:
            raise ValueError("backend initialization contract must be finite JSON") from exc
        actual = hashlib.sha256(raw).hexdigest()
        return json.loads(raw), actual

    return runtime.actor.submit(read_contract)


def _initialization_contract(runtime: GpuRuntime, expected_sha256: Any) -> tuple[dict[str, Any], str]:
    """Validate an explicit client pin before allocating a session."""
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("expected_initialization_contract_sha256 must be lowercase SHA256")
    contract, actual = _read_initialization_contract(runtime)
    if actual != expected_sha256:
        raise ValueError("initialization contract SHA256 mismatch; no session created")
    return contract, actual


def _status_for(error: BaseException) -> int:
    if isinstance(error, SessionNotFoundError):
        return 404
    if isinstance(error, (DuplicateSessionError, VersionConflictError, EventConflictError)):
        return 409
    if isinstance(error, (InvalidIdentifierError, ValueError)):
        return 400
    if isinstance(error, GpuRuntimeError):
        return 422
    return 500


class RuntimeHandler(BaseHTTPRequestHandler):
    runtime: GpuRuntime
    backend_name: str
    server_version = "ReapGpuRuntime/1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send(self, status: int, value: Any) -> None:
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 8 * 1024 * 1024:
            raise ValueError("request body exceeds 8 MiB")
        raw = self.rfile.read(length)
        value = json.loads(raw or b"{}")
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _route(self) -> tuple[str, str]:
        match = _SESSION_PATH.fullmatch(self.path.split("?", 1)[0])
        if not match:
            raise ValueError("unknown path")
        return match.group(1), match.group(2) or ""

    def _send_error(self, error: BaseException) -> None:
        status = _status_for(error)
        # Lean may retain only a failed-call marker. Do not log request bodies,
        # headers, query strings, or generated model output here.
        route = _SESSION_PATH.fullmatch(self.path.split("?", 1)[0])
        record = {"event": "gpu_http_error", "time_unix": time.time(),
                  "method": self.command, "session_id": route.group(1) if route else None,
                  "action": (route.group(2) or "create") if route else "unknown",
                  "status": status, "error": type(error).__name__, "message": str(error)[:2000]}
        try:
            print(json.dumps(record, ensure_ascii=True), file=sys.stderr, flush=True)
        except OSError:
            pass  # Diagnostic output must not change the mutation/HTTP outcome.
        self._send(status, {"error": type(error).__name__, "message": str(error)})

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == "/initialization-contract":
            try:
                contract, pin = _read_initialization_contract(self.runtime)
                self._send(200, {"initialization_contract": contract,
                                 "initialization_contract_sha256": pin})
            except BaseException as error:
                self._send(_status_for(error), {"error": type(error).__name__, "message": str(error)})
            return
        if self.path.split("?", 1)[0] == "/health":
            actor_metrics = self.runtime.actor.metrics()
            self._send(200, {
                "ok": True,
                "backend": self.backend_name,
                "gpu_actor_completed": actor_metrics["completed"],
                "gpu_actor_max_active": actor_metrics["max_active"],
                "gpu_actor": actor_metrics,
                "max_resident_sessions": self.runtime.max_resident_sessions,
            })
            return
        if not _SESSION_PATH.fullmatch(self.path.split("?", 1)[0]):
            self._send(404, {"error": "not_found"})
            return
        try:
            session_id, action = self._route()
            if action == "retire/v1":
                self._send(200, self.runtime.retirement_receipt(session_id))
                return
        except BaseException as error:
            self._send(_status_for(error), {"error": type(error).__name__, "message": str(error)})
            return
        self._send(404, {"error": "not_found"})

    def do_POST(self) -> None:
        try:
            session_id, action = self._route()
            body = self._read()
            if action == "":
                if set(body) - {"theorem_id", "experience_id", "experience_weights_sha256", "experience_snapshot_sha256",
                                "model_release_sha256", "role", "expected_initialization_contract_sha256"}:
                    raise ValueError("unknown session initialization fields")
                initialization = None
                if "expected_initialization_contract_sha256" in body:
                    initialization = _initialization_contract(self.runtime, body.pop("expected_initialization_contract_sha256"))
                created = self.runtime.create_session(session_id, **body)
                if initialization is not None:
                    contract, contract_sha256 = initialization
                    created = {**created, "initialization_contract": contract,
                               "initialization_contract_sha256": contract_sha256}
                self._send(201, created)
            elif action == "policy/v1/chat/completions":
                self._send(200, self.runtime.policy(session_id, body))
            elif action == "value/v1/chat/completions":
                self._send(200, self.runtime.value(session_id, body))
            elif action == "learn/v1":
                event = body.get("event")
                self._send(200, self.runtime.learn(
                    session_id,
                    expected_policy_version=body.get("expected_policy_version"),
                    event=event,
                ))
            elif action == "snapshot/v1":
                path = self.runtime.snapshot(session_id, body.get("name"),
                                             for_experience=body.get("for_experience", False))
                receipt = {"session_id": session_id, "snapshot": path.name}
                if body.get("for_experience"):
                    import hashlib
                    state, _ = self.runtime.snapshots.load(session_id, path.name)
                    receipt["source"] = {"session_id": session_id, "theorem_id": state["theorem_id"],
                        "policy_version": state["policy_version"], "snapshot": path.name,
                        "snapshot_sha256": hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest(),
                        "parent_experience_id": state["lineage"].get("experience_id")}
                self._send(201, receipt)
            elif action == "experience/v1":
                self._send(201, self.runtime.publish_experience(session_id, body.get("snapshot"),
                    body.get("experience_id"), body.get("acceptance")))
            elif action == "restore/v1":
                state = self.runtime.restore(session_id, body.get("name"))
                self._send(200, state)
            elif action == "retire/v1":
                if set(body) not in ({"name", "expected_policy_version"}, {"name", "expected_policy_version", "reuse_snapshot"}):
                    raise ValueError("retirement requires name/version and optional explicit reuse_snapshot")
                self._send(200, self.runtime.retire_session(session_id, body["name"],
                    expected_policy_version=body["expected_policy_version"], reuse_snapshot=body.get("reuse_snapshot", False)))
            else:
                self._send(404, {"error": "not_found"})
        except BaseException as error:
            self._send_error(error)

    def do_DELETE(self) -> None:
        try:
            session_id, action = self._route()
            if action:
                self._send(404, {"error": "not_found"})
                return
            self.runtime.delete_session(session_id)
            self._send(200, {"deleted": True, "session_id": session_id})
        except BaseException as error:
            self._send(_status_for(error), {
                "error": type(error).__name__,
                "message": str(error),
            })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8760)
    parser.add_argument("--backend", choices=("toy", "real", "real-search", "real-search-categorical",
                                              "qwen35-search", "verified-replay", "mixed-replay"), default="real")
    def thinking_budget(value: str) -> int:
        number = int(value)
        if not 1 <= number <= 8192:
            raise argparse.ArgumentTypeError("thinking budget must be in [1, 8192]")
        return number
    parser.add_argument("--thinking-budget", type=thinking_budget,
                        help="qwen35-search only: required per-state shared thinking token limit")
    parser.add_argument("--qwen-action-format", choices=("raw-v1", "strict-single-wrapper-v1"),
                        help="qwen35-search only: explicit action serialization contract")
    parser.add_argument("--verified-dataset-root", type=Path,
                        help="verified/mixed-replay: admitted content-addressed Lean replay bundles")
    parser.add_argument("--mathlib-dataset-root", type=Path,
                        help="mixed-replay only: independently verified human Mathlib bundles")
    parser.add_argument("--verified-max-distance", type=int,
                        help="verified/mixed-replay: explicit categorical support 1..D; overflow rejected")
    parser.add_argument("--gamma", type=float, help="required for real-search; must match actual Lean visit_discount")
    parser.add_argument("--categorical-value-artifact", type=Path,
                        help="real-search-categorical: immutable P64/R64 value-head checkpoint")
    parser.add_argument("--categorical-value-artifact-sha256",
                        help="real-search-categorical: exact lowercase artifact SHA256")
    parser.add_argument("--categorical-value-artifact-role",
                        choices=("pretrained", "matched-random-initial"),
                        help="real-search-categorical: expected artifact role")
    parser.add_argument('--success-dataset-root', type=Path,
                        help='opt-in real-search terminal learning from trusted full Lean replay bundles')
    parser.add_argument("--model-path", default="/opt/models/REAL-Prover")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--snapshot-root", type=Path, default=Path("/workspace/out/snapshots"))
    parser.add_argument("--experience-root", type=Path, help="explicit shared immutable release store")
    parser.add_argument("--learner-release-root", type=Path, help="explicit independent learner checkpoint/release store")
    parser.add_argument("--learner-profile", choices=("continual-mixed-v3",),
                        help="opt-in append-only mixed learner store; default fixed v2 remains unchanged")
    def positive_capacity(value: str) -> int:
        number = int(value)
        if number < 1:
            raise argparse.ArgumentTypeError("resident session capacity must be positive")
        return number
    parser.add_argument("--max-resident-sessions", type=positive_capacity,
                        help="opt-in resident-state cap; callers must explicitly retire completed sessions")
    def positive_finite(value: str) -> float:
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise argparse.ArgumentTypeError("value must be finite and positive")
        return number
    parser.add_argument("--max-post-update-kl", type=positive_finite,
                        help="real-search/verified-replay: opt-in post-step current-to-base KL rejection and rollback")
    parser.add_argument("--policy-scoring", choices=("tokenwise", "candidate_chunks", "tokenwise_deferred"), default="tokenwise",
                        help="explicit raw-logprob implementation; default remains unchanged")
    parser.add_argument("--policy-score-batch-size", type=int, default=2)
    parser.add_argument("--policy-score-token-chunk", type=int, default=8)
    return parser


def build_backend(args: argparse.Namespace):
    # Validate optional modes before allocating/loading model weights.
    qwen_search = args.backend == 'qwen35-search'
    thinking_budget = getattr(args, 'thinking_budget', None)
    action_format = getattr(args, 'qwen_action_format', None)
    if not qwen_search and action_format is not None:
        raise ValueError('--qwen-action-format requires qwen35-search')
    if qwen_search and thinking_budget is None:
        raise ValueError('qwen35-search requires an explicit --thinking-budget')
    if not qwen_search and thinking_budget is not None:
        raise ValueError('--thinking-budget requires qwen35-search')
    categorical_search = args.backend == "real-search-categorical"
    categorical_fields = (args.categorical_value_artifact,
                          args.categorical_value_artifact_sha256,
                          args.categorical_value_artifact_role)
    if categorical_search != all(value is not None for value in categorical_fields):
        raise ValueError("categorical value artifact/path/SHA/role are jointly required only for real-search-categorical")
    if not categorical_search and any(value is not None for value in categorical_fields):
        raise ValueError("categorical value artifact flags require real-search-categorical")
    if getattr(args, 'success_dataset_root', None) is not None and args.backend not in {
            'real-search', 'real-search-categorical', 'qwen35-search'}:
        raise ValueError('--success-dataset-root requires a search backend')
    if args.learner_profile is not None and (args.backend != "mixed-replay" or args.learner_release_root is None):
        raise ValueError("continual-mixed-v3 requires mixed-replay and an explicit learner release root")
    scoring = PolicyScoringConfig(args.policy_scoring, args.policy_score_batch_size, args.policy_score_token_chunk)
    if qwen_search and scoring.mode != 'tokenwise':
        raise ValueError('qwen35-search does not support alternative scoring modes')
    if scoring.mode != "candidate_chunks" and (scoring.candidate_batch_size != 2 or scoring.token_chunk_size != 8):
        raise ValueError("scoring size flags require --policy-scoring candidate_chunks")
    if args.max_post_update_kl is not None and args.backend not in {
            "real-search", "real-search-categorical", "verified-replay", "mixed-replay"}:
        raise ValueError("--max-post-update-kl requires real-search or verified/mixed-replay")
    if args.backend not in {"verified-replay", "mixed-replay"} and (args.verified_dataset_root is not None or args.verified_max_distance is not None):
        raise ValueError("verified dataset/support flags require verified-replay or mixed-replay")
    if args.backend != "mixed-replay" and args.mathlib_dataset_root is not None:
        raise ValueError("Mathlib dataset root requires mixed-replay")
    if args.backend == "toy" and scoring.mode != "tokenwise":
        raise ValueError("nondefault policy scoring requires a real model backend")
    options = {"policy_scoring": scoring} if scoring.mode != "tokenwise" else {}
    if args.backend in {"verified-replay", "mixed-replay"}:
        if args.gamma is not None:
            raise ValueError("verified replay uses undiscounted successful returns, not gamma conversion")
        if args.verified_dataset_root is None or args.verified_max_distance is None:
            raise ValueError("verified-replay requires explicit dataset root and maximum distance")
        from .verified_backend import VerifiedReplayBackend
        backend_class = VerifiedReplayBackend
        if args.backend == "mixed-replay":
            if args.mathlib_dataset_root is None:
                raise ValueError("mixed-replay requires an explicit Mathlib dataset root")
            from .mixed_backend import MixedReplayBackend
            backend_class = MixedReplayBackend
            options["mathlib_dataset_root"] = args.mathlib_dataset_root
        if args.max_post_update_kl is not None:
            options["max_post_update_kl"] = args.max_post_update_kl
        return backend_class(args.model_path, device=args.device,
            dataset_root=args.verified_dataset_root, max_distance=args.verified_max_distance, **options)
    if args.backend in {"real-search", "real-search-categorical", "qwen35-search"}:
        if args.gamma is None:
            raise SystemExit("--gamma is required for search backends")
        if args.max_post_update_kl is not None:
            options["max_post_update_kl"] = args.max_post_update_kl
        if getattr(args, 'success_dataset_root', None) is not None:
            options['success_dataset_root'] = args.success_dataset_root
        if qwen_search:
            from .qwen35_backend import Qwen35SearchBackend
            if action_format is not None:
                options['action_format'] = action_format
            return Qwen35SearchBackend(args.model_path, device=args.device, gamma=args.gamma,
                thinking_budget=thinking_budget, thinking_mode='shared-state-context-v1', **options)
        if categorical_search:
            from .categorical_search_backend import CategoricalSearchBackend
            return CategoricalSearchBackend(args.model_path, device=args.device, gamma=args.gamma,
                value_artifact=args.categorical_value_artifact,
                value_artifact_sha256=args.categorical_value_artifact_sha256,
                value_artifact_role=args.categorical_value_artifact_role, **options)
        return RealSearchBackend(args.model_path, device=args.device, gamma=args.gamma, **options)
    elif args.backend == "toy":
        return ToyBackend()
    return RealProverBackend(args.model_path, device=args.device, **options)


def main() -> int:
    args = build_parser().parse_args()
    backend = build_backend(args)
    runtime = GpuRuntime(backend=backend, snapshot_root=args.snapshot_root, experience_root=args.experience_root,
                         learner_release_root=args.learner_release_root,
                         learner_profile=args.learner_profile,
                         max_resident_sessions=args.max_resident_sessions)
    RuntimeHandler.runtime = runtime
    RuntimeHandler.backend_name = args.backend
    server = ThreadingHTTPServer((args.host, args.port), RuntimeHandler)
    try:
        print(json.dumps({"ready": True, "host": args.host, "port": args.port, "backend": args.backend,
                          "policy_scoring": args.policy_scoring,
                          "policy_score_batch_size": args.policy_score_batch_size,
                          "policy_score_token_chunk": args.policy_score_token_chunk,
                          "max_post_update_kl": args.max_post_update_kl}), flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
