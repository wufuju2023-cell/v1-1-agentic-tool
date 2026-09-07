#!/usr/bin/env python3
"""v1_smoke.py — v1-1 smoke（零云端干扰，默认 mock 后端）

链路：problems.json(3 题) → policy/value transport → 证据环(本地 opencode 骨架) →
     phi 重加权 → 报告断言。
环境:
  V11_TRANSPORT=mock        # 默认: 本地 mock (8760)
  MOCK_POLICY_URL=...       # 可覆盖
"""
import json, os, sys, time, pathlib, hashlib

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "driver"))

from v1_contract import Evidence, RingEvent, evidence_composite
from v1_transports import make_transport, OpencodeEvidenceBackend
from runtime_transport import RuntimeTransport


def make_transport_for(transport: str, base: str):
    if transport == "runtime":
        return RuntimeTransport(base)
    return make_transport()


def plausible_prompt(problem: dict, related: list[str]) -> str:
    """构造 Lean 侧 mkPrompt 等价物（v1-1 Lean 实现见 Reap/Tactic/Generator#mkPrompt）"""
    ps = "\n".join(f"lemma {r}" for r in related if r)
    return ("State:\n" + problem["goal"] + "\n\nRelated theorems:\n" + (ps or "(none)"))


def run_one(problem: dict, transport) -> dict:
    pid = problem["id"]
    prompt = plausible_prompt(problem, [])
    t0 = time.time()

    # 1) policy（应为 ≥1 条 candidate；GPU 真任务运行时走同一契约）
    candidates = transport.policy(prompt, n=4)
    # 2) value（-d 语义由 GPU 侧保证；此处只验证标量）
    value = transport.value(prompt)
    # 3) 证据环：本地 opencode 骨架证据（0 云端负载）
    backend = OpencodeEvidenceBackend()
    evs = [Evidence(**e) for e in backend.evidence_snapshot(pid, step=0)]
    phi = evidence_composite(evs)
    ring = RingEvent(atStep=0, kind="leanSearch", evidenceId=evs[0].id if evs else "none",
                     payloadHash=hashlib.sha256(pid.encode()).hexdigest()[:12], phi=phi)

    ok = bool(candidates) and isinstance(value, float)
    return {"id": pid, "candidates": len(candidates), "sample": candidates[0] if candidates else None,
            "value": value, "phi": phi, "ring_events": 1, "ok": ok,
            "wall_ms": int((time.time() - t0) * 1000)}


def main():
    transport = os.environ.get("V11_TRANSPORT", "mock")
    base = os.environ.get("V11_POLICY_URL", "http://127.0.0.1:8760")
    t = make_transport_for(transport, base)
    problems = json.loads((ROOT / "problems.json").read_text())
    print(f"[v1-smoke] transport={type(t).__name__} problems={len(problems)}")
    rows = []
    for p in problems:
        if transport == "runtime":
            r = t.run_problem(p)
        else:
            r = run_one(p, t)
        rows.append(r)
        print(f"  - {r['id']:44s} candidates={r['candidates']:<2d} value={r['value']:.4f} ok={r['ok']} {r['wall_ms']}ms")
    n_ok = sum(1 for r in rows if r["ok"])
    assert n_ok == len(rows), f"smoke FAIL {n_ok}/{len(rows)}"
    print(f"[v1-smoke] PASS {n_ok}/{len(rows)}")


if __name__ == "__main__":
    main()
