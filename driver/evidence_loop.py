"""evidence_loop.py — v1-1 后验闭环：证据注入 → prompt 条件化 → GPU policy/value 对比

对应 9-7 系列 09-bayesian-view 的 B 路（推理时条件注入）：
  P(t|s,R) ∝ P(t|s)·P(R|t,s)   工程实现 = 证据内容进 prompt prior 段。
"""
import json
import pathlib
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from v1_contract import Evidence, RingEvent, evidence_composite
from v1_transports import OpencodeEvidenceBackend
from runtime_transport import RuntimeTransport


def build_prompt(goal: str, related: list[str]) -> str:
    ps = "\n".join(f"lemma {r}" for r in related if r)
    return ("State:\n" + goal + "\n\nRelated theorems:\n" + (ps or "(none)"))


@dataclass
class Trial:
    pid: str
    value0: float
    value1: float
    phi: float
    samples: list
    delta: float


def run_problem_with_evidence(t: RuntimeTransport, problem: dict, evidence_backend) -> Trial:
    pid = problem["id"]
    goal = problem["goal"]
    s0 = t.session(pid)
    try:
        t._last_sid = s0
        v0 = t.value(build_prompt(goal, []))
        t.retire(s0)
    except Exception:
        v0 = -1000.0
    s1 = t.session(pid)
    try:
        t._last_sid = s1
        evs = [Evidence(**e) for e in evidence_backend.evidence_snapshot(pid, step=0)]
        phi = evidence_composite(evs)
        hints = [e.payload for e in evs if e.kind == "leanSearch"]
        cands = t.policy(build_prompt(goal, hints), n=4)
        v1 = t.value(build_prompt(goal, hints))
        t.retire(s1)
    except Exception:
        v1 = -1000.0
        cands = []
    return Trial(pid=pid, value0=v0, value1=v1, phi=phi, samples=cands, delta=v1 - v0)


def main():
    import os
    base = os.environ.get("V11_POLICY_URL", "http://100.91.25.4:8000")
    t = RuntimeTransport(base)
    problems = json.loads((pathlib.Path(__file__).resolve().parent.parent /
                           "smoke" / "problems.json").read_text())
    backend = OpencodeEvidenceBackend()
    ok = 0
    for p in problems:
        trial = run_problem_with_evidence(t, p, backend)
        print(f"  - {trial.pid:44s} value0={trial.value0:7.3f} value1={trial.value1:7.3f} "
              f"delta={trial.delta:+.3f} phi={trial.phi:.4f} candidates={len(trial.samples)}")
        if trial.value1 > -1000.0:
            ok += 1
    print(f"[v1-1 loop] PASS {ok}/{len(problems)}")


if __name__ == "__main__":
    main()
