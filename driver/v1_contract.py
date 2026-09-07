"""v1_contract.py — v1-1 smoke 契约（对齐 cpulean/Reap/Agentic.lean）"""
from dataclasses import dataclass, field, asdict
from typing import Optional

EVIDENCE_KINDS = ["linkSearch", "leanSearch", "fileSearch", "toolCall", "manual"]

@dataclass
class Evidence:
    id: str
    kind: str
    weight: float
    payload: str
    sourceDesc: str = ""
    def to_json(self) -> dict:
        return {"id": self.id, "kind": self.kind, "weight": self.weight,
                "payload": self.payload, "sourceDesc": self.sourceDesc}

@dataclass
class RingEvent:
    atStep: int
    kind: str
    evidenceId: str
    payloadHash: str
    phi: float
    def to_json(self) -> dict:
        return asdict(self)

def evidence_composite(evs: list[Evidence]) -> float:
    import math
    ell = sum(e.weight for e in evs)
    return math.tanh(ell)          # phi = tanh(sum w)，饱和压缩（Lean 侧同公式）

def ring_log(pred: list[RingEvent]) -> dict:
    return {"events": [e.to_json() for e in pred]}

def policy_request(prompt: str, n: int = 4, temperature: float = 1.0) -> dict:
    return {"prompt": prompt, "n": n, "temperature": temperature}

def value_request(prompt: str) -> dict:
    return {"prompt": prompt}
