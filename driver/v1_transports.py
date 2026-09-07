"""driver/v1_transports.py — transport 后端选择（v1-1）

模式（env `V11_TRANSPORT`）:
  mock       本地 mock_policy_server（smoke 默认，零云端负载）
  local-opencode 本地 opencode 证据环（evid backend，默认 stub）
  remote     云端/用户侧 GPU HttpPolicyServer（仅注册；本 smoke 不启用）
"""
import os
import json
import urllib.request

def _post(url, payload, timeout=10):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

class MockTransport:
    """本地 mock 后端：CPU-smoke。可被任何 HTTP 契约后端替换（同 3 端点）。"""
    def __init__(self, base: str):
        self.base = base
    def policy(self, prompt: str, n: int = 4) -> list:
        out = _post(self.base + "/v1/chat/completions", {"prompt": prompt, "n": n})
        return [(c.get("text", ""), c.get("logprob_avg", 0.0)) for c in out.get("choices", [])]
    def value(self, prompt: str) -> float:
        out = _post(self.base + "/value", {"prompt": prompt})
        return float(out.get("score", -1000.0))
    def approach(self, prompt: str) -> list:
        return _post(self.base + "/premises", {"prompt": prompt})

class OpencodeEvidenceBackend:
    """本地 opencode 证据环（v1-1 证据生产者骨架）。

    真正接线：通过 opencode CLI/run 子进程产出 linkSearch/leanSearch/fileSearch 证据。
    smoke 阶段用固定 evidence；后续替换为实际调用。
    """
    def evidence_snapshot(self, problem_id: str, step: int) -> list:
        return [
            {"id": f"{problem_id}-l0", "kind": "linkSearch", "weight": 0.30,
             "payload": "https://example.org/lemma:mathlib:sq_connected", "sourceDesc": "opencode:link"},
            {"id": f"{problem_id}-s0", "kind": "leanSearch", "weight": 0.25,
             "payload": "lemma sq_connected {a b : IR} : a^2+1 = b^2+1 -> a = b or a = -b",
             "sourceDesc": "opencode:lean-search"},
        ]

def make_transport():
    mode = os.environ.get("V11_TRANSPORT", "mock")
    if mode == "mock":
        base = os.environ.get("MOCK_POLICY_URL", "http://127.0.0.1:8760")
        return MockTransport(base)
    if mode == "local-opencode":
        return MockTransport("http://127.0.0.1:8760")  # 证据环后端单独启用
    if mode == "remote":
        base = os.environ.get("V11_POLICY_URL", "https://100.91.25.4:8000")
        return MockTransport(base)
    raise SystemExit(f"unknown V11_TRANSPORT={mode}")
