"""runtime_transport.py — 真实 GPU RuntimeHandler 协议（session-based）

端点（见 gpu_runtime/server.py）:
  POST /sessions/{id}                        create (body: theorem_id, role…)
  POST /sessions/{id}/policy/v1/chat/completions   body: prompt, n, temperature, model
  POST /sessions/{id}/value/v1/chat/completions    body: prompt, model
  GET  /sessions/{id}/retire/v1              回收

仅测试用途：smoke 每题 1 会话 4 policy + 1 value，随用随 retire。
"""
import json
import time
import urllib.request

from v1_transports import MockTransport


class RuntimeTransport(MockTransport):
    def __init__(self, base: str, model: str = "awesome-reaper"):
        self.base = base.rstrip("/")
        self.model = model

    def _post(self, url: str, payload: dict, timeout: int = 120) -> dict:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def session(self, pid: str) -> str:
        sid = f"v11-smoke-{pid}-{int(time.time()) % 1000000}"
        self._post(f"{self.base}/sessions/{sid}", {"theorem_id": pid})
        return sid

    def _chat(self, sid: str, action: str, payload: dict) -> dict:
        return self._post(f"{self.base}/sessions/{sid}/{action}", payload)

    def policy(self, prompt: str, n: int = 4) -> list:
        sid = self._last_sid
        out = self._chat(sid, "policy/v1/chat/completions",
                         {"model": self.model,
                          "messages": [{"role": "user", "content": prompt}],
                          "n": n, "temperature": 1.0})
        return [(c.get("text", ""), c.get("logprob", 0.0)) for c in out.get("choices", [])]

    def value(self, prompt: str) -> float:
        sid = self._last_sid
        out = self._chat(sid, "value/v1/chat/completions",
                         {"model": self.model,
                          "messages": [{"role": "user", "content": prompt}]})
        try:
            inner = out["choices"][0]["message"]["content"]
            return float(json.loads(inner).get("score", -1000.0))
        except Exception:
            return -1000.0

    def retire(self, sid: str) -> None:
        import urllib.request as _rq
        try:
            with _rq.urlopen(f"{self.base}/sessions/{sid}/retire/v1", timeout=30):
                pass
        except Exception:
            pass

    def run_problem(self, problem: dict) -> dict:
        pid = problem["id"]
        self._last_sid = self.session(pid)
        prompt = ("State:\n" + problem["goal"] + "\n\nRelated theorems:\n(none)")
        t0 = time.time()
        try:
            candidates = self.policy(prompt, n=4)
            value = self.value(prompt)
        finally:
            self.retire(self._last_sid)
        return {"id": pid, "candidates": len(candidates),
                "sample": candidates[0] if candidates else None,
                "value": value, "ok": bool(candidates) and isinstance(value, float),
                "wall_ms": int((time.time() - t0) * 1000)}
