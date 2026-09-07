"""Small, strict OpenAI Chat-compatible schemas without third-party imports."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
import uuid
from typing import Any


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: tuple[dict[str, str], ...]
    n: int
    temperature: float
    max_tokens: int
    logprobs: bool

    @property
    def prompt(self) -> str:
        return "\n".join(message["content"] for message in self.messages)


def parse_chat_request(raw: dict[str, Any]) -> ChatRequest:
    if not isinstance(raw, dict):
        raise ValueError("chat request must be an object")
    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array")
    parsed_messages: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"messages[{index}].role is invalid")
        if not isinstance(content, str):
            raise ValueError(f"messages[{index}].content must be a string")
        parsed_messages.append({"role": role, "content": content})

    n = raw.get("n", 1)
    max_tokens = raw.get("max_tokens", 1024)
    temperature = raw.get("temperature", 0.99)
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 64:
        raise ValueError("n must be an integer in [1, 64]")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= 8192:
        raise ValueError("max_tokens must be an integer in [1, 8192]")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or temperature < 0:
        raise ValueError("temperature must be a non-negative number")
    model = raw.get("model", "reap")
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a non-empty string")
    logprobs = raw.get("logprobs", False)
    if not isinstance(logprobs, bool):
        raise ValueError("logprobs must be boolean")
    return ChatRequest(
        model=model,
        messages=tuple(parsed_messages),
        n=n,
        temperature=float(temperature),
        max_tokens=max_tokens,
        logprobs=logprobs,
    )


def chat_response(
    *,
    model: str,
    contents: list[str],
    token_logprobs: list[list[dict[str, Any]]] | None = None,
    policy_version: int,
) -> dict[str, Any]:
    if token_logprobs is not None and len(token_logprobs) != len(contents):
        raise ValueError("token_logprobs and contents lengths differ")
    choices = []
    for index, content in enumerate(contents):
        choice: dict[str, Any] = {
            "index": index,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }
        if token_logprobs is not None:
            choice["logprobs"] = {"content": token_logprobs[index]}
        choices.append(choice)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "policy_version": policy_version,
        "choices": choices,
    }


def value_chat_response(*, model: str, score: float, policy_version: int) -> dict[str, Any]:
    # Reap parses choice.message.content as JSON and applies its own sign
    # conversion.  Keep the transport OpenAI Chat-compatible.
    content = json.dumps({"score": float(score)}, separators=(",", ":"), allow_nan=False)
    return chat_response(model=model, contents=[content], policy_version=policy_version)
