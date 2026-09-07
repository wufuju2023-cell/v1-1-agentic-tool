"""Canonical, platform-independent manifest for a pinned safetensors base.

This reads immutable files only.  It does not import torch, instantiate a
model, or serialize environment-owned configuration objects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import struct
from typing import Any


SCHEMA = "reap.frozen-safetensors-base-manifest.v2"
LOCK_SCHEMA = "reap.model-lock.v2"
SHA256 = __import__("re").compile(r"[0-9a-f]{64}\Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (not isinstance(relative, str) or pure.is_absolute()
            or pure.as_posix() != relative
            or any(part in ("", ".", "..") for part in pure.parts)):
        raise ValueError("unsafe model-lock path")
    path = root.joinpath(*pure.parts)
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("model-lock file missing, linked, or outside model root")
    return path


def _tensor_rows(path: Path, shard: str) -> list[dict[str, Any]]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError("truncated safetensors length")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length <= 1 or header_length > size - 8:
            raise ValueError("invalid safetensors header length")
        raw_header = stream.read(header_length)
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid safetensors header JSON") from exc
    if not isinstance(header, dict):
        raise ValueError("invalid safetensors header")
    data_start = 8 + header_length
    descriptors = []
    for name, value in header.items():
        if name == "__metadata__":
            continue
        if (not isinstance(name, str) or not name or not isinstance(value, dict)
                or set(value) != {"dtype", "shape", "data_offsets"}
                or not isinstance(value["dtype"], str)
                or not isinstance(value["shape"], list)
                or not all(type(x) is int and x >= 0 for x in value["shape"])
                or not isinstance(value["data_offsets"], list)
                or len(value["data_offsets"]) != 2
                or not all(type(x) is int and x >= 0 for x in value["data_offsets"])):
            raise ValueError("invalid safetensors tensor descriptor")
        start, end = value["data_offsets"]
        if end < start or data_start + end > size:
            raise ValueError("safetensors tensor range outside file")
        descriptors.append((start, end, name, value))
    descriptors.sort()
    previous = 0
    rows = []
    with path.open("rb") as stream:
        for start, end, name, value in descriptors:
            if start != previous:
                raise ValueError("safetensors payload has a gap or overlap")
            stream.seek(data_start + start)
            remaining = end - start
            digest = hashlib.sha256()
            while remaining:
                block = stream.read(min(4 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError("truncated safetensors tensor payload")
                digest.update(block); remaining -= len(block)
            rows.append({"name": name, "shard": shard, "dtype": value["dtype"],
                         "shape": value["shape"], "bytes": end-start,
                         "sha256": digest.hexdigest()})
            previous = end
    if data_start + previous != size:
        raise ValueError("unclaimed bytes after safetensors payload")
    return rows


def build_manifest(model_root: str | Path) -> dict[str, Any]:
    root = Path(model_root).resolve()
    lock_path = _regular(root, "reap-model-lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock.get("schema_version") != LOCK_SCHEMA or not isinstance(lock.get("files"), dict)
            or not isinstance(lock.get("repo"), str) or not isinstance(lock.get("revision"), str)):
        raise ValueError("unsupported model verification lock")
    file_rows = []
    for name, expected in sorted(lock["files"].items()):
        if (not isinstance(expected, dict) or type(expected.get("size")) is not int
                or not isinstance(expected.get("sha256"), str)
                or SHA256.fullmatch(expected["sha256"]) is None):
            raise ValueError("invalid model-lock file pin")
        path = _regular(root, name)
        actual = _sha256(path)
        if path.stat().st_size != expected["size"] or actual != expected["sha256"]:
            raise ValueError("model file differs from verification lock: " + name)
        file_rows.append({"path": name, "bytes": expected["size"], "sha256": actual})
    index = json.loads(_regular(root, "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("safetensors index has no weight map")
    shards = sorted(set(weight_map.values()))
    if not all(isinstance(name, str) and name.endswith(".safetensors") for name in shards):
        raise ValueError("invalid safetensors shard name")
    tensors = []
    for shard in shards:
        tensors.extend(_tensor_rows(_regular(root, shard), shard))
    tensors.sort(key=lambda row: row["name"])
    if (len(tensors) != len(weight_map) or len({row["name"] for row in tensors}) != len(tensors)
            or any(weight_map.get(row["name"]) != row["shard"] for row in tensors)
            or set(weight_map) != {row["name"] for row in tensors}):
        raise ValueError("safetensors index/header tensor mapping mismatch")
    expected_count = lock.get("safetensors", {}).get("tensor_count")
    expected_bytes = lock.get("safetensors", {}).get("total_tensor_bytes")
    if len(tensors) != expected_count or sum(row["bytes"] for row in tensors) != expected_bytes:
        raise ValueError("safetensors aggregate differs from verification lock")
    manifest = {"schema_version": SCHEMA, "repository": lock["repo"],
                "revision": lock["revision"], "model_lock_sha256": _sha256(lock_path),
                "files": file_rows, "tensor_count": len(tensors),
                "total_tensor_bytes": sum(row["bytes"] for row in tensors),
                "tensors": tensors}
    # The lock file itself is an audit receipt and may differ in harmless
    # serialization metadata between platforms.  Every claim it makes has
    # already been checked above against the actual immutable files.  Bind the
    # transferable identity to those verified claims and tensor bytes, while
    # retaining the local lock-file hash explicitly for audit.
    identity = {key: value for key, value in manifest.items()
                if key != "model_lock_sha256"}
    manifest["sha256"] = hashlib.sha256(_canonical(identity)).hexdigest()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    manifest = build_manifest(args.model_root)
    with output.open("xb") as stream:
        stream.write(_canonical(manifest) + b"\n")
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"manifest_sha256": manifest["sha256"],
                      "tensor_count": manifest["tensor_count"],
                      "total_tensor_bytes": manifest["total_tensor_bytes"]}, sort_keys=True))


if __name__ == "__main__":
    main()
