"""Immutable, checksummed, atomically published session snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Any

from .errors import SnapshotIntegrityError, SnapshotNotFoundError
from .identifiers import contained_path, validate_identifier


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_durable(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


class SnapshotStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        session_id: str,
        name: str,
        *,
        session_state: dict[str, Any],
        backend_state: dict[str, Any],
    ) -> Path:
        session_id = validate_identifier(session_id, kind="session_id")
        name = validate_identifier(name, kind="snapshot name")
        session_root = contained_path(self.root, session_id)
        session_root.mkdir(parents=True, exist_ok=True)
        target = contained_path(session_root, name)
        if target.exists():
            raise FileExistsError(f"snapshot already exists: {session_id}/{name}")
        temporary = contained_path(session_root, f".{name}.tmp-{uuid.uuid4().hex}")
        temporary.mkdir()
        try:
            payloads = {
                "session.json": _json_bytes(session_state),
                "backend.json": _json_bytes(backend_state),
            }
            files: dict[str, dict[str, Any]] = {}
            for filename, data in payloads.items():
                _write_durable(temporary / filename, data)
                files[filename] = {"sha256": _sha256(data), "bytes": len(data)}
            manifest = {
                "schema_version": "reap.gpu.snapshot.v1",
                "session_id": session_id,
                "snapshot": name,
                "files": files,
            }
            _write_durable(temporary / "manifest.json", _json_bytes(manifest))
            os.replace(temporary, target)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return target

    def load(self, session_id: str, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
        session_id = validate_identifier(session_id, kind="session_id")
        name = validate_identifier(name, kind="snapshot name")
        target = contained_path(self.root, session_id, name)
        if not target.is_dir():
            raise SnapshotNotFoundError(f"snapshot not found: {session_id}/{name}")
        try:
            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotIntegrityError("snapshot manifest is missing or invalid") from exc
        if manifest.get("schema_version") != "reap.gpu.snapshot.v1":
            raise SnapshotIntegrityError("unsupported snapshot schema")
        if manifest.get("session_id") != session_id or manifest.get("snapshot") != name:
            raise SnapshotIntegrityError("snapshot identity mismatch")
        files = manifest.get("files")
        if not isinstance(files, dict) or set(files) != {"session.json", "backend.json"}:
            raise SnapshotIntegrityError("snapshot file manifest is incomplete")
        decoded: dict[str, dict[str, Any]] = {}
        for filename in ("session.json", "backend.json"):
            path = contained_path(target, filename)
            try:
                data = path.read_bytes()
            except OSError as exc:
                raise SnapshotIntegrityError(f"snapshot file is missing: {filename}") from exc
            expected = files[filename]
            if expected.get("bytes") != len(data) or expected.get("sha256") != _sha256(data):
                raise SnapshotIntegrityError(f"snapshot checksum mismatch: {filename}")
            try:
                decoded[filename] = json.loads(data)
            except json.JSONDecodeError as exc:
                raise SnapshotIntegrityError(f"snapshot JSON is invalid: {filename}") from exc
        return decoded["session.json"], decoded["backend.json"]

    def verify_for_reuse(self, session_id: str, name: str, *, expected_manifest_sha256: str) -> dict[str, Any]:
        """Verify every immutable byte without materializing backend base64.

        The expected manifest digest comes from this runtime's successful
        snapshot creation, not from a caller-supplied current disk manifest.
        Only the logical session JSON is decoded. Backend bytes are streamed.
        """
        session_id = validate_identifier(session_id, kind="session_id")
        name = validate_identifier(name, kind="snapshot name")
        target = contained_path(self.root, session_id, name)
        if not target.is_dir():
            raise SnapshotNotFoundError(f"snapshot not found: {session_id}/{name}")

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise SnapshotIntegrityError("duplicate snapshot JSON key")
                result[key] = value
            return result

        try:
            manifest_path = contained_path(target, "manifest.json")
            with manifest_path.open("rb") as stream:
                raw_manifest = stream.read(65537)
            if len(raw_manifest) > 65536 or _sha256(raw_manifest) != expected_manifest_sha256:
                raise SnapshotIntegrityError("snapshot manifest differs from runtime witness")
            manifest = json.loads(raw_manifest, object_pairs_hook=unique)
            if (not isinstance(manifest, dict) or set(manifest) != {"schema_version", "session_id", "snapshot", "files"}
                    or manifest["schema_version"] != "reap.gpu.snapshot.v1"
                    or manifest["session_id"] != session_id or manifest["snapshot"] != name):
                raise SnapshotIntegrityError("snapshot identity/schema mismatch")
            files = manifest["files"]
            if not isinstance(files, dict) or set(files) != {"session.json", "backend.json"}:
                raise SnapshotIntegrityError("snapshot file manifest is incomplete")
            logical_bytes = bytearray()
            for filename in ("session.json", "backend.json"):
                expected = files[filename]
                if (not isinstance(expected, dict) or set(expected) != {"bytes", "sha256"}
                        or type(expected["bytes"]) is not int or expected["bytes"] < 0
                        or not isinstance(expected["sha256"], str) or len(expected["sha256"]) != 64
                        or any(c not in "0123456789abcdef" for c in expected["sha256"])):
                    raise SnapshotIntegrityError("invalid snapshot file checksum metadata")
                digest, count = hashlib.sha256(), 0
                with contained_path(target, filename).open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        count += len(chunk)
                        if count > expected["bytes"]:
                            raise SnapshotIntegrityError(f"snapshot byte count mismatch: {filename}")
                        digest.update(chunk)
                        if filename == "session.json":
                            logical_bytes.extend(chunk)
                if count != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
                    raise SnapshotIntegrityError(f"snapshot checksum mismatch: {filename}")
            # Detect ordinary replacement while checking the payloads too.
            with manifest_path.open("rb") as stream:
                current_manifest = stream.read(65537)
            if current_manifest != raw_manifest:
                raise SnapshotIntegrityError("snapshot manifest changed during verification")
            logical = json.loads(logical_bytes, object_pairs_hook=unique)
            if not isinstance(logical, dict):
                raise SnapshotIntegrityError("snapshot logical state must be an object")
            return logical
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotIntegrityError("snapshot missing or invalid during reuse verification") from exc
