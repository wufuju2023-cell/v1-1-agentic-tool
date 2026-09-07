"""Opt-in publication heads over the existing learner control log.

No mutable HEAD, no implicit source-release bootstrap. Atomic file links work
on the existing NFS path; unresolved/staged writes fail closed. Read-only
callers do NOT gain atomicity with another process's attempt reservation.
"""
from copy import deepcopy
from pathlib import Path
import re
import uuid

from .learner_release_store import canonical_bytes, content_sha256

DIRECTORY = "release-head"
SCHEMA = "reap.learner.release-head.v1"


def _require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read_training_commits(control, run_sha):
    from .learner import _no_links, _read_control
    _no_links(control)
    names = {p.name for p in control.iterdir()}
    _require("created.json" in names and _read_control(control/"created.json") == {"run_sha256": run_sha},
             "durable learner run identity missing")
    names.remove("created.json"); names.discard(DIRECTORY)
    commits = []; head = None; step = 1
    while names:
        a, b = f"intent-{step:08d}.json", f"commit-{step:08d}.json"
        _require(a in names and b in names, "durable learner has an unresolved intent or invalid step gap; reconcile before resuming")
        intent, commit = _read_control(control/a), _read_control(control/b)
        _require(intent.get("run_sha256") == run_sha and type(intent.get("step")) is int and intent["step"] == step
            and intent.get("parent_checkpoint_sha256") == head
            and set(commit) == {"step", "checkpoint_sha256", "intent_sha256"}
            and type(commit["step"]) is int and commit["step"] == step
            and commit["intent_sha256"] == content_sha256(intent), "durable learner intent/commit chain differs")
        head = commit["checkpoint_sha256"]; commits.append(commit)
        names.difference_update((a, b)); step += 1
    return commits


def _write_record(directory, name, value):
    # Reuse learner CAS's no-replace link over an already fsynced regular file.
    # No dependency on CPU scheduling modules is needed in the GPU image.
    from cpu_runtime.verified_dataset_store import safe_directory
    from .learner_release_store import _link_noreplace
    with safe_directory(directory) as handle:
        staging = ".staged-" + uuid.uuid4().hex
        raw = canonical_bytes(value)
        handle.write_new(staging, raw)
        _require(handle.read(staging) == raw, "publication staging changed")
        _link_noreplace(handle, staging, handle, name)
        handle.sync()


def _records(directory, *, permitted_staging=()):
    from .learner import _no_links, _read_control
    _no_links(directory)
    files = list(directory.iterdir())
    records = {p.name: _read_control(p) for p in files if not p.name.startswith(".staged-")}
    bodies = {canonical_bytes(v) for v in records.values()} | set(permitted_staging)
    for p in files:
        if p.name.startswith(".staged-"):
            _require(canonical_bytes(_read_control(p)) in bodies,
                     "unresolved publication staging; inspect without retry")
    return records


def _mode(run_sha, run):
    return {"schema_version": SCHEMA, "run_sha256": run_sha,
        "contract_sha256": content_sha256(run["contract"]), "scope_sha256": content_sha256(run["scope"]),
        "policy": "publish-each-committed-step", "bootstrap": "no-source-release-alias"}


def publication_state(store, run_sha, *, commits=None, verify_release=True, permitted_staging=()):
    _require(isinstance(run_sha, str) and re.fullmatch(r"[0-9a-f]{64}", run_sha), "invalid publication run SHA")
    control = store.root/"control"/run_sha
    if commits is None:
        commits = read_training_commits(control, run_sha)
    directory = control/DIRECTORY
    from .learner import _no_links
    _no_links(directory)
    if not directory.exists():
        return None
    run = store.load_run(run_sha); records = _records(directory, permitted_staging=permitted_staging)
    _require(records.pop("mode.json", None) == _mode(run_sha, run), "publication mode/run/contract/scope differs")
    head = None; pending = None
    for commit in commits:
        step = commit["step"]; pin = commit["checkpoint_sha256"]
        intent_name, done_name = f"intent-{step:08d}.json", f"published-{step:08d}.json"
        expected = {"schema_version": SCHEMA, "run_sha256": run_sha, "step": step,
            "checkpoint_sha256": pin, "previous_head_sha256": None if head is None else content_sha256(head)}
        intent = records.pop(intent_name, None); done = records.pop(done_name, None)
        if intent is None or done is None:
            _require(pending is None and step == len(commits) and done is None,
                     "publication gap or training advanced beyond unpublished head")
            _require(intent is None or canonical_bytes(intent) == canonical_bytes(expected), "publication intent differs from committed checkpoint")
            pending = {"intent": intent, "expected_intent": expected}
            continue
        _require(canonical_bytes(intent) == canonical_bytes(expected), "publication intent differs from committed checkpoint")
        _validate_confirmation(store, run_sha, expected, done, verify_release=False)
        head = done
    _require(not records, "unexpected publication records or rollback")
    if verify_release and head is not None:
        _validate_confirmation(store, run_sha, {k: head[k] for k in ("schema_version", "run_sha256", "step", "checkpoint_sha256", "previous_head_sha256")}, head)
    return {"head": deepcopy(head), "pending": deepcopy(pending), "committed_steps": len(commits)}


def _validate_confirmation(store, run_sha, intent, confirmation, *, verify_release=True):
    _require(set(confirmation) == {"schema_version", "run_sha256", "step", "checkpoint_sha256",
        "previous_head_sha256", "publication_intent_sha256", "model_release_sha256"}, "invalid publication confirmation")
    expected = {**intent, "publication_intent_sha256": content_sha256(intent),
                "model_release_sha256": confirmation["model_release_sha256"]}
    _require(canonical_bytes(expected) == canonical_bytes(confirmation), "publication confirmation differs")
    _require(isinstance(confirmation["model_release_sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", confirmation["model_release_sha256"]), "invalid publication release SHA")
    if not verify_release:
        return
    release, _ = store.load_release(confirmation["model_release_sha256"])
    _require(release["run_sha256"] == run_sha and release["source"]["checkpoint_sha256"] == intent["checkpoint_sha256"]
        and type(release["source"]["learner_step"]) is int and release["source"]["learner_step"] == intent["step"],
        "publication release belongs to a different run/checkpoint/step")


def read_publication_head(store, run_sha):
    """Read-only validated head; not a cross-process reservation transaction."""
    state = publication_state(store, run_sha)
    _require(state is not None, "publication head is not enabled")
    _require(state["pending"] is None, "unpublished/unknown commit blocks new dispatch")
    _require(state["head"] is not None, "no published checkpoint for this run; explicit bootstrap is separate")
    return state["head"]


def enable(learner):
    from .learner import _sync_directory
    with learner.lock:
        learner._check_live()
        _require(learner.parent_checkpoint is None and learner.sampler_state["step"] == 0,
                 "release head must be explicitly enabled before first training commit")
        directory = learner.control/DIRECTORY
        try:
            directory.mkdir(exist_ok=False); _sync_directory(learner.control)
            _write_record(directory, "mode.json", _mode(learner.run_sha, learner.run))
        except BaseException:
            learner.blocked = True
            raise
        return deepcopy(_mode(learner.run_sha, learner.run))


def publish(learner, checkpoint_sha256=None):
    from .learner import publish_checkpoint
    state = publication_state(learner.store, learner.run_sha, verify_release=False)
    _require(state is not None and state["pending"] is not None, "no unpublished committed head")
    pending = state["pending"]; intent = pending["expected_intent"]
    _require(pending["intent"] is None, "unknown publication must be reconciled, never retried")
    _require(checkpoint_sha256 is None or checkpoint_sha256 == intent["checkpoint_sha256"],
             "publication must use the latest committed checkpoint")
    directory = learner.control/DIRECTORY; step = intent["step"]
    _write_record(directory, f"intent-{step:08d}.json", intent)
    metadata = publish_checkpoint(learner.runtime, checkpoint_sha256=intent["checkpoint_sha256"],
        journal_root=learner.journal/("publish-"+intent["checkpoint_sha256"]), expected_run=learner.run)
    confirmation = {**intent, "publication_intent_sha256": content_sha256(intent),
        "model_release_sha256": metadata["model_release_sha256"]}
    _validate_confirmation(learner.store, learner.run_sha, intent, confirmation)
    _write_record(directory, f"published-{step:08d}.json", confirmation)
    learner._publication_head_cache = deepcopy(confirmation)
    return metadata


def reconcile_published(store, *, run_sha, model_release_sha256):
    """Explicit offline read-CAS/confirm only. Close the coordinator first.

    Never extracts tensors, publishes a release, or trains. The existing run
    writer lease excludes an active coordinator, including failed ones.
    """
    from .learner import _WriterLease
    _require(isinstance(run_sha, str) and re.fullmatch(r"[0-9a-f]{64}", run_sha), "invalid publication run SHA")
    lease = _WriterLease(store.root, run_sha)
    try:
        # A complete orphan confirmation stage may be recovered only when the
        # caller supplies the exact independently readable CAS release. Partial
        # stages and unrelated stages remain blocked and are never overwritten.
        release, _ = store.load_release(model_release_sha256)
        _require(release["run_sha256"] == run_sha, "reconcile release belongs to different run")
        step = release["source"]["learner_step"]
        _require(type(step) is int and step > 0, "invalid released step")
        from .learner import _read_control
        directory = store.root/"control"/run_sha/DIRECTORY
        intent = _read_control(directory/f"intent-{step:08d}.json")
        candidate = {**intent, "publication_intent_sha256": content_sha256(intent),
                     "model_release_sha256": model_release_sha256}
        _validate_confirmation(store, run_sha, intent, candidate)
        state = publication_state(store, run_sha, permitted_staging=(canonical_bytes(candidate),))
        _require(state is not None, "publication head is not enabled")
        if state["pending"] is None:
            _require(state["head"] is not None and state["head"]["model_release_sha256"] == model_release_sha256,
                     "reconcile pin is not the current published head")
            return state["head"]
        intent = state["pending"]["intent"]
        _require(intent is not None, "no publication intent; do not guess a release")
        confirmation = {**intent, "publication_intent_sha256": content_sha256(intent),
            "model_release_sha256": model_release_sha256}
        _validate_confirmation(store, run_sha, intent, confirmation)
        _write_record(store.root/"control"/run_sha/DIRECTORY, f"published-{intent['step']:08d}.json", confirmation)
        return confirmation
    finally:
        lease.close()


def available_head(learner):
    """Only the owning live coordinator; caller must hold its RLock.

    The release was fully CAS-validated at publish/restore. Disk control records
    are rechecked against that cached immutable receipt without decoding weights
    while reserving an attempt. A different writer cannot own this run's lease.
    """
    learner._check_live()
    state = publication_state(learner.store, learner.run_sha, verify_release=False)
    _require(state is not None and state["pending"] is None and state["head"] is not None,
             "no published checkpoint for this run; explicit bootstrap is separate")
    _require(canonical_bytes(state["head"]) == canonical_bytes(learner._publication_head_cache),
             "publication head differs from owning coordinator validation")
    return deepcopy(state["head"])
