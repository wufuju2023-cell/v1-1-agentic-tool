"""Explicit same-session terminal CE/discounted-value objective, never visits.

The immutable dataset contains the student's original generated actions and a
complete independent Lean state replay. Labels cannot be supplied by the event.
This capability is absent from the default real-search initialization contract.
"""
from __future__ import annotations

import json
from pathlib import Path

from .identifiers import validate_identifier
from .search_objective import distance_to_value, nonnegative_integer
from .verified_objective import SHA256

OBJECTIVE_KIND = 'verified_success_discounted_v1'
MAX_ROWS = 32
CONTRACT = {
    'kind': OBJECTIVE_KIND,
    'profile': 'verified-generated-action-negative-longest-branch-v1',
    'policy': 'uniform_action_joint_token_ce_no_implicit_eos',
    'value': 'gamma_power_remaining_generated_distance_minus_one_with_search_floor',
    'source': 'same_session_independent_full_state_replay',
    'max_rows': MAX_ROWS,
    'updates': 'one_full_trajectory_once_after_search_before_seal',
}


def prepare_success_event(event, *, session_id, policy_version, dataset_root, gamma, value_floor,
                          dataset_directory=None):
    from cpu_runtime.verified_trajectory import load_verified_dataset
    from cpu_runtime.verified_dataset_store import read_bundle
    fields = {'kind', 'event_id', 'session_id', 'tree_id', 'theorem_id', 'policy_version',
              'dataset_sha256', 'course_acceptance_sha256'}
    if type(event) is not dict or set(event) != fields or event['kind'] != OBJECTIVE_KIND:
        raise ValueError('successful finalization accepts only exact dataset/source references')
    if dataset_root is None:
        raise ValueError('successful finalization capability is disabled')
    if event['session_id'] != session_id:
        raise ValueError('successful finalization session mismatch')
    for key in ('event_id', 'session_id', 'tree_id'):
        validate_identifier(event[key], kind=key)
    for key in ('dataset_sha256', 'course_acceptance_sha256', 'theorem_id'):
        if not isinstance(event[key], str) or SHA256.fullmatch(event[key]) is None:
            raise ValueError('successful finalization requires exact SHA256: ' + key)
    if nonnegative_integer(event['policy_version'], 'policy_version') != policy_version:
        raise ValueError('successful finalization policy version mismatch')
    # dataset_directory is only for CPU import admission before installation.
    # The backend never forwards an event-selected path; it always uses its
    # fixed trusted root and the content digest.
    directory = Path(dataset_directory) if dataset_directory is not None else Path(dataset_root) / event['dataset_sha256']
    dataset = load_verified_dataset(directory, expected_sha256=event['dataset_sha256'])
    # The strict loader pins this envelope as a frozen replay input. It does not
    # accept a naked caller-supplied "verified" flag or a supplied target label.
    import hashlib
    historical_raw = read_bundle(directory)['historical-proof-receipt.json']
    if hashlib.sha256(historical_raw).hexdigest() != dataset['inputs_sha256']['historical-proof-receipt.json']:
        raise ValueError('successful replay envelope changed during admission')
    historical = json.loads(historical_raw)
    binding = historical.get('course_acceptance')
    if (not isinstance(binding, dict)
            or binding.get('accepted_sha256') != event['course_acceptance_sha256']
            or binding.get('schema_version') != 'reap.course-proof-replay-binding.v1'):
        raise ValueError('successful replay missing original course acceptance pin')
    accepted = binding.get('accepted')
    raw = binding.get('accepted_utf8')
    if (not isinstance(raw, str) or hashlib.sha256(raw.encode()).hexdigest() != event['course_acceptance_sha256']
            or json.loads(raw) != accepted or not isinstance(accepted, dict)
            or accepted.get('session_id') != session_id
            or accepted.get('theorem_sha256') != event['theorem_id']
            or accepted.get('passed') is not True):
        raise ValueError('successful replay original accepted bytes/identity mismatch')
    if (dataset['session_id'] != session_id or dataset['tree_id'] != event['tree_id']
            or dataset['theorem_sha256'] != event['theorem_id']
            or type(dataset['final_policy_version']) is not int
            or dataset['final_policy_version'] != policy_version):
        raise ValueError('successful replay is not the current session/search version')
    rows = dataset['rows']
    if not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError('successful finalization requires 1..32 complete action rows; never truncate')
    samples = []
    for index, row in enumerate(rows):
        distance = -row['return']
        if type(row['return']) is not int or distance < 1:
            raise ValueError('invalid successful generated-action distance')
        behavior = nonnegative_integer(row['policy_version'], 'source policy version')
        if behavior > policy_version:
            raise ValueError('successful action behavior version is from the future')
        samples.append({**row, 'row': index, 'value_target': distance_to_value(distance, gamma, value_floor=value_floor),
                        'target_probability': 1.0 / len(rows)})
    return {'event': dict(event), 'samples': samples, 'dataset': dataset,
            'sample_weight': 1.0 / len(rows)}
