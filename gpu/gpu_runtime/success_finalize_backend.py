"""One terminal update; GpuRuntime owns version commit and rollback/quarantine."""
from __future__ import annotations

import hashlib
from time import perf_counter

from .backend import BackendLearnResult
from .search_objective import finite_number
from .success_finalize_objective import OBJECTIVE_KIND, prepare_success_event


def learn_success(backend, session_id, event):
    session = backend._activate(session_id)
    example = prepare_success_event(event, session_id=session_id, policy_version=session.optimizer_steps,
        dataset_root=backend.success_dataset_root, gamma=backend.gamma, value_floor=backend.value_floor)
    torch = backend.torch
    tokenized = []
    # Every row is checked before zero_grad or any optimizer mutation. No token,
    # action or branch truncation is allowed to make a trajectory fit.
    for row in example['samples']:
        prompt = backend._tokenize_prompt(row['prompt'])
        target = backend.tokenizer(row['tactic'], return_tensors='pt', add_special_tokens=False)['input_ids'].to(backend.device)
        if (prompt['input_ids'].shape[0] != 1 or prompt['input_ids'].shape[1] < 1
                or target.shape[0] != 1 or target.shape[1] < 1
                or prompt['input_ids'].shape[1] + target.shape[1] > backend.max_sequence_tokens):
            raise ValueError('successful training sequence exceeds limit or is empty; never truncate')
        tokenized.append((row, prompt, target))
    parameters = backend._assert_optimizer_scope(session_id, session)
    before = backend.session_fingerprints(session_id)
    old_cache = backend.model.config.use_cache
    session.optimizer.zero_grad(set_to_none=True)
    policy_total = value_total = kl_total = 0.0
    trace = []
    guard_detail = None
    try:
        with backend._session_rng(session):
            backend.model.config.use_cache = False
            backend.model.eval()
            session.value_head.train()
            for row, prompt, target_ids in tokenized:
                weight = example['sample_weight']
                ids = torch.cat((prompt['input_ids'], target_ids), dim=1)
                mask = torch.cat((prompt.get('attention_mask', torch.ones_like(prompt['input_ids'])),
                                  torch.ones_like(target_ids)), dim=1)
                length = target_ids.shape[1]
                output = backend.model(input_ids=ids, attention_mask=mask, use_cache=False,
                    logits_to_keep=length + 1, return_dict=True)
                logits = output.logits[:, -length - 1:-1, :].float()
                if logits.shape[1] != length:
                    raise RuntimeError('successful policy scoring positions mismatch')
                backend._require_finite(logits, 'successful policy logits')
                logs = backend.functional.log_softmax(logits, dim=-1)
                nll = -logs.gather(-1, target_ids.unsqueeze(-1)).sum()
                with torch.no_grad(), backend.model.disable_adapter():
                    reference = backend.model(input_ids=ids, attention_mask=mask, use_cache=False,
                        logits_to_keep=length + 1, return_dict=True)
                    reference_logits = reference.logits[:, -length - 1:-1, :].float()
                    if reference_logits.shape != logits.shape:
                        raise RuntimeError('successful reference scoring positions mismatch')
                    backend._require_finite(reference_logits, 'successful reference logits')
                    reference_logs = backend.functional.log_softmax(reference_logits, dim=-1)
                kl = (logs.exp() * (logs - reference_logs)).sum()
                loss = weight * (nll + backend.kl_beta * kl)
                backend._require_finite({'nll': nll, 'kl': kl, 'loss': loss}, 'successful policy loss')
                loss.backward()
                policy_total += weight * float(nll.detach().cpu())
                kl_total += weight * float(kl.detach().cpu())
                del output, logits, logs, nll, reference, reference_logits, reference_logs, kl, loss
                output = backend.model(**prompt, output_hidden_states=True, logits_to_keep=1,
                    use_cache=False, return_dict=True)
                prediction, target, loss, value_training = backend._success_value_loss(
                    session, output.hidden_states[-1][:, -1, :].float(), row)
                backend._require_finite({'prediction': prediction, 'target': target, 'loss': loss}, 'successful value')
                (weight * backend.value_coefficient * loss).backward()
                value_total += weight * float(loss.detach().cpu())
                sample_trace = {'row': row['row'], 'node_index': row['node_index'],
                    'generation_sequence': row['generation_sequence'], 'eval_sequence': row['eval_sequence'],
                    'source_policy_version': row['policy_version'], 'return': row['return'],
                    'value_target': value_training['target'],
                    'target_tokens': length,
                    'prompt_sha256': hashlib.sha256(row['prompt'].encode()).hexdigest(),
                    'tactic_sha256': hashlib.sha256(row['tactic'].encode()).hexdigest()}
                if getattr(backend, '_categorical_value_training', False):
                    sample_trace['value_training'] = value_training
                trace.append(sample_trace)
                del output, prediction, target, loss
            total = policy_total + backend.kl_beta * kl_total + backend.value_coefficient * value_total
            backend._require_finite(total, 'successful total loss')
            backend._require_finite([p.grad for p in parameters if p.grad is not None], 'successful gradients')
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, backend.max_grad_norm, error_if_nonfinite=True)
            session.optimizer.step()
            backend._require_finite(parameters, 'successful updated parameters')
            backend._require_finite(session.optimizer.state_dict(), 'successful optimizer state')
            if backend.max_post_update_kl is not None:
                started = perf_counter()
                post_kl = sum(example['sample_weight'] * backend._measure_post_update_kl(
                    prompt, [target_ids], [{'target_probability': 1.0}]) for _, prompt, target_ids in tokenized)
                post_kl = finite_number(post_kl, 'successful post-update KL')
                if post_kl < 0:
                    raise FloatingPointError('negative successful post-update KL; no clamping')
                guard_detail = {'maximum': backend.max_post_update_kl, 'post_update_kl': post_kl,
                    'reduction': 'uniform_action_sum_of_prefix_next_token_KL_current_to_frozen_base',
                    'timing': 'after_optimizer_step_before_commit', 'scope': 'verified_success_rows_only',
                    'measurement_seconds': perf_counter() - started, 'accepted': post_kl <= backend.max_post_update_kl}
                if not guard_detail['accepted']:
                    from .search_backend import KLGuardExceeded
                    raise KLGuardExceeded(guard_detail)
    finally:
        backend.model.config.use_cache = old_cache
        backend.model.eval()
        session.value_head.eval()
        session.optimizer.zero_grad(set_to_none=True)
    session.optimizer_steps += 1
    session.examples_seen += len(trace)
    after = backend.session_fingerprints(session_id)
    detail = {'objective': OBJECTIVE_KIND, 'loss': total, 'policy_loss': policy_total, 'kl': kl_total,
        'value_loss': value_total, 'grad_norm': float(grad_norm.detach().cpu()),
        'optimizer_steps': session.optimizer_steps, 'finite_loss': True, 'finite_gradients': True,
        'finite_parameters': True, 'finite_optimizer_state': True, 'base_parameters_frozen': True,
        'base_content_hash_checked': False, 'parameter_diffs': backend._fingerprint_changes(before, after),
        'training_config': backend._search_config(), 'source': dict(event), 'samples': trace,
        'online_update_consumed_by_later_generation': False}
    if guard_detail is not None:
        detail['kl_guard'] = guard_detail
    return BackendLearnResult(
        adapter_metadata={'kind': 'lora', 'rank': backend.lora_config.r,
            'objective': 'search_visit_backup', 'examples_seen': session.examples_seen},
        value_metadata=backend._value_metadata(last_loss=value_total),
        optimizer_metadata={'kind': 'AdamW', 'steps': session.optimizer_steps}, detail=detail)
