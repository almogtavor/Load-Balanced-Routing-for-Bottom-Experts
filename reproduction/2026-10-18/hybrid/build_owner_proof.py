"""Create fresh archival lineage after exact subset and live-owner preflight.

CPU/files only. Full original archives are hash-verified locally; the original
281-file subset is copied to the fresh pod and rehashed here. The frozen plan
separately rehashes every required recovered dependency.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import recovery_provenance as recovery
import archived_h200_closure as archived

PRIOR_SHA = 'a25b1dec8131dcd3d5e26370d460a8b4574efd7487223095ec6b1a5562be4e18'


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
         'Only regular non-symlink inputs accepted')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def build(args):
    need(recovery.OWNER['uid'] != 'PENDING_FRESH_ALLOCATION', 'Fresh exact source owner not frozen')
    need(sha(args.prior_proof) == PRIOR_SHA, 'Prior H100 recovery receipt differs')
    prior = read(args.prior_proof)
    allocation, deployment = read(args.allocation_proof), read(args.source_proof)
    need(all(allocation[k] == v for k, v in recovery.OWNER.items())
         and allocation['verified_live'] is True and allocation['gpu_count'] == 8
         and 'H200' in allocation['gpu_name']
         and all(deployment['pod_identity'][k] == v for k, v in recovery.OWNER.items()),
         'Fresh allocation/deployment owner mismatch')
    need(sha(args.stage_receipt) == args.stage_receipt_sha256, 'Frozen actual subset stage receipt changed')
    stage = read(args.stage_receipt)
    need(all(stage[point][key] == value for point in ('before', 'after') for key, value in recovery.OWNER.items()),
         'Actual subset copy owner differs')
    need(len(stage['files']) == 281 and stage['payload']['payload_bytes'] == 71977220
         and stage['payload']['payload_sha256'] == 'c97685d04ff41d343e09f0eff393ce513a18025a41853c3da40ae9ae3f13afcd',
         'Original exact281-file payload changed')
    need(sha(args.remote_stage_receipt) == stage['result']['receipt_sha256'], 'Actual remote subset receipt changed')
    remote_stage = read(args.remote_stage_receipt)
    need(remote_stage['complete'] is True and remote_stage['pod_uid'] == recovery.OWNER['uid']
         and remote_stage['files'] == stage['files'], 'Remote actual subset receipt differs')
    for name, expected in stage['files'].items():
        path = Path('/work') / name
        need(path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'],
             'Actual staged dependency changed: ' + name)
    archives = {}
    for name in ('source', 'results'):
        expected = recovery.ARCHIVES[name]
        actual = stage['local_full_archive_lineage'][name]
        need(all(actual[key] == value for key, value in expected.items())
             and actual['verified_locally'] is True and actual['remote_full_archive_copied'] is False,
             'Original full archive local SHA lineage differs')
        archives[name + '_archive'] = {**prior[name + '_archive'], 'verified': True,
            'verification_scope': 'Full original archive SHA verified locally; original281-file subset actually copied and rehashed on fresh owner',
            'actual_subset_receipt_sha256': stage['result']['receipt_sha256']}
    for root in (recovery.RESULTS, recovery.SOURCE):
        path = Path(root)
        need(path.is_dir() and not any(p.is_symlink() for p in (path, *path.parents)),
             'Actual recovered archive root missing or aliased')
    required = {
        '/work/recovered-primary-results/prefill-e2e-v1/timed/receipt.json': '355bf94ffd88aad97675559bcfd1726d671228fa80dfe3a06a67c913941ad458',
        '/work/recovered-primary-results/prefill-profiles-v4/receipt.json': '2b3c6aa97dd4c102998e8d7fc1684b89c3a0fd958564c2861189a5bdc4299ace',
        '/work/recovered-primary-source/followup/performance/prefill_e2e.py': '2b1832a73ec1d84ef16746c99a8ad94c82f4300ef7464fc4d2250e63b66b6db3',
    }
    for path, expected in required.items():
        need(sha(path) == expected, 'Recovered required source/evidence differs: ' + path)
    inputs = {'allocation-proof.json': args.allocation_proof,
              'provenance/task-source-deployment.json': args.source_proof,
              'checkpoint-proof.json': args.checkpoint_proof,
              'hybrid-runtime-preflight-v1.json': args.runtime_preflight}
    fresh_files = {name: sha(path) for name, path in inputs.items()}
    runtime = read(args.runtime_preflight)
    need(runtime['complete'] is True and runtime['packages']['vllm'].split('+')[0] == '0.31.0'
         and runtime['packages']['nixl'] == '1.4.1' and runtime['nixl_agent_constructed'] is True,
         'Actual native NIXL runtime not ready')
    proof = {'schema': 'hybrid-primary-owner-recovery-v1', 'complete': True,
             'created_unix_ns': time.time_ns(), 'original_pod': recovery.OLD,
             'recovered_results_root': recovery.RESULTS, 'recovered_source_root': recovery.SOURCE,
             'replacement_pod': dict(recovery.OWNER),
             'replacement_hardware': {'gpu_count': 8, 'gpu_name': allocation['gpu_name']},
             'old_pod_not_found': prior['old_pod_not_found'],
             'historical_h100_recovery_receipt_sha256': PRIOR_SHA,
             'previous_h200_closure': archived.verify(),
             'fresh_current_files_sha256': fresh_files,
             'builder_sha256': sha(__file__),
             'root_recovery_receipt_sha256': args.stage_receipt_sha256,
             'runtime_recovery_assets_sha256': stage['result']['receipt_sha256'],
             'scientific_scope': 'Fresh complete3proof+9timing+3quality H200 campaign; stock handoff baseline, original partial V6 excluded; no old PID lookup or monolithic equivalence claim.',
             **archives}
    recovery.verify_value(proof, allocation)
    output = Path(args.output)
    need(not output.exists() and not output.is_symlink(), 'Fresh immutable owner proof already exists')
    with output.open('x') as stream:
        json.dump(proof, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return {'complete': True, 'proof_path': str(output.resolve()), 'sha256': sha(output),
            'fresh_owner': recovery.OWNER}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('prior-proof', 'allocation-proof', 'source-proof', 'checkpoint-proof',
                 'runtime-preflight', 'stage-receipt', 'remote-stage-receipt', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--stage-receipt-sha256', required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))
