"""Preserved predecessor closure; never inspect an archived PID locally.

The whole final archive and exact-UID deletion are completed historical
evidence. They are distinct from fresh allocation and activation proofs.
"""
import hashlib
import json
from pathlib import Path

OLD = {'namespace': 'default', 'pod': 'par-followup-hybrid-h200-20261018',
       'uid': '23dd50e5-3fd6-420f-84e1-07b54ee38ecd', 'node': 'g134dfa',
       'image': 'vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b'}
ROOT = '/work/results/provenance/'
RECORDS = {
    'marker': ('previous-h200-final-marker-v1.json', '0608cc7618fe5ded8fd26f64e05b6ea7a6177b39a1ce1db9c5c2c8d766c52b37'),
    'release': ('previous-h200-release-v1.json', '9e080dae8285941ec984e9668fdcb79cffd29ef62dfe1b265e0a9e3c175a9c20'),
    'freeze': ('previous-h200-final-freeze-v1.json', 'd09ec0b6ec8f67d35b58f4032f92fec77a27fc2281a7ff5f2bccaf862120aaf8'),
    'outer': ('previous-h200-outer-v6.json', '59411cd3a0e0b41f481af96aaa51b2072356e8628d55f2b59cddd31a179a802b'),
    'proof': ('previous-h200-proof-v6.json', '9ac14950b25f42fdeb2802d93383fb75790cd7328172c9a7d217ba8cc7eaac3c'),
    'timed': ('previous-h200-timed-v6.json', 'f1e10f3eff57899c56ffe10b4da5e9c18f0f14abc35c868c610b8f811cf3cc1a'),
}
FILES = {ROOT + name: digest for name, digest in RECORDS.values()}
ARCHIVE_SHA = '65ede7edb7fca550ba7af6a66aaa00e18f0e6a679bf855c0dbdc453df8c7c6a9'
MANIFEST_SHA = '01b0950f0b68c7eb87a9da25a28b1b040030f9f346a4756633d7de023575d338'
PLAN_SHA = '05e4d78070a2399ec34e640120adc5b93fe88c6b150ee72be3f21a0723dc9805'
CLOSURE = 'Preserved exact-UID release after final whole archive; no fresh-namespace lookup of predecessor PIDs'
SUMMARY = {'schema': 'previous-h200-closed-partial-v6-v1', 'owner': OLD,
           'archive_sha256': ARCHIVE_SHA, 'raw_manifest_sha256': MANIFEST_SHA,
           'previous_plan_sha256': PLAN_SHA, 'previous_complete': False,
           'previous_timed_complete_launches': 7, 'previous_quality_launched': False,
           'partial_effects_excluded': True, 'original_allocation_deleted': True,
           'closure_scope': CLOSURE, 'files_sha256': FILES}


def need(value, message):
    if not value:
        raise RuntimeError(message)


def verify_value(records):
    need(set(records) == set(RECORDS), 'Exact predecessor closure inventory required')
    marker, release, freeze = (records[k] for k in ('marker', 'release', 'freeze'))
    need(marker['final_verified'] is True and marker['writers_stopped_asserted'] is True
         and marker['archive_sha256'] == ARCHIVE_SHA and marker['raw_manifest_sha256'] == MANIFEST_SHA
         and marker['raw_file_count'] == 47730 and marker['raw_bytes'] == 1138626000
         and marker['archive_bytes'] == 220816629
         and marker['remote_verification']['ok'] is True
         and all(marker['source'][k] == OLD[k] for k in ('namespace', 'pod'))
         and marker['source']['pod_uid'] == OLD['uid'], 'Historical final whole archive differs')
    need(release['admitted'] is True and release['deleted'] is True and release['absence_verified'] is True
         and release['pod_uid'] == OLD['uid']
         and all(release['before'][k] == v for k, v in OLD.items())
         and release['delete_options']['preconditions'] == {'uid': OLD['uid']}
         and release['archive_marker_sha256'] == RECORDS['marker'][1]
         and release['archive_sha256'] == ARCHIVE_SHA
         and release['current_remote_verification']['ok'] is True
         and release['current_remote_verification']['raw_manifest_sha256'] == MANIFEST_SHA,
         'Historical exact-owned deletion/NotFound authority differs')
    need(freeze['complete'] is True and freeze['writers_stopped'] is True
         and freeze['pod_uid'] == OLD['uid']
         and all(freeze['allocation'][k] == v for k, v in OLD.items()),
         'Historical final source freeze differs')
    outer, proof, timed = (records[k] for k in ('outer', 'proof', 'timed'))
    need(outer['complete'] is False and outer['cleanup_complete'] is True and outer['cleanup_errors'] == []
         and outer['hybrid_plan_sha256'] == PLAN_SHA
         and outer['controller_process']['pid'] == outer['controller_process']['pgrp'] == 4441
         and str(outer['controller_process']['start_ticks']) == '1121406677'
         and outer['final_gpu_idle']['returncode'] == 0 and outer['final_gpu_idle']['stdout'].strip() == '',
         'Preserve original incomplete/closed V6 outer')
    need(proof['complete'] is True and proof['cleanup_complete'] is True
         and proof['plan_sha256'] == PLAN_SHA and len(proof['launches']) == 3
         and all(row['complete'] is True for row in proof['launches']),
         'Historical phase proof closure differs')
    need(timed['complete'] is False and timed['cleanup_complete'] is True
         and timed['plan_sha256'] == PLAN_SHA and len(timed['launches']) == 8
         and [row['complete'] for row in timed['launches']] == [True] * 7 + [False]
         and all(role['remaining_members'] == [] for role in
                 timed['emergency_closure'].values() if isinstance(role, dict) and 'remaining_members' in role),
         'Historical partial timing must remain incomplete and excluded')
    need(freeze['finished_unix'] < release['finished_unix'], 'Historical deletion precedes source freeze')
    return dict(SUMMARY)


def verify():
    records = {}
    for name, (filename, expected) in RECORDS.items():
        path = Path(ROOT + filename)
        need(path.is_file() and not path.is_symlink(), 'Missing/nonregular predecessor closure record')
        raw = path.read_bytes()
        need(hashlib.sha256(raw).hexdigest() == expected, 'Predecessor closure bytes changed')
        records[name] = json.loads(raw)
    return verify_value(records)
