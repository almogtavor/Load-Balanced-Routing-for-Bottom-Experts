import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import archived_h200_closure as archived
import recovery_provenance as recovery
import prospective_gate as prospective
import hybrid_driver
import hybrid_outer_driver as outer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OLD = REPO / 'runs/paper-followup-20261018'
BINDINGS = json.loads((HERE.parent / 'previous-h200-bindings-v1.json').read_text())
BASE = OLD / 'hybrid-causality/hybrid-v6'


def historical():
    return {key: json.loads(Path(row['local']).read_text()) for key, row in BINDINGS.items()}


def fresh_fixture():
    old = OLD / 'remote-capture/hybrid-h200-final/archive-captures/20261018T104936Z-b1c126517511/extracted-20261018T105257Z-c1b8618e9260/files'
    proof = json.loads((old / 'provenance/hybrid-recovery-owner-binding-v1.json').read_text())
    owner = {**recovery.OWNER, 'uid': 'a4ea601d-5e25-4ed2-8c75-6f5fd2b01111'}
    proof['replacement_pod'] = owner
    proof['previous_h200_closure'] = copy.deepcopy(archived.SUMMARY)
    allocation = {**owner, 'verified_live': True, **proof['replacement_hardware']}
    return owner, proof, allocation


class FreshOwnerTests(unittest.TestCase):
    def test_actual_predecessor_source_and_closed_partial_metadata(self):
        rows = historical()
        self.assertEqual(archived.verify_value(rows), archived.SUMMARY)
        for name, row in BINDINGS.items():
            self.assertEqual(hashlib.sha256(Path(row['local']).read_bytes()).hexdigest(),
                             archived.RECORDS[name][1])

    def test_incomplete_or_wrong_uid_deletion_cannot_prove_exit(self):
        for change in ('deleted', 'absence_verified', 'uid'):
            rows = historical()
            if change == 'uid':
                rows['release']['delete_options']['preconditions']['uid'] = 'wrong'
            else:
                rows['release'][change] = False
            with self.assertRaises(RuntimeError):
                archived.verify_value(rows)

    def test_partial_timing_cannot_be_promoted_or_pooled(self):
        rows = historical()
        rows['timed']['complete'] = True
        with self.assertRaises(RuntimeError):
            archived.verify_value(rows)
        rows = historical()
        rows['timed']['launches'][7]['complete'] = True
        with self.assertRaises(RuntimeError):
            archived.verify_value(rows)

    def test_marker_from_another_owner_or_archive_rejected(self):
        for field, value in [('archive_sha256', '0' * 64), ('raw_file_count', 47729)]:
            rows = historical()
            rows['marker'][field] = value
            with self.assertRaises(RuntimeError):
                archived.verify_value(rows)
        rows = historical()
        rows['marker']['source']['pod_uid'] = 'another-pod'
        with self.assertRaises(RuntimeError):
            archived.verify_value(rows)

    def test_fresh_owner_exact_matching_and_old_owner_rejected(self):
        owner, proof, allocation = fresh_fixture()
        with patch.object(recovery, 'OWNER', owner):
            result = recovery.verify_value(proof, allocation)
            self.assertEqual(result['replacement_pod'], owner)
            self.assertEqual(result['previous_h200_closure'], archived.SUMMARY)
            for field in ('uid', 'node', 'image', 'namespace'):
                wrong = copy.deepcopy(allocation)
                wrong[field] = 'different'
                with self.assertRaises(RuntimeError):
                    recovery.verify_value(proof, wrong)
        with patch.object(recovery, 'OWNER', archived.OLD):
            with self.assertRaises(RuntimeError):
                recovery.verify_value(proof, allocation)

    def test_placeholder_or_missing_historical_exclusion_fails_closed(self):
        owner, proof, allocation = fresh_fixture()
        with patch.object(recovery, 'OWNER', {**owner, 'uid': 'PENDING_FRESH_ALLOCATION'}):
            with self.assertRaises(RuntimeError):
                recovery.verify_value(proof, allocation)
        proof.pop('previous_h200_closure')
        with patch.object(recovery, 'OWNER', owner):
            with self.assertRaises(RuntimeError):
                recovery.verify_value(proof, allocation)

    def test_prospective_closed_admission_never_reads_predecessor_proc(self):
        raw = OLD / 'remote-capture/hybrid-h200-final/archive-captures/20261018T104936Z-b1c126517511/extracted-20261018T105257Z-c1b8618e9260/files'
        mapping = {name: raw / name.removeprefix('/work/results/') for name in prospective.FILES}
        mapping[prospective.ANALYSIS] = OLD / 'audit/stock-kv-numerics-raw-analysis-v1.json'
        mapping['/work/gap/hybrid/archived_h200_closure.py'] = HERE / 'archived_h200_closure.py'
        mapping.update({row['remote']: Path(row['local']) for row in BINDINGS.values()})
        with patch.object(archived, 'verify', return_value=archived.verify_value(historical())), \
             patch.object(prospective, 'Path', side_effect=lambda name: mapping[str(name)]), \
             patch.object(prospective, 'read', side_effect=lambda name: json.loads(mapping[str(name)].read_text())):
            def forbidden(pid):
                raise AssertionError('Predecessor PID consulted in fresh namespace')
            value = prospective.verify(SimpleNamespace(process_record=forbidden))
            self.assertEqual(value['files_sha256'], prospective.FILES)
            self.assertEqual(value['policy']['revision'], 6)

    def test_scientific_execution_sources_and_policies_byte_identical(self):
        for name in ('hybrid_driver.py', 'hybrid_client.py', 'hybrid_timing.py', 'hybrid_phase_probe.py',
                     'audit_hybrid.py', 'score_hybrid_gsm.py', 'native_accounting.py',
                     'native_runner_contract.py', 'native_request_id.py', 'audit_native_metrics.py', 'common_v2.py'):
            self.assertEqual((HERE / name).read_bytes(), (BASE / name).read_bytes(), name)
        old = ast.parse((BASE / 'prospective_gate.py').read_text())
        assignment = next(n for n in old.body if isinstance(n, ast.Assign)
                          and any(isinstance(x, ast.Name) and x.id == 'POLICY' for x in n.targets))
        self.assertEqual(ast.literal_eval(assignment.value), prospective.POLICY)
        plan = {'vllm': '/usr/local/bin/vllm', 'model': '/pinned-model', 'gpu_util': .88,
                'site_root': '/pristine-site', 'phase_probe': str(HERE / 'hybrid_phase_probe.py')}
        for role in ('prefill', 'decode'):
            environment = hybrid_driver.environment(plan, role, 'keep6p8')
            self.assertNotIn('PLACEMENT_PROBE', environment)
            self.assertFalse(any('PROFILER' in name for name in environment))
            self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '0,1,2,3' if role == 'prefill' else '4,5,6,7')

    def test_fresh_paths_and_unbound_outer_cannot_launch(self):
        self.assertEqual(str(outer.OUTPUT), '/work/results/gap-hybrid-v1')
        self.assertEqual(str(outer.FINAL), '/work/results/gap-hybrid-outer-driver-v1.json')
        self.assertNotEqual(str(outer.TOOLS), '/work/hybrid-tools/hybrid-v6')
        with self.assertRaisesRegex(RuntimeError, 'must be frozen'):
            outer.request_gate({}, 'unused', Path('unused'))


if __name__ == '__main__':
    unittest.main()
