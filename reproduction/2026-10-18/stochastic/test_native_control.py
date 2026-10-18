#!/usr/bin/env python3
"""Native pause/RPC transport fixtures, including first-core-only response."""
import hashlib
import http.client
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import urllib.error

import task_adapter as adapter


class Response:
    status=200
    headers={"Content-Type":"application/json"}
    def __init__(self,value):self.value=value
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def read(self):return json.dumps(self.value,sort_keys=True).encode()


class ControlTests(unittest.TestCase):
    def fixture(self,root,attack=None):
        output=root/'stage';output.mkdir();ledgers=root/'ledgers';ledgers.mkdir()
        plan={'evidence_directory':str(ledgers),'parameters':{'seed':0},'private_site':str(root/'private'),
              'private_source_sha256':{'stochastic_tail/worker_extension.py':'ext'}}
        workers=[{'pid':100+i,'ep_rank':i,'actual_runner':{'native':'V2'}} for i in range(4)]
        paused=False;paths=[]
        def urlopen(request,timeout):
            nonlocal paused
            paths.append(request.full_url)
            if '/pause?' in request.full_url:
                paused=True;return Response({'status':'invalid' if attack=='bad_pause' else 'paused'})
            if request.full_url.endswith('/is_paused'):return Response({'is_paused':paused})
            if request.full_url.endswith('/resume'):
                paused=False;return Response({'status':'resumed'})
            self.assertTrue(request.full_url.endswith('/collective_rpc'));self.assertTrue(paused)
            if attack=='http500':raise urllib.error.HTTPError(request.full_url,500,'RPCfailed',{},io.BytesIO(b'{"error":"failed"}'))
            body=json.loads(request.data);payload=json.loads(body['args'][0]);records=[]
            for worker in workers:
                if attack=='missing_rank' and worker['ep_rank']==3:continue
                row={'kind':'native_checkpoint','complete':True,'reason':'collective_rpc_after_native_pause','unix_ns':time.time_ns(),
                     'pid':worker['pid'],'ep_rank':worker['ep_rank'],'parameters':plan['parameters'],'plan_sha256':'plan',
                     'phase':payload['phase'],'nonce':payload['nonce'],'sealed':payload['sealed'],'active_step':None,
                     'counters':{str(i):7 for i in range(48)},'actual_runner':worker['actual_runner'],'cuda_synchronized':True,
                     'checkpoint_callable':{'module':'stochastic_tail.worker_extension','code_filename':str(root/'private/stochastic_tail/worker_extension.py'),
                                            'source_sha256':'ext'}}
                if attack=='wrong_nonce' and worker['ep_rank']==3:row['nonce']='wrong'
                raw=json.dumps(row,sort_keys=True,separators=(',',':')).encode()+b'\n'
                path=ledgers/f"pid{worker['pid']}.jsonl";path.write_bytes(raw)
                records.append({'row':row,'record_sha256':hashlib.sha256(raw).hexdigest(),'ledger_path':str(path)})
            return Response({'results':records[:1]})
        return output,plan,workers,urlopen,paths

    def test_first_core_rpc_response_still_requires_all4_raw_nonce_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            output,plan,workers,urlopen,paths=self.fixture(Path(folder))
            with patch.object(adapter.urllib.request,'urlopen',urlopen):result=adapter.native_checkpoint('http://127.0.0.1:8077',output,plan,'plan','proof',workers)
            self.assertTrue(result['sealed']);receipt=json.loads(Path(result['path']).read_text())
            self.assertEqual(len(receipt['snapshots']),4)
            self.assertEqual([r['name'] for r in receipt['transcript']],['pause','is-paused','checkpoint'])
            rpc=receipt['transcript'][-1]
            self.assertTrue(all(rpc['submitted_unix_ns']<=r['row']['unix_ns']<=rpc['returned_unix_ns'] for r in receipt['snapshots']))
            for record in receipt['transcript']:
                self.assertEqual(adapter.sha(record['body_path']),record['body_sha256'])
                self.assertEqual(adapter.sha(record['response_path']),record['response_sha256'])

    def test_unsealed_training_resume_is_explicit_and_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            output,plan,workers,urlopen,paths=self.fixture(Path(folder))
            with patch.object(adapter.urllib.request,'urlopen',urlopen):result=adapter.native_checkpoint('http://127.0.0.1:8077',output,plan,'plan','preflight',workers)
            self.assertFalse(result['sealed']);receipt=json.loads(Path(result['path']).read_text())
            self.assertEqual([r['name'] for r in receipt['transcript']],['pause','is-paused','checkpoint','resume','is-resumed'])
            self.assertEqual(json.loads(Path(receipt['transcript'][-1]['response_path']).read_text()),{'is_paused':False})

    def test_missing_or_wrong_nonce_rank_and_failed_pause_fail_closed(self):
        for attack in ('missing_rank','wrong_nonce','bad_pause'):
            with self.subTest(attack=attack),tempfile.TemporaryDirectory() as folder:
                output,plan,workers,urlopen,paths=self.fixture(Path(folder),attack)
                with patch.object(adapter.urllib.request,'urlopen',urlopen),self.assertRaises(RuntimeError):
                    adapter.native_checkpoint('http://127.0.0.1:8077',output,plan,'plan','proof',workers)
                self.assertFalse((output/'native-control/receipt.json').exists())
                self.assertTrue((output/'native-control/pause.response.json').exists())
                if attack=='bad_pause':self.assertEqual(len(paths),1)

    def test_http_error_keeps_exact_response_and_status_timestamps(self):
        with tempfile.TemporaryDirectory() as folder:
            output,plan,workers,urlopen,paths=self.fixture(Path(folder),'http500')
            with patch.object(adapter.urllib.request,'urlopen',urlopen),self.assertRaises(urllib.error.HTTPError):
                adapter.native_checkpoint('http://127.0.0.1:8077',output,plan,'plan','proof',workers)
            self.assertEqual((output/'native-control/checkpoint.response.json').read_bytes(),b'{"error":"failed"}')
            transport=json.loads((output/'native-control/checkpoint.transport.json').read_text())
            self.assertEqual(transport['status'],500);self.assertLessEqual(transport['submitted_unix_ns'],transport['returned_unix_ns'])

    def test_incomplete_and_connection_failure_preserve_response_artifacts(self):
        class Broken(Response):
            def read(self):raise http.client.IncompleteRead(b'{"stat',10)
        for kind in ('partial','connect'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as folder:
                output,plan,workers,_,_=self.fixture(Path(folder))
                def failed(*args,**kwargs):
                    if kind=='connect':raise OSError('connection closed')
                    return Broken(None)
                with patch.object(adapter.urllib.request,'urlopen',failed),self.assertRaises((OSError,http.client.IncompleteRead)):
                    adapter.native_checkpoint('http://127.0.0.1:8077',output,plan,'plan','proof',workers)
                prefix=output/'native-control'
                self.assertEqual((prefix/'pause.response.json').read_bytes(),b'{"stat' if kind=='partial' else b'')
                meta=json.loads((prefix/'pause.transport.json').read_text())
                self.assertFalse(meta['response_complete']);self.assertIsNotNone(meta['error'])
                self.assertEqual(meta['status'],200 if kind=='partial' else None)


if __name__=='__main__':unittest.main(verbosity=2)
