#!/usr/bin/env python3
"""Bounded remote causal sequencer; local source requires root review first.

Waits for the exact nine-quality campaign and original controller closure before
any child or GPU check. Then invokes only the reviewed plan/training/freeze/
heldout interfaces. Failed stages and owned cleanup are retained without retry.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

STOP_UNIX = 1792336755
TOOLS = Path('/work/hybrid-tools/causal-v2')
OUTPUT = Path('/work/results/causal-v2')
OUTER = Path('/work/results/causal-outer-v1')
QUALITY = Path('/work/results/quality-repeats-v1/driver-status.json')
QUALITY_PLAN = Path('/work/results/provenance/quality-repeat-plan-v1.json')
QUALITY_LAUNCH = Path('/work/results/provenance/quality-controller-launch-v1.json')
LIFECYCLE = Path('/work/followup/performance/prefill_e2e.py')
LIFECYCLE_SHA = '2b1832a73ec1d84ef16746c99a8ad94c82f4300ef7464fc4d2250e63b66b6db3'
HELPER_SHA = 'ff27eca971679a8bf24bbee01505ab1f96d74a9855a9fed07858eb266bbec1df'
MODEL = '/work/hf/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe'
UID = 'c42b0a8c-eb60-4837-9bd9-e83e4fceaa3c'
ARMS = ('stock','keep6p8','keep4p8')


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def load_lifecycle():
    need(sha(LIFECYCLE)==LIFECYCLE_SHA,'Frozen lifecycle helper changed')
    spec=importlib.util.spec_from_file_location('causal_outer_frozen_lifecycle',LIFECYCLE)
    value=importlib.util.module_from_spec(spec)
    previous=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    try:spec.loader.exec_module(value)
    finally:sys.dont_write_bytecode=previous
    return value


def plan_command():
    return ['python3','-u',str(TOOLS/'causal_driver_v1.py'),'plan','--output',str(OUTPUT),'--model',MODEL,
        '--production-root','/work/task-router','--site-root','/work/task-site',
        '--source-proof','/work/results/provenance/task-source-deployment.json',
        '--checkpoint-proof','/work/results/checkpoint-proof.json','--allocation-proof','/work/results/allocation-proof.json',
        '--lifecycle-helper',str(LIFECYCLE),'--vllm','/usr/local/bin/vllm',
        '--preflight-inputs','/work/results/quality-prepared-v1/preflight.json',
        '--full-inputs','/work/results/quality-prepared-v1/requests.json',
        '--quality-receipt',str(QUALITY),'--heldout-helper',str(TOOLS/'heldout_causal_v2.py'),
        '--heldout-review',str(TOOLS/'heldout-causal-v2-independent-review-v1.json'),'--stop-unix',str(STOP_UNIX)]


def stage_commands():
    helper=['python3','-u',str(TOOLS/'causal_driver_v1.py')]
    return [('plan',plan_command()),('training',helper+['run','--plan',str(OUTPUT/'plan.json'),'--mode','training']),
            ('freeze-heldout',helper+['freeze-heldout','--plan',str(OUTPUT/'plan.json')]),
            ('heldout',helper+['run','--plan',str(OUTPUT/'plan.json'),'--mode','heldout'])]


def request_gate(request, expected_sha, path):
    need(sha(path)==expected_sha and request['schema']=='bounded-causal-outer-request-v1','Outer request changed')
    need(request['stop_unix']==STOP_UNIX and request['tools']==str(TOOLS) and request['output']==str(OUTPUT)
         and request['outer_output']==str(OUTER) and request['quality_receipt']==str(QUALITY)
         and request['quality_controller']=={'pid':167,'start_ticks':'5555001'}
         and request['child_cleanup_grace_seconds']==720 and request['poll_seconds']==10,'Outer scope/deadline changed')
    need(request['driver_sha256']==sha(__file__) and request['commands']==[[k,v] for k,v in stage_commands()],
         'Outer source/commands changed')
    for filename,expected in request['files'].items():
        need(sha(filename)==expected,'Frozen outer prerequisite changed: '+filename)
    need(request['files'][str(TOOLS/'causal_driver_v1.py')]==HELPER_SHA
         and request['files'][str(LIFECYCLE)]==LIFECYCLE_SHA,'Wrong approved controller/lifecycle source')
    allocation=read('/work/results/allocation-proof.json')
    need(allocation.get('verified_live') is True and allocation['namespace']=='itay' and allocation['uid']==UID
         and allocation['pod']=='par-followup-quality-h100-20261018'
         and allocation['node']=='llmd-roce-mnbt6-worker-h100-2-gdr-qvrws'
         and allocation['gpu_count']==8 and 'H100' in allocation['gpu_name'],'Wrong secondary allocation proof')
    proof=read('/work/results/provenance/task-source-deployment.json')['pod_identity']
    need(all(proof[k]==allocation[k] for k in ('namespace','pod','uid','node','image')),'Source/allocation ownership differs')
    launch=read(QUALITY_LAUNCH)
    need(launch['pid']==167 and str(launch['start_ticks'])=='5555001' and launch['pgrp']==167 and launch['pod_uid']==UID
         and launch['plan_sha256']==sha(QUALITY_PLAN),'Wrong original quality controller launch proof')


def upstream_ready(path, lib):
    """False only while the correctly identified upstream controller is running."""
    value=read(path)
    need(value.get('pid')==167 and value.get('plan_sha256')==sha(QUALITY_PLAN),'Upstream status belongs to another campaign')
    if value.get('error') or value.get('cleanup_errors'):
        raise RuntimeError('Upstream quality failed; no causal GPU work permitted')
    live=lib.process_record(167)
    exited=not live or str(live['start_ticks'])!='5555001' or live['state']=='Z'
    if not value.get('finished_unix'):
        need(not exited,'Original quality controller exited without terminal closure proof')
        return False
    need(value.get('complete') is True and value.get('execution_complete') is True
         and value.get('cleanup_complete') is True and value.get('cleanup_errors')==[],
         'Upstream quality ended incomplete; no causal GPU work permitted')
    rows=value['launches']
    expected={(r,a) for r in (1,2,3) for a in ARMS}
    need(len(rows)==9 and {(r['replicate'],r['arm']) for r in rows}==expected
         and all(r['complete'] is True and r['ep']==4 for r in rows),'Upstream nine-cell matrix incomplete')
    return exited


def cleanup_owned(child, identity, lib, *, grace=720):
    need(identity and identity['pid']==identity['pgrp']==child.pid,'Cleanup needs owned child group')
    current=lib.process_record(child.pid)
    need(not current or str(current['start_ticks'])==str(identity['start_ticks']),'Owned child PID reused')
    result={'identity':identity,'signals':[],'started_unix':time.time(),'grace_seconds':grace}
    for signum,duration in ((signal.SIGTERM,grace),(signal.SIGKILL,15)):
        if not lib.group_members(child.pid):break
        try:os.killpg(child.pid,signum);result['signals'].append(signum.name)
        except ProcessLookupError:break
        deadline=time.monotonic()+duration
        while lib.group_members(child.pid) and time.monotonic()<deadline:
            child.poll();time.sleep(.25)
    try:child.wait(timeout=2)
    except subprocess.TimeoutExpired:pass
    result.update(remaining_members=lib.group_members(child.pid),returncode=child.poll(),finished_unix=time.time())
    need(not result['remaining_members'],'Owned causal child group remains')
    return result


def mode_receipt(path, mode, plan_sha, expected_count):
    value=read(path)
    need(value.get('complete') is True and value.get('cleanup_complete') is True
         and value['plan_sha256']==plan_sha and value['mode']==mode and len(value['launches'])==expected_count,
         'Causal stage receipt incomplete or drifted: '+mode)
    expected={(a,b) for b in ((16384,1024) if mode=='training' else (16384,)) for a in ARMS}
    need({(r['arm'],r['budget']) for r in value['launches']}==expected,'Causal stage matrix differs')
    for row in value['launches']:
        need(row['complete'] is True and sha(row['receipt'])==row['sha256'],'Causal failed/changed launch')
        launch=read(row['receipt'])
        need(launch['complete'] is True and launch['workload_complete'] is True and not launch['finalization_errors']
             and launch['arm']==row['arm'] and launch['budget']==row['budget'] and launch['plan_sha256']==plan_sha
             and launch['closure']['causal']['remaining_members']==[],'Causal owned launch not closed')
    return {'path':str(path),'sha256':sha(path),'launches':expected_count}


class Outer:
    def __init__(self,path,expected_sha,lib):
        self.path,self.expected_sha,self.lib=path,expected_sha,lib
        self.request,self.initial_error=None,None
        OUTER.mkdir(exist_ok=False)
        self.lock=(OUTER/'outer.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.status=OUTER/'status.json'
        self.child=self.child_identity=None
        self.upstream_qualified=False
        self.state={'complete':False,'cleanup_complete':False,'request_sha256':expected_sha,'driver_sha256':sha(__file__),
                    'controller_process':lib.process_record(os.getpid()),'started_unix':time.time(),'stages':[],
                    'stage':'waiting for nine quality launches and original controller exit'}
        self.save()
        try:self.request=read(path)
        except BaseException:self.initial_error=traceback.format_exc()

    def save(self):
        self.state['updated_unix']=time.time();self.lib.save(self.status,self.state)

    def validate(self):
        need(time.time()<STOP_UNIX,'Absolute causal compute cutoff reached')
        request_gate(self.request,self.expected_sha,self.path)
        if self.upstream_qualified:
            need(sha(QUALITY)==self.state['upstream_quality_receipt_sha256'] and upstream_ready(QUALITY,self.lib),
                 'Closed upstream receipt changed or original quality controller revived')

    def stage_receipt(self,stage):
        plan_path=OUTPUT/'plan.json'
        plan=read(plan_path);plan_sha=sha(plan_path)
        need(plan['schema']=='secondary-causal-v1' and plan['output']==str(OUTPUT) and plan['model']==MODEL
             and plan['stop_unix']==STOP_UNIX and plan['quality_receipt']==str(QUALITY)
             and plan['files'][str(QUALITY)]==self.state['upstream_quality_receipt_sha256'],
             'Approved operational plan differs')
        if stage=='plan':
            self.state['causal_plan_sha256']=plan_sha
            return {'path':str(plan_path),'sha256':plan_sha}
        need(plan_sha==self.state['causal_plan_sha256'],'Operational plan changed after creation')
        if stage in ('training','heldout'):
            return mode_receipt(OUTPUT/stage/'receipt.json',stage,plan_sha,6 if stage=='training' else 3)
        administrative=read(OUTPUT/'freeze-heldout/receipt.json')
        need(administrative['complete'] is True and administrative['cleanup_complete'] is True
             and administrative['plan_sha256']==plan_sha,'Post-training administrative freeze incomplete')
        path=OUTPUT/'heldout-freeze-receipt.json';value=read(path)
        need(value['complete'] is True and value['plan_sha256']==plan_sha
             and value['training_receipt_sha256']==sha(OUTPUT/'training/receipt.json')
             and value['review_sha256']==sha(TOOLS/'heldout-causal-v2-independent-review-v1.json')
             and value['protocol_sha256']==sha(OUTPUT/'heldout-prepared/protocol.json')
             and value['selected_requests_sha256']==sha(OUTPUT/'heldout-prepared/selected-requests.json')
             and value['no_heldout_model_queries_before_freeze'] is True,'Heldout protocol freeze changed')
        return {'path':str(path),'sha256':sha(path),'administrative_receipt_sha256':sha(OUTPUT/'freeze-heldout/receipt.json')}

    def run_stage(self,stage,command):
        self.validate()
        entry={'stage':stage,'command':command,'complete':False,'started_unix':time.time()}
        self.state['stages'].append(entry);self.state['stage']=stage;self.save()
        failure=None
        try:
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
            for key in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HF_HUB_TOKEN'):env.pop(key,None)
            with (OUTER/(stage+'.log')).open('xb') as log:
                self.child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                self.child_identity=self.lib.process_record(self.child.pid)
                need(self.child_identity and self.child_identity['pid']==self.child_identity['pgrp'],'Cannot bind owned causal child')
                entry['process']=self.child_identity;self.state['active_child']=self.child_identity;self.save()
                while self.child.poll() is None:
                    self.validate();self.save();time.sleep(1)
                entry['returncode']=self.child.returncode
                need(self.child.returncode==0,'Causal child failed: '+stage)
            entry['receipt']=self.stage_receipt(stage)
        except BaseException:
            failure=traceback.format_exc();entry['error']=failure
        finally:
            errors=[]
            if self.child is not None:
                try:
                    entry['cleanup']=cleanup_owned(self.child,self.child_identity,self.lib,grace=720)
                    self.child=self.child_identity=None;self.state.pop('active_child',None)
                except BaseException:errors.append({'scope':'owned_child','error':traceback.format_exc()})
            if self.child is None:
                try:entry['gpu_idle_after_child']=self.lib.gpu_idle(wait=60)
                except BaseException:errors.append({'scope':'whole_GPU_idle_after_owned_child','error':traceback.format_exc()})
            entry.update(complete=failure is None and not errors,cleanup_errors=errors,finished_unix=time.time())
            with (OUTER/(stage+'-receipt.json')).open('x') as stream:
                json.dump(entry,stream,sort_keys=True,indent=2);stream.write('\n')
            self.save()
        need(entry['complete'],'Causal stage failed; immutable stage evidence retained without retry')

    def run(self):
        try:
            need(self.initial_error is None,'Outer request could not be read: '+str(self.initial_error))
            while True:
                self.validate()
                if upstream_ready(QUALITY,self.lib):break
                self.save();time.sleep(10)
            self.upstream_qualified=True
            self.state['upstream_quality_receipt_sha256']=sha(QUALITY)
            self.state['upstream_quality_plan_sha256']=sha(QUALITY_PLAN)
            self.state['stage']='quality complete and original controller exited; preparing approved causal plan';self.save()
            for stage,command in stage_commands():self.run_stage(stage,command)
            self.state.update(complete=True,stage='training6, post-training freeze and heldout3 complete')
        except BaseException:
            self.state.update(error=traceback.format_exc(),stage='failed or upstream failed; no replacement work')
            raise
        finally:
            errors=[]
            if self.child is not None:
                try:
                    self.state['final_child_cleanup']=cleanup_owned(self.child,self.child_identity,self.lib,grace=720)
                    self.child=self.child_identity=None;self.state.pop('active_child',None)
                except BaseException:errors.append({'scope':'owned_child','error':traceback.format_exc()})
            # Never inspect idle GPUs while the upstream campaign owns them or
            # while our child remains live. No cleanup action targets upstream.
            if self.upstream_qualified and self.child is None:
                try:self.state['final_gpu_idle']=self.lib.gpu_idle(wait=60)
                except BaseException:errors.append({'scope':'whole_GPU_idle_after_owned_child','error':traceback.format_exc()})
            if errors:self.state['complete']=False
            self.state.update(cleanup_complete=not errors,cleanup_errors=errors,finished_unix=time.time())
            self.save()
            with (OUTER/'receipt.json').open('x') as stream:
                json.dump(self.state,stream,sort_keys=True,indent=2);stream.write('\n')
            self.lock.close()
            need(not errors,'Owned causal outer cleanup failed; retained without retry')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--request-sha256',required=True)
    args=parser.parse_args()
    lib=load_lifecycle()
    def cancel(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        raise InterruptedError('causal outer received '+signal.Signals(signum).name)
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    Outer(args.request,args.request_sha256,lib).run()


if __name__=='__main__':main()
