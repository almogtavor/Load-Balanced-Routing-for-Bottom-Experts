#!/usr/bin/env python3
"""Bounded primary-host sequencer: closed profiles, proof3, timing9, quality3.

No cluster operations. Frozen inner helpers own every server/HTTP/raw contract.
This outer driver waits without touching GPUs and preserves failed attempts.
"""
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
import recovery_provenance as recovery
import prospective_gate as prospective

STOP_UNIX=1792349474  # Fresh compute cutoff18:51:14UTC, one hour before pod deadline.
TOOLS=Path('/work/gap/hybrid')
OUTPUT=Path('/work/results/gap-hybrid-v1')
OUTER=Path('/work/results/gap-hybrid-outer-v1')
FINAL=Path('/work/results/gap-hybrid-outer-driver-v1.json')
UPSTREAM=Path('/work/recovered-primary-results/prefill-profiles-v4/receipt.json')
UPSTREAM_STATUS=Path('/work/recovered-primary-results/prefill-profiles-v4/status.json')
UPSTREAM_LAUNCH=Path('/work/recovered-primary-results/provenance/profile-v4-launch-v1.json')
PROFILE_PLAN=Path('/work/recovered-primary-results/prefill-profiles-v4/plan.json')
PROFILE_SHA='132e245b1d8be5abcc81df1e405e170ef91e80c7b4acc1a9515a7694b3bc89c2'
MAIN_PLAN=Path('/work/recovered-primary-results/prefill-e2e-v1/plan.json')
MAIN_SHA='aa8224c1a76be70ad0a61c097dd2561e0e01efb5bfc4126970065b3892aec4bf'
PROFILE_HELPER_SHA='4fc188540740144ab6811e6041fff807243b60a6c1b8a177191748ae88b68858'
LIFECYCLE=Path('/work/followup/performance/prefill_e2e.py')
LIFECYCLE_SHA='2b1832a73ec1d84ef16746c99a8ad94c82f4300ef7464fc4d2250e63b66b6db3'
HELPER_SHA='65f846b2b79ed440e133dd3a8b11a8c449d37172297fd083bc3cf648875fc32a'
FREEZE_SHA='cbaa2f390d8f7f46e3eda7e93e3668921211073f44de4acbe93288167af8246b'
MODEL='/work/hf/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe'
UID=recovery.OWNER['uid']
RECOVERY=Path('/work/results/provenance/gap-hybrid-owner-binding-v1.json')
RECOVERY_SHA='a5a3c743f4c46539b8f6b05c60ff454fac0b8fb60de866d9138a9071fdbc5088'
ARMS=['stock','keep6p8','keep4p8']
SWEEPS=[ARMS,['keep6p8','keep4p8','stock'],['keep4p8','stock','keep6p8']]


def need(value,message):
    if not value:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4<<20),b''):h.update(part)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def write_new(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')


def load_lifecycle():
    need(sha(LIFECYCLE)==LIFECYCLE_SHA,'Frozen lifecycle helper changed before import')
    spec=importlib.util.spec_from_file_location('hybrid_outer_frozen_lifecycle',LIFECYCLE)
    module=importlib.util.module_from_spec(spec);previous=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    try:spec.loader.exec_module(module)
    finally:sys.dont_write_bytecode=previous
    return module


def exited(lib,identity):
    live=lib.process_record(identity['pid'])
    return not live or str(live['start_ticks'])!=str(identity['start_ticks']) or live['state']=='Z'


def plan_command():
    return ['python3','-u',str(TOOLS/'hybrid_driver.py'),'plan','--output',str(OUTPUT),'--model',MODEL,
        '--production-root','/work/task-router','--site-root','/work/task-site',
        '--source-proof','/work/results/provenance/task-source-deployment.json',
        '--checkpoint-proof','/work/results/checkpoint-proof.json','--allocation-proof','/work/results/allocation-proof.json',
        '--runtime-preflight','/work/results/hybrid-runtime-preflight-v1.json','--lifecycle-helper',str(LIFECYCLE),
        '--recovery-proof',str(RECOVERY),'--recovery-proof-sha256',RECOVERY_SHA,
        '--vllm','/usr/local/bin/vllm','--prefill-prompts','/work/followup/performance/prompts64.json',
        '--preflight-inputs','/work/results/followup-prepared-v1/preflight.json',
        '--quality-inputs','/work/results/followup-prepared-v1/requests.json',
        '--prerequisite-receipts',str(recovery.prior_path('/work/results/prefill-e2e-v1/timed/receipt.json')),
        str(UPSTREAM),str(recovery.prior_path('/work/results/prefill-driver-v1.json')),
        '--failed-prior-receipts',*[str(recovery.prior_path(p)) for p in (
        '/work/results/prefill-profiles-v1/receipt.json','/work/results/profile-waiter-v1.json',
        '/work/results/hybrid-outer-driver-v1.json','/work/results/prefill-profiles-v2/receipt.json',
        '/work/results/prefill-profiles-v3/receipt.json','/work/results/hybrid-v2/proof/receipt.json',
        '/work/results/hybrid-outer-driver-v3.json')], '--stop-unix',str(STOP_UNIX)]


def stage_commands():
    base=['python3','-u',str(TOOLS/'hybrid_driver.py'),'run','--plan',str(OUTPUT/'plan.json')]
    return [('plan',plan_command())]+[(mode,base+['--mode',mode]) for mode in ('proof','timed','quality')]


def request_gate(request,expected_sha,path,*,rehash_files=True):
    need(STOP_UNIX>0 and RECOVERY_SHA!='PENDING_FRESH_OWNER_PROOF' and FREEZE_SHA!='PENDING_FRESH_CORE_FREEZE', 'Fresh owner/core/cutoff must be frozen before launch')
    need(sha(path)==expected_sha and request['schema']=='bounded-fresh-hybrid-outer-request-v1','Frozen outer request differs')
    need(request['driver_sha256']==sha(__file__) and request['stop_unix']==STOP_UNIX
         and request['tools']==str(TOOLS) and request['output']==str(OUTPUT) and request['outer_output']==str(OUTER)
         and request['final_receipt']==str(FINAL) and request['upstream_profile_receipt']==str(UPSTREAM)
         and request['upstream_controller']=={'pid':38261,'start_ticks':'6345502'}
         and request['poll_seconds']==10 and request['child_cleanup_grace_seconds']==720
         and request['commands']==[[k,v] for k,v in stage_commands()], 'Outer commands/owner/deadline differs')
    need(request['files'][str(RECOVERY)]==RECOVERY_SHA
         and request['files'][str(TOOLS/'hybrid_driver.py')]==HELPER_SHA
         and request['files'][str(TOOLS/'helper-freeze-fresh-v1.json')]==FREEZE_SHA
         and request['files'][str(LIFECYCLE)]==LIFECYCLE_SHA,'Wrong reviewed scientific helpers')
    if rehash_files:
        for path,digest in request['files'].items():need(sha(path)==digest,'Frozen prerequisite changed: '+path)
        need(request['prospective_admission']==prospective.verify(load_lifecycle()),'Prospective native diagnostic/failure admission changed')
        allocation=read('/work/results/allocation-proof.json')
        binding=recovery.verify(RECOVERY,RECOVERY_SHA,allocation)
        need(request['recovery_binding']==binding and request['pod_uid']==UID,'Recovery owner request differs')
        source=read('/work/results/provenance/task-source-deployment.json')['pod_identity']
        need(all(source[k]==allocation[k] for k in ('namespace','pod','uid','node','image')),'Source/allocation differ')
        launch=read(UPSTREAM_LAUNCH)
        need(launch['complete'] is True and launch['started'] is True
             and launch['process']['pid']==launch['process']['pgrp']==38261 and str(launch['process']['start_ticks'])=='6345502'
             and launch['pod_uid']==recovery.OLD['uid'] and launch['plan_sha256']==PROFILE_SHA
             and sha(PROFILE_PLAN)==PROFILE_SHA and sha(MAIN_PLAN)==MAIN_SHA,'V2 profile launch/plan identity differs')
        profile=read(PROFILE_PLAN)
        need(profile['schema']=='headline-prefill-profiles-v4' and profile['main_plan_sha256']==MAIN_SHA
             and profile['profile_files']['/work/followup/profiles-v4/profile_prefill.py']==PROFILE_HELPER_SHA,
             'Actual-V2 profile helper/source plan differs')


def upstream_ready(lib):
    need(UPSTREAM.is_file(),'Archived closed profile receipt missing')
    value=read(UPSTREAM)
    identity=value['controller_process']
    need(identity['pid']==identity['pgrp']==38261 and str(identity['start_ticks'])=='6345502'
         and value['plan_sha256']==PROFILE_SHA,'Upstream V2 profile belongs to another process/plan')
    if value.get('error') or value.get('cleanup_errors'):raise RuntimeError('V2 profile failed; no hybrid child permitted')
    need(value.get('complete') is True and value.get('cleanup_complete') is True
         and value.get('cleanup_errors')==[] and value.get('finished_unix_ns')
         and not value.get('active_server') and not value.get('active_client')
         and len(value['launches'])==3 and {r['arm'] for r in value['launches']}==set(ARMS)
         and value['main_guard']['receipt_sha256']==sha(recovery.prior_path('/work/results/prefill-e2e-v1/timed/receipt.json'))
         and value['main_guard']['complete_launches']==36 and value['main_guard']['controller_original_exited'] is True
         and value['main_guard']['main_lock_held'] is True,'V2 profile incomplete/unclean/unbound')
    for row in value['launches']:
        need(row['complete'] is True and sha(recovery.prior_path(row['receipt_file']))==row['receipt_sha256'],'V2 profile failed/changed launch')
        launch=read(recovery.prior_path(row['receipt_file']))
        need(launch['complete'] is True and launch['plan_sha256']==PROFILE_SHA and launch['arm']==row['arm']
             and launch['finalization_errors']==[] and launch['client_cleanup']['remaining_members']==[]
             and launch['server_cleanup']['remaining_members']==[]
             and set(launch['actual_runner_startup_contract'])=={'0','1','2','3'},'V2 profile launch lacks actual native/closure proof')
    return True


def cleanup_owned(child,identity,lib):
    need(identity and identity['pid']==identity['pgrp']==child.pid,'Owned child group identity required')
    current=lib.process_record(child.pid)
    need(not current or str(current['start_ticks'])==str(identity['start_ticks']),'Owned child PID reused')
    result={'identity':identity,'signals':[],'started_unix':time.time(),'grace_seconds':720}
    for signum,seconds in ((signal.SIGTERM,720),(signal.SIGKILL,15)):
        if not lib.group_members(child.pid):break
        try:os.killpg(child.pid,signum);result['signals'].append(signum.name)
        except ProcessLookupError:break
        deadline=time.monotonic()+seconds
        while lib.group_members(child.pid) and time.monotonic()<deadline:child.poll();time.sleep(.25)
    try:child.wait(timeout=2)
    except subprocess.TimeoutExpired:pass
    result.update(remaining_members=lib.group_members(child.pid),returncode=child.poll(),finished_unix=time.time())
    need(not result['remaining_members'],'Owned child group remains')
    return result


def mode_receipt(mode,plan_sha,lib):
    path=OUTPUT/mode/'receipt.json';value=read(path)
    schedule=SWEEPS if mode=='timed' else [ARMS]
    expected=[(s,p,a) for s,order in enumerate(schedule) for p,a in enumerate(order)]
    need(value.get('complete') is True and value.get('cleanup_complete') is True and value['plan_sha256']==plan_sha
         and value['mode']==mode and not value.get('error') and not value.get('cleanup_error') and not value.get('active_servers')
         and [(r['sweep'],r['position'],r['arm']) for r in value['launches']]==expected,'Hybrid stage matrix incomplete or changed')
    for row in value['launches']:
        need(row['complete'] is True and sha(row['receipt'])==row['sha256'],'Hybrid failed/changed launch')
        launch=read(row['receipt'])
        need(launch['complete'] is True and launch['workload_complete'] is True and launch['finalization_errors']==[]
             and launch['plan_sha256']==plan_sha and (launch['sweep'],launch['position'],launch['arm'])==(row['sweep'],row['position'],row['arm']),
             'Hybrid launch identity/closure differs')
        for role in ('prefill','decode'):
            need(launch['closure'][role]['remaining_members']==[] and launch[role+'_activation_contract'] is True,'Hybrid role not closed/active correctly')
        if mode!='timed':
            audit=launch['phase_audit'];need(sha(audit['path'])==audit['sha256'],'Phase proof changed')
            phase=read(audit['path'])
            need(phase['complete'] is True and phase['actual_nixl_bytes']>0
                 and phase['prospective_revision']==6 and phase['monolithic_equivalence_claim'] is False
                 and phase['every_request_has_positive_remote_allocation_and_receive'] is True
                 and phase['every_decode_scheduled_request_one_token'] is True
                 and phase['actual_v2_runner_on_all8_workers'] is True
                 and phase['every_producer_full_prompt_all48_routers'] is True
                 and phase['decoder_exact_transferred_prefix_then_native_last_prompt_token'] is True
                 and phase['exact_native_external_internal_mapping'] is True
                 and phase['request_id_contract_sha256']=='2191e7e4d88f9ae42b0b99f3614ef50865c50eb8ed7daf1ee21994321bb07df6',
                 'Actual native V2 last-token handoff proof incomplete')
            score_path=Path(launch['directory'])/'gsm-score.json';score=read(score_path)
            need(score['complete'] is True and score['n']==(4 if mode=='proof' else 1319)
                 and score['prefill_arm']==row['arm'] and score['decode_arm']=='stock'
                 and score['phase_audit_sha256']==audit['sha256'],'Actual hybrid GSM score scope differs')
        else:
            need([(r['phase'],r['number']) for r in launch['rounds']]==[(p,n) for p in ('warmup','measured') for n in range(3)],'Fixed3+3 timing rounds differ')
            for ref in launch['rounds']:
                need(sha(ref['path'])==ref['sha256'] and read(ref['path'])['complete'] is True,'Raw timed round incomplete/changed')
                result=read(ref['path']);native=result['native_metrics']
                need(native['complete'] is True and sha(native['path'])==native['sha256']
                     and native['native_coverage']['decode']['native_transfer_count']==256
                     and native['native_coverage']['decode']['native_transfer_bytes']==25769803776
                     and native['native_coverage']['prefill']['native_transfer_count']==0
                     and native['native_coverage']['prefill']['native_transfer_bytes']==0,
                     'Fixed native KV-handoff count/bytes coverage incomplete')
    need(exited(lib,value['controller_process']),'Original inner hybrid controller still active')
    return {'path':str(path),'sha256':sha(path),'launches':len(expected),'controller_process':value['controller_process'],'original_controller_exited':True}


class Outer:
    def __init__(self,path,expected_sha,lib):
        self.path,self.expected_sha,self.lib=path,expected_sha,lib
        OUTER.mkdir(exist_ok=False);need(not FINAL.exists(),'Final outer receipt already exists; preserve it')
        self.lock=(OUTER/'outer.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.status=OUTER/'status.json';self.child=self.child_identity=None;self.upstream_qualified=False
        self.initial_error=None;self.request=None
        self.state={'complete':False,'cleanup_complete':False,'request_sha256':expected_sha,'driver_sha256':sha(__file__),
            'controller_process':lib.process_record(os.getpid()),'started_unix':time.time(),'stages':[],
            'hybrid_plan_sha256':None,'hybrid_receipt_sha256':{},'hybrid_controller_processes':{},'stage':'waiting for archived closed H100 profiles and exact original pod absence'}
        self.save()
        try:self.request=read(path)
        except BaseException:self.initial_error=traceback.format_exc()

    def save(self):self.state['updated_unix']=time.time();self.lib.save(self.status,self.state)

    def validate(self,*,rehash_files=False):
        need(time.time()<STOP_UNIX,'Absolute hybrid compute cutoff reached')
        request_gate(self.request,self.expected_sha,self.path,rehash_files=rehash_files)
        if self.upstream_qualified:
            need(sha(UPSTREAM)==self.state['upstream_profile_receipt_sha256'] and upstream_ready(self.lib),'Archived V2 profile evidence changed')

    def stage_receipt(self,stage):
        path=OUTPUT/'plan.json';value=read(path);digest=sha(path)
        need(value['schema']=='real-nixl-hybrid-v6' and value['output']==str(OUTPUT) and value['model']==MODEL
             and value['stop_unix']==STOP_UNIX and value['timed_sweeps']==SWEEPS
             and value['files'][str(TOOLS/'hybrid_driver.py')]==HELPER_SHA,'Reviewed hybrid operational plan differs')
        if stage=='plan':self.state['hybrid_plan_sha256']=digest;return {'path':str(path),'sha256':digest}
        need(digest==self.state['hybrid_plan_sha256'],'Hybrid plan changed after freeze')
        result=mode_receipt(stage,digest,self.lib)
        self.state['hybrid_receipt_sha256'][stage]=result['sha256']
        self.state['hybrid_controller_processes'][stage]=result['controller_process']
        return result

    def run_stage(self,stage,command):
        self.validate(rehash_files=True)
        entry={'stage':stage,'command':command,'complete':False,'started_unix':time.time()}
        self.state['stages'].append(entry);self.state['stage']=stage;self.save();failure=None
        try:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
            for key in ('HF_TOKEN','HF_HUB_TOKEN','HUGGING_FACE_HUB_TOKEN'):env.pop(key,None)
            with (OUTER/(stage+'.log')).open('xb') as stream:
                self.child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                self.child_identity=self.lib.process_record(self.child.pid)
                need(self.child_identity and self.child_identity['pid']==self.child_identity['pgrp'],'Cannot bind owned hybrid child')
                entry['process']=self.child_identity;self.state['active_child']=self.child_identity;self.save()
                while self.child.poll() is None:self.validate();self.save();time.sleep(1)
                entry['returncode']=self.child.returncode;need(self.child.returncode==0,'Hybrid child failed: '+stage)
            entry['receipt']=self.stage_receipt(stage)
            self.validate(rehash_files=True)
        except BaseException:failure=traceback.format_exc();entry['error']=failure
        finally:
            errors=[]
            if self.child is not None:
                try:
                    entry['cleanup']=cleanup_owned(self.child,self.child_identity,self.lib)
                    self.child=self.child_identity=None;self.state.pop('active_child',None)
                except BaseException:errors.append({'scope':'owned_child','error':traceback.format_exc()})
            if self.child is None:
                try:entry['gpu_idle_after_child']=self.lib.gpu_idle(wait=60)
                except BaseException:errors.append({'scope':'GPU_idle_after_owned_child','error':traceback.format_exc()})
            entry.update(complete=failure is None and not errors,cleanup_errors=errors,finished_unix=time.time())
            write_new(OUTER/(stage+'-receipt.json'),entry);self.save()
        need(entry['complete'],'Hybrid stage failed; immutable raw/receipt retained, no replacement')

    def run(self):
        try:
            need(self.initial_error is None,'Outer request unreadable: '+str(self.initial_error))
            need(self.state['controller_process'] and self.state['controller_process']['pid']==os.getpid()
                 and self.state['controller_process']['start_ticks'],'Cannot bind outer controller identity')
            self.validate(rehash_files=True)
            while True:
                self.validate()
                if upstream_ready(self.lib):break
                self.save();time.sleep(10)
            self.upstream_qualified=True;self.state['upstream_profile_receipt_sha256']=sha(UPSTREAM)
            self.state['upstream_profiles_sha256']=self.state['upstream_profile_receipt_sha256'];self.save()
            for stage,command in stage_commands():self.run_stage(stage,command)
            need(set(self.state['hybrid_receipt_sha256'])=={'proof','timed','quality'},'Allthree inner receipt hashes required')
            need(all(exited(self.lib,p) for p in self.state['hybrid_controller_processes'].values()),'An original inner controller remains active')
            self.state.update(complete=True,stage='proof3, clean timed9 and actual hybrid quality3 closed')
        except BaseException:self.state.update(error=traceback.format_exc(),stage='failed/upstream failed; raw retained, no replacement');raise
        finally:
            errors=[]
            if self.child is not None:
                try:
                    self.state['final_child_cleanup']=cleanup_owned(self.child,self.child_identity,self.lib)
                    self.child=self.child_identity=None;self.state.pop('active_child',None)
                except BaseException:errors.append({'scope':'owned_child','error':traceback.format_exc()})
            if self.upstream_qualified and self.child is None:
                try:self.state['final_gpu_idle']=self.lib.gpu_idle(wait=60)
                except BaseException:errors.append({'scope':'GPU_idle_after_owned_child','error':traceback.format_exc()})
            if errors:self.state['complete']=False
            self.state.update(cleanup_complete=not errors,cleanup_errors=errors,finished_unix=time.time())
            self.save();write_new(OUTER/'receipt.json',self.state);write_new(FINAL,self.state);self.lock.close()
            need(not errors,'Owned hybrid outer cleanup not verified')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True);p.add_argument('--request-sha256',required=True);args=p.parse_args()
    lib=load_lifecycle()
    def cancel(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        raise InterruptedError('Owned hybrid outer received '+signal.Signals(signum).name)
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    Outer(args.request,args.request_sha256,lib).run()


if __name__=='__main__':main()
