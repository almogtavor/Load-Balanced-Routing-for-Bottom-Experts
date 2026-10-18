#!/usr/bin/env python3
"""Offline full3 GSM8K V6 audit; replays unchanged native phase/scoring sources.

All proof/timed/quality matrices and the finite outer must be closed before
quality payloads are read. No GPU/model/HTTP call, new prompt, scoring rule,
confidence interval, or comparison with monolithic launches is introduced.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
V6 = HERE / 'core'
CORE_SHA = 'cbaa2f390d8f7f46e3eda7e93e3668921211073f44de4acbe93288167af8246b'
RAW_SHA = '595e6a14c4c36d2230d61699576ead34cf5e3ea504edf13b6ea1f3d009f081ac'
SCORE_SHA = '605a20e78871816792847bc9520022af696f493ab48944e8fe893724698c3604'
ARMS = ('stock','keep6p8','keep4p8')
FULL_SHA = '6c3cdfb53702f5b1e07f8d38a9770b5da45b5d419b5ad81d5b9e81908c0cf63b'
OUTER = '/work/results/gap-hybrid-outer-driver-v1.json'
OUTER_SOURCE_SHA = 'ca0dc2f4fd50349e9ab37ef67bb6da966540a244c74e4d73c580933415421956'
OUTER_REQUEST_SHA = '5d66fa4d9ccbac46f223d6e81d35301eb64638e68a9122157245f8731e99ba54'
OUTER_REQUEST = '/work/gap/hybrid/hybrid-outer-request-fresh-v1.json'
OUTER_SOURCE = '/work/gap/hybrid/hybrid_outer_driver.py'
EXIT = '/work/results/provenance/gap-hybrid-outer-original-exit-v1.json'
POD_UID = '8e098972-bdeb-4589-b79f-324bfd601784'
OUTER_ID = {'pid':223,'pgrp':223,'start_ticks':'1122977981'}
EXIT_COLLECTOR_SHA = '23a6e79ebfefe43f0ae09c1f8b5cd299e14521809c56c2f31213763a6c62b26d'
EXIT_COLLECTOR = '/work/results/provenance/gap-hybrid-outer-exit-collector-source-v1.py'


def need(value,message):
    if not value:
        raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


@contextlib.contextmanager
def frozen():
    need(sha(V6/'helper-freeze-fresh-v2.json')==CORE_SHA,'Frozen V6 core manifest changed')
    for name,digest in load(V6/'helper-freeze-fresh-v2.json')['files'].items():
        need(sha(V6/name)==digest,'Frozen V6 source asset changed: '+name)
    names=['common_v2','hybrid_client','native_runner_contract','native_request_id','recovery_provenance',
           'native_accounting','prospective_gate','audit_native_metrics','archived_h200_closure']
    prior={n:sys.modules.get(n) for n in names};oldpath=list(sys.path);bytecode=sys.dont_write_bytecode
    try:
        for n in names:sys.modules.pop(n,None)
        sys.path.insert(0,str(V6));sys.dont_write_bytecode=True
        need(sha(V6/'audit_hybrid_timing_raw_v6.py')==RAW_SHA,'Frozen V6 raw reader changed')
        spec=importlib.util.spec_from_file_location('full_quality_v6_frozen_raw',V6/'audit_hybrid_timing_raw_v6.py')
        raw=importlib.util.module_from_spec(spec);spec.loader.exec_module(raw)
        scorer=raw.import_frozen(V6/'score_hybrid_gsm.py',SCORE_SHA,'full_quality_v6_frozen_scorer')
        for n in names:
            if n in sys.modules:
                need(Path(sys.modules[n].__file__).resolve()==(V6/(n+'.py')).resolve(),'Wrong frozen offline import: '+n)
        yield raw,scorer
        need(sha(V6/'helper-freeze-fresh-v2.json')==CORE_SHA,'Frozen V6 manifest changed during replay')
        for name,digest in load(V6/'helper-freeze-fresh-v2.json')['files'].items():
            need(sha(V6/name)==digest,'Frozen V6 source changed during replay: '+name)
    finally:
        sys.path[:]=oldpath;sys.dont_write_bytecode=bytecode
        for n in names:
            sys.modules.pop(n,None)
            if prior[n] is not None:sys.modules[n]=prior[n]


def closed_modes(bundle,plan,plan_sha,raw):
    """Closure/source first. This function never reads quality choices/scores."""
    bundle.checked_file(OUTER_SOURCE,OUTER_SOURCE_SHA)
    request=bundle.json(OUTER_REQUEST,OUTER_REQUEST_SHA)
    need(request['schema']=='bounded-fresh-hybrid-outer-request-v1' and request['pod_uid']==POD_UID
         and request['output']==plan['output'] and request['stop_unix']==plan['stop_unix']
         and request['driver_sha256']==OUTER_SOURCE_SHA,'Frozen outer request differs')
    for name,digest in request['files'].items():bundle.checked_file(name,digest)
    outer=bundle.json(OUTER)
    need(outer['complete'] is True and outer['cleanup_complete'] is True and not outer.get('error')
         and outer['cleanup_errors']==[] and not outer.get('active_child')
         and outer==bundle.json('/work/results/gap-hybrid-outer-v1/receipt.json')
         ==bundle.json('/work/results/gap-hybrid-outer-v1/status.json')
         and outer['driver_sha256']==OUTER_SOURCE_SHA and outer['request_sha256']==OUTER_REQUEST_SHA
         and outer['hybrid_plan_sha256']==plan_sha and set(outer['hybrid_receipt_sha256'])=={'proof','timed','quality'},
         'All-stage finite outer closure required before quality payloads')
    need(outer['final_gpu_idle']['returncode']==0 and not outer['final_gpu_idle']['stdout'].strip(),
         'Final whole-host GPU idle absent')
    process=outer['controller_process']
    need(all(process[k]==v for k,v in OUTER_ID.items()),'Outer original process identity missing')
    exit_record=bundle.json(EXIT)
    bundle.checked_file(EXIT_COLLECTOR,EXIT_COLLECTOR_SHA)
    need(exit_record['schema']=='fresh-hybrid-outer-original-exit-v1' and exit_record['complete'] is True
         and exit_record['original_exited'] is True and exit_record['pod_uid']==POD_UID
         and exit_record['pod_uid_before']==exit_record['pod_uid_after']==POD_UID
         and all(exit_record['pod_identity_before'][key]==exit_record['pod_identity_after'][key]
                 for key in ('pod','namespace','node','image','uid'))
         and all(exit_record['pod_identity_before'][key]==plan['allocation'][key]
                 for key in ('pod','namespace','node','image','uid'))
         and exit_record['pod_identity_before']['uid']==POD_UID
         and exit_record['original_controller']==OUTER_ID
         and exit_record['final_receipt']=={'path':OUTER,'sha256':sha(bundle.resolve(OUTER))}
         and exit_record['collector_source_sha256']==EXIT_COLLECTOR_SHA
         and finite(exit_record['checked_unix']) and finite(outer['finished_unix'])
         and outer['finished_unix']<exit_record['checked_unix'], 'Original outer exit/source/receipt proof missing')
    observed=exit_record['observed_process']
    need(observed is None or (type(observed['pid']) is int and observed['pid']==OUTER_ID['pid']
         and str(observed['start_ticks']).isdigit() and (observed['state']=='Z'
         or str(observed['start_ticks'])!=OUTER_ID['start_ticks'])), 'Original outer process remains active')
    need([r['stage'] for r in outer['stages']]==['plan','proof','timed','quality']
         and [[r['stage'],r['command']] for r in outer['stages']]==request['commands']
         and all(r['complete'] is True and not r['cleanup_errors'] for r in outer['stages']),
         'Fixed prospective outer sequence incomplete')
    for index,stage in enumerate(outer['stages']):
        need(stage==bundle.json('/work/results/gap-hybrid-outer-v1/'+stage['stage']+'-receipt.json')
             and stage['returncode']==0 and stage['cleanup']['remaining_members']==[]
             and stage['cleanup']['identity']==stage['process']
             and stage['gpu_idle_after_child']['returncode']==0 and not stage['gpu_idle_after_child']['stdout'].strip()
             and finite(stage['started_unix']) and finite(stage['finished_unix'])
             and stage['started_unix']<=stage['finished_unix']
             and (index==0 or outer['stages'][index-1]['finished_unix']<=stage['started_unix']),
             'Outer stage receipt/closed ordered interval differs')
    need(outer['stages'][0]['receipt']=={'path':plan['output']+'/plan.json','sha256':plan_sha}, 'Outer CPU plan receipt differs')
    modes={};launches={};seen=set()
    for mode in ('proof','timed','quality'):
        remote=plan['output']+'/'+mode
        receipt=bundle.json(remote+'/receipt.json',outer['hybrid_receipt_sha256'][mode])
        need(receipt==bundle.json(remote+'/status.json') and receipt['complete'] is True
             and receipt['cleanup_complete'] is True and receipt['plan_sha256']==plan_sha and receipt['mode']==mode
             and not receipt.get('error') and not receipt.get('cleanup_error') and not receipt.get('active_servers'),
             'Full V6 mode is not closed: '+mode)
        identity=receipt['controller_process']
        need(identity==outer['hybrid_controller_processes'][mode] and identity['pid']==identity['pgrp']
             and str(identity['start_ticks']).isdigit(),'Original inner controller binding differs')
        stage=next(r for r in outer['stages'] if r['stage']==mode)
        need(stage['cleanup']['remaining_members']==[] and stage['cleanup']['identity']==stage['process']
             and all(stage['process'][k]==identity[k] for k in ('pid','pgrp','start_ticks'))
             and stage['receipt']['original_controller_exited'] is True
             and stage['receipt']['controller_process']==identity
             and stage['receipt']['sha256']==outer['hybrid_receipt_sha256'][mode],
             'Original inner process exit/owned outer closure not established')
        need(receipt['emergency_closure']['gpu_idle']['returncode']==0
             and not receipt['emergency_closure']['gpu_idle']['stdout'].strip(),'Inner final GPU-idle closure absent')
        orders=raw.ORDERS if mode=='timed' else [list(ARMS)]
        expected=[(s,p,a) for s,order in enumerate(orders) for p,a in enumerate(order)]
        need([(r['sweep'],r['position'],r['arm']) for r in receipt['launches']]==expected,'Full fixed mode inventory differs: '+mode)
        expected_paths=set();launches[mode]=[]
        for reference,(sweep,position,arm) in zip(receipt['launches'],expected):
            directory=remote+f'/sweep{sweep:02d}-position{position}-{arm}'
            need(reference['complete'] is True and reference['receipt']==directory+'/launch.json', 'Canonical launch identity differs')
            launch=bundle.json(reference['receipt'],reference['sha256']);expected_paths.add(bundle.resolve(reference['receipt']))
            need(launch['plan_sha256']==plan_sha and (launch['sweep'],launch['position'],launch['arm'])==(sweep,position,arm),
                 'Launch identity/plan differs')
            raw.pair_contract(bundle,plan,launch,directory,arm,mode!='timed',seen)
            launches[mode].append((directory,launch))
        need(set(bundle.root.glob(mode+'/sweep*/launch.json'))==expected_paths,'Unexpected/missing launch attempt')
        modes[mode]=receipt
    need(modes['timed']['proof_receipt_sha256']==outer['hybrid_receipt_sha256']['proof']
         and modes['quality']['proof_receipt_sha256']==outer['hybrid_receipt_sha256']['proof'], 'Fresh modes not bound to proof3')
    return modes,launches


def recheck_phase(bundle,plan,directory,launch,prepared,raw):
    """Frozen native raw audit API, explicit prepared input and path-key rebasing."""
    stored=bundle.json(launch['phase_audit']['path'],launch['phase_audit']['sha256'])
    need(stored['complete'] is True and stored['prospective_revision']==6 and stored['monolithic_equivalence_claim'] is False,
         'V6 native phase evidence missing/relabeled')
    for name,digest in stored['phase_file_sha256'].items():bundle.checked_file(name,digest)
    client=directory+'/client';bundle.checked_file(client+'/result.json',stored['campaign_result_sha256'])
    result=bundle.json(client+'/result.json');start=bundle.json(client+'/start.json',result['start_sha256'])
    quality=prepared==plan['quality_inputs']
    need((start['mode'],start['expected_split'],start['jobs'],start['max_requests'],start['max_tokens'],start['concurrency'])
         ==(('generate','test',1319,1319,2048,64) if quality else ('smoke','train',4,4,128,4))
         and start['arm']==launch['arm'] and start['launch_id']==directory.rsplit('/',1)[1]
         and start['served_model']=='task-model' and start['performance_claim'] is False
         and start['model_id']=='Qwen/Qwen3-30B-A3B-Instruct-2507' and start['revision']==raw.REVISION
         and start['source_commit']==raw.COMMIT and start['stop_unix']==plan['stop_unix']
         and start['input_file_sha256']==plan['files'][prepared]
         and all(start[r+'_contract']==bundle.json(directory+'/'+r+'-contract.json') for r in ('prefill','decode')),
         'Native phase client input/geometry/owned role contract differs')
    module=raw.import_frozen(V6/'audit_hybrid.py',raw.HELPERS['audit_hybrid.py'],'full_quality_v6_frozen_phase')
    with tempfile.TemporaryDirectory(prefix='hybrid-v6-quality-phase-') as temporary:
        output=Path(temporary)/'phase.json'
        module.audit(SimpleNamespace(result=bundle.resolve(client+'/result.json').parent,prepared=bundle.resolve(prepared),
            prefill_probe=bundle.resolve(next(n for n in stored['phase_file_sha256'] if '/prefill-probe/' in n)).parent,
            decode_probe=bundle.resolve(next(n for n in stored['phase_file_sha256'] if '/decode-probe/' in n)).parent,output=output))
        actual=load(output)
    actual['phase_file_sha256']={name:actual['phase_file_sha256'][str(bundle.resolve(name))] for name in stored['phase_file_sha256']}
    need(actual==stored,'Unchanged native raw phase replay differs')
    for role in ('prefill','decode'):
        need({r['pid'] for r in actual['worker_identity'][role]}<=raw.owned_members(launch,role)
             and set(actual['request_id_mapping_pids'][role])<=raw.owned_members(launch,role)
             and actual['exact_native_external_internal_mapping'] is True,
             'Actual request/router native worker outside owned role group')
    for name,digest in {**result['raw_files_sha256'],**result['record_files_sha256'],**result.get('stock_reference_files_sha256',{})}.items():
        bundle.checked_file(client+'/'+name,digest)
    return stored


def replay_score(bundle,directory,launch,prepared,scorer):
    """Every phase must pass before caller invokes this unchanged score routine."""
    remote=directory+'/gsm-score.json';stored=bundle.json(remote)
    with tempfile.TemporaryDirectory(prefix='hybrid-v6-quality-score-') as temporary:
        output=Path(temporary)/'score.json'
        scorer.main(SimpleNamespace(result=bundle.resolve(directory+'/client/result.json').parent,
            prepared=bundle.resolve(prepared),phase_audit=bundle.resolve(launch['phase_audit']['path']),output=output))
        actual=load(output)
    need(actual==stored,'Unchanged full1319 frozen scoring replay differs')
    need(actual['complete'] is True and actual['n']==1319 and actual['prefill_arm']==launch['arm']
         and actual['decode_arm']=='stock' and actual['helper_sha256']==SCORE_SHA,'Full canonical GSM score scope differs')
    return actual


def observed_contrasts(scores):
    need(set(scores)==set(ARMS),'All three observed launches required before quality contrasts')
    baseline=scores['stock'];ids=[r['id'] for r in baseline['records']]
    need(len(ids)==1319 and len(set(ids))==1319,'Full unique GSM baseline required')
    results={}
    for arm in ARMS[1:]:
        treatment=scores[arm]
        need([r['id'] for r in treatment['records']]==ids and treatment['n']==baseline['n']==1319,
             'Paired canonical item identity differs')
        endpoints={}
        for endpoint in ('correct','strict_correct'):
            pairs=[(r[endpoint],b[endpoint]) for r,b in zip(treatment['records'],baseline['records'])]
            need(all(type(x) is bool and type(y) is bool for x,y in pairs),'Scoring endpoint type differs')
            gain=sum(x and not y for x,y in pairs);loss=sum(y and not x for x,y in pairs)
            endpoints[endpoint]={'treatment_correct':sum(x for x,y in pairs),'stock_handoff_correct':sum(y for x,y in pairs),
                'paired_gain_items':gain,'paired_loss_items':loss,'accuracy_difference_pp':100*(gain-loss)/1319}
        results[arm]=endpoints
    return results


def replay_all(bundle,plan,launches,raw,scorer):
    """All6 raw phase gates precede even the first stored accuracy artifact."""
    proofs=[recheck_phase(bundle,plan,directory,launch,plan['preflight_inputs'],raw)
            for directory,launch in launches['proof']]
    phases=[recheck_phase(bundle,plan,directory,launch,plan['quality_inputs'],raw)
            for directory,launch in launches['quality']]
    scores={launch['arm']:replay_score(bundle,directory,launch,plan['quality_inputs'],scorer)
            for directory,launch in launches['quality']}
    return proofs,phases,scores


def audit(root,plan_sha,bindings):
    root=Path(root).resolve();need(sha(root/'plan.json')==plan_sha,'Operational plan bytes changed')
    plan=load(root/'plan.json')
    with frozen() as (raw,scorer):
        bundle=raw.Bundle(root,plan,bindings);raw.audit_plan(bundle,plan,plan_sha)
        need(plan['files'][plan['quality_inputs']]==FULL_SHA,'Canonical full task input source differs')
        modes,launches=closed_modes(bundle,plan,plan_sha,raw)
        inputs=bundle.json(plan['quality_inputs']);gsm=[r for r in inputs if r['task']=='gsm8k']
        need(len(gsm)==1319 and all(r['split']=='test' for r in gsm)
             and [r['id'] for r in gsm]==['gsm8k:test:'+str(i) for i in range(1319)],'Full canonical held-out GSM set required')
        # No score artifact or comparative endpoint is read before all3 raw
        # quality/native phases and every required closure/source gate pass.
        proof,phases,scores=replay_all(bundle,plan,launches,raw,scorer)
        contrasts=observed_contrasts(scores)
        for name,item in list(bundle.checked.items()):bundle.checked_file(name,item['sha256'])
        return {'complete':True,'full_three_heldout_arms_before_effects':True,'plan_sha256':plan_sha,
            'quality_requests':3957,'quality_completion_http_requests':7914,'canonical_full_input_sha256':FULL_SHA,
            'frozen_core_sha256':CORE_SHA,'frozen_phase_sha256':raw.HELPERS['audit_hybrid.py'],'frozen_scorer_sha256':SCORE_SHA,
            'quality_scores':scores,'observed_paired_contrasts':contrasts,
            'native_quality_phase':phases,'native_training_proofs':proof,'checked_inputs':bundle.checked,
            'reader_sha256':sha(__file__),
            'qualifications':['Each arm has one fresh server-pair launch on the same H200 allocation; this is not a launch-population confidence interval.',
                'Observed fixed-corpus GSM8K comparison is against stock-prefill/stock-decode NIXL handoff, not stock monolithic serving.',
                'Full1319 canonical8shot/greedy/naturalEOS/max2048; empty/parse failures and length-finished generations remain in the denominator.',
                'No pooling with public monolithic9 or H100 prefill throughput; no new confidence interval, seed selection, or numerical-equivalence claim.',
                'Unchanged raw phase replay proves recorded native source/routing/positive allocation/receive and L-1 boundary, not clean-timing per-request tensor hashes.',
                'Timed9 closure is required here for protocol sequence; its performance/raw native-metrics effects require the separate full9 timing reader.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--bindings',type=Path,required=True);parser.add_argument('--bindings-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    need(sha(args.bindings)==args.bindings_sha256,'Explicit external bindings changed')
    need(not args.output.resolve().is_relative_to(args.root.resolve()),'Audit output must be outside immutable raw campaign')
    result=audit(args.root,args.plan_sha256,args.bindings)
    with args.output.open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
