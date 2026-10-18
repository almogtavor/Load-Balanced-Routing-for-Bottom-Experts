#!/usr/bin/env python3
"""Independent all-wire adapter for the fixed nine-pair real-NIXL timing study.

Remote output paths map to a downloaded campaign root. Other frozen paths need
explicit {remote:absolute -> local:relative-to-binding-file} bindings. Does not
run models or import the timing harness. The source-bound prospective phase auditor
is rerun independently for the three prior handoff proofs.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys
import tempfile
import traceback
from types import SimpleNamespace
if hashlib.sha256(Path(__file__).with_name('native_runner_contract.py').read_bytes()).hexdigest() != '8a5581ff2d5ccc066dcaf4b1c440103b369b872537a5916ee632a4e1c366fcb6':
    raise RuntimeError('Frozen local native runner source helper changed before import')
from native_runner_contract import NATIVE
if hashlib.sha256(Path(__file__).with_name('native_request_id.py').read_bytes()).hexdigest() != '2191e7e4d88f9ae42b0b99f3614ef50865c50eb8ed7daf1ee21994321bb07df6':
    raise RuntimeError('Frozen native request-ID source helper changed before import')
from native_request_id import SOURCES as REQUEST_ID_SOURCES, producer_internal_id
if hashlib.sha256(Path(__file__).with_name('recovery_provenance.py').read_bytes()).hexdigest() != 'ee44309b9ac6f1c47a839f84def89ce051a4a033620d911ab20f3f950ae1d5c3':
    raise RuntimeError('Frozen owner/recovery helper changed before import')
import recovery_provenance as recovery
for _name,_expected in {'native_accounting.py': '90a4af8427c51ce418080c2d08283d260ad6fd0ae6505c6df832ea77c8f218b0', 'prospective_gate.py': '2f6ba71b144d70a1858e95e531efa812611cfd90cb139ef74bdf7f7d7c40594f', 'audit_native_metrics.py': '4c8133eaa3a92086047adc85d8d35760ec9dd3334f3fd229dc4e156fd9ff3f06'}.items():
    if hashlib.sha256(Path(__file__).with_name(_name).read_bytes()).hexdigest()!=_expected:
        raise RuntimeError("Frozen prospective native accounting helper changed before import: "+_name)
import native_accounting as accounting
import prospective_gate as prospective
import audit_native_metrics as native_wire
FAILED_SHAS = {'/work/results/profile-waiter-v1.json': '2a58d98169cd9d5aed45b0989103b04debb9df902d6b32a4956196bf52f7ae2d',
 '/work/results/hybrid-outer-driver-v1.json': 'fd9384b46b531cf1623cbc77b8646df139039436b29489b9762aa38a2e35180a',
 '/work/results/prefill-profiles-v1/receipt.json': '8248cd8684a535ecd6b8446b329a02cdaabc19c9d17a496a1a8aedac22daa246',
 '/work/results/prefill-profiles-v2/receipt.json': '8a891170a632e82f79e3012435772d1bdb04b93d180c2df7d7590385aa6e889e',
 '/work/results/prefill-profiles-v3/receipt.json': 'eb14d91b42ea63d336805ad7818c66f42888ab690cd05eb9b84eb41f23875ad3',
 '/work/results/hybrid-v2/proof/receipt.json':'878101ff1d7b2178a4d866d393976477dd2e325be71c2b6b28dc6b9cb1323496',
 '/work/results/hybrid-outer-driver-v3.json':'98e67057a50b4d92665b9389beb9964b872f4830c47bf7cdde92956fb63ad883'}

HERE=Path(__file__).resolve().parent
ARMS=('stock','keep6p8','keep4p8')
ORDERS=[list(ARMS),['keep6p8','keep4p8','stock'],['keep4p8','stock','keep6p8']]
COMMIT='57c67bbe4228313a684d15815a684209f56e4de0'
REVISION='0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe'
STATS_SHA='5a9b1b5653b9a9c576bedc96a9ed2d3ce461de1b35b2eeb80c138022bfe0906d'
PROTOCOL_SHA='71d04fd879febaf5b9c41eff2789219ef31b66d3686d81e0c618fd27c75fa584'
PROMPTS_SHA='7660341b6e1419c80a71caec4a4c364c7e44339c6089c7c60d62390d1975f479'
SOURCE={'placement_router/__init__.py':'8425741d4c8d90b5ce2ce122cff4abb8a095b8e488ce9e18ce468a0916f5d0cf',
 'placement_router/kernel.py':'06bc0d28295d23d4f0ae5715a4ffa555bb8d986118af3f52fc5fdc4cf0432854',
 'placement_router/plugin.py':'c432fc5a35793afb8b721806656de651bc755192000bf8ad9c9eead5084cffd8',
 'placement_router/vendor/__init__.py':'623bf62f4ae70782a32c1691d3648c120c67c5e4853725d3881265b7dfd96b25',
 'placement_router/vendor/fused_topk_router.py':'eecf00dffc63e72289032ece5a4634e45d97a1a1ff860901ba54b6b826e5e5fc'}
HELPERS={'archived_h200_closure.py':'72515b110e24f0f7e9633a8c101c36ef9ef2b4c4957172848c5d2ef6558c39a1', 'recovery_provenance.py': 'ee44309b9ac6f1c47a839f84def89ce051a4a033620d911ab20f3f950ae1d5c3', 'hybrid_driver.py': '65f846b2b79ed440e133dd3a8b11a8c449d37172297fd083bc3cf648875fc32a', 'common_v2.py': '9e64f7f704346115f20387240a2a20faba0914b8bc33283f688239679aef8700', 'hybrid_timing.py': '10dbb9a4395d916e31370d0faa2ebe1b145ec62e8e26030ce17b648a50d724a4', 'hybrid_client.py': 'f43073b731be36f4045a2f47fd3be9f7a94838f43bf99e5a3a225650d919ffc2', 'hybrid_phase_probe.py': 'e1a83c619e9f58d77266084eb3116ecadade8467022409f86559e5f32a294592', 'audit_hybrid.py': '07e90b14cdc58db0e8ac58675faacb2dfbf3a54da6ffdc2f5301932606ef07ef', 'score_hybrid_gsm.py': '605a20e78871816792847bc9520022af696f493ab48944e8fe893724698c3604', 'native_runner_contract.py': '8a5581ff2d5ccc066dcaf4b1c440103b369b872537a5916ee632a4e1c366fcb6', 'native_request_id.py': '2191e7e4d88f9ae42b0b99f3614ef50865c50eb8ed7daf1ee21994321bb07df6', 'native_accounting.py': '90a4af8427c51ce418080c2d08283d260ad6fd0ae6505c6df832ea77c8f218b0', 'prospective_gate.py': '2f6ba71b144d70a1858e95e531efa812611cfd90cb139ef74bdf7f7d7c40594f', 'audit_native_metrics.py': '4c8133eaa3a92086047adc85d8d35760ec9dd3334f3fd229dc4e156fd9ff3f06'}


def need(value,message):
    if not value:raise ValueError(message)


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4<<20),b''):result.update(part)
    return result.hexdigest()


def load(path):return json.loads(Path(path).read_bytes())
def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def finite(value):return type(value) in (int,float) and math.isfinite(value)
def integer(value,minimum=0):return type(value) is int and value>=minimum


def close(a,b,message):
    need(finite(a) and finite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),message)


def relative(value):
    p=PurePosixPath(value)
    need(isinstance(value,str) and value and not p.is_absolute() and str(p)==value and '..' not in p.parts,'Unsafe relative artifact path')
    return p


def local_file(root,name):
    name=relative(name);root=Path(root).resolve();p=root.joinpath(*name.parts)
    for a in (p,*p.parents):
        need(not a.is_symlink(),'Symlink artifact path')
        if a==root:break
    need(p.is_file() and p.resolve().is_relative_to(root),'Missing/escaping artifact')
    return p


class Bundle:
    def __init__(self,root,plan,bindings):
        self.root=Path(root).resolve();self.remote=PurePosixPath(plan['output']);self.checked={}
        need(self.remote.is_absolute() and str(self.remote)==plan['output'] and '..' not in self.remote.parts,'Unsafe campaign output')
        self.binding_path=bindings;self.bindings=load(bindings) if bindings else {}
        need(isinstance(self.bindings,dict),'Explicit input bindings must be a JSON object')
    def resolve(self,name):
        p=PurePosixPath(name)
        need(p.is_absolute() and str(p)==name and '..' not in p.parts,'Unsafe remote artifact path')
        if p.is_relative_to(self.remote):return local_file(self.root,p.relative_to(self.remote).as_posix())
        need(self.binding_path and name in self.bindings,'External input needs binding: '+name)
        return local_file(Path(self.binding_path).parent,self.bindings[name])
    def checked_file(self,name,expected=None):
        p=self.resolve(name);digest=sha(p)
        need(expected is None or digest==expected,'Raw artifact SHA differs: '+name)
        self.checked[name]={'sha256':digest,'bytes':p.stat().st_size}
        return p
    def json(self,name,expected=None):return load(self.checked_file(name,expected))


def inventory(folder,expected):
    folder=Path(folder);entries=list(folder.iterdir())
    need(not folder.is_symlink() and all(p.is_file() and not p.is_symlink() for p in entries),'Nonregular/symlink raw artifact')
    need({p.name for p in entries}==set(expected),'Unknown/missing raw artifact inventory')


def body(prompt,request_id,max_tokens=128):
    return {'model':'task-model','prompt':prompt,'request_id':request_id,'temperature':0,'top_p':1,'top_k':-1,'min_p':0,
            'repetition_penalty':1,'frequency_penalty':0,'presence_penalty':0,'seed':0,'max_tokens':max_tokens,
            'ignore_eos':True,'stream':False,'n':1,'return_token_ids':True,'add_special_tokens':False}


def transfer(producer,prompt):
    need(isinstance(producer.get('id'),str) and producer['id'],'Missing native producer response ID')
    choices=producer.get('choices');need(isinstance(choices,list) and len(choices)==1,'Producer choice count differs')
    choice=choices[0]
    need(choice.get('index')==0 and choice.get('prompt_token_ids')==prompt
         and isinstance(choice.get('token_ids'),list) and len(choice['token_ids'])==1
         and all(integer(t) for t in choice['token_ids']) and choice.get('finish_reason')=='length','Producer token/prompt/finish differs')
    usage=producer.get('usage',{})
    need(all(type(usage.get(k)) is int for k in ('prompt_tokens','completion_tokens','total_tokens'))
         and (usage['prompt_tokens'],usage['completion_tokens'],usage['total_tokens'])==(len(prompt),1,len(prompt)+1),'Producer accounting differs')
    p=producer.get('kv_transfer_params')
    try:full_native_id=producer_internal_id(producer)
    except RuntimeError as error:raise ValueError(str(error)) from error
    need(isinstance(p,dict) and p.get('do_remote_prefill') is True and p.get('do_remote_decode') is False
         and p.get('remote_request_id')==full_native_id and p.get('remote_engine_id') is not None
         and p.get('remote_host') is not None and p.get('remote_port') is not None,'Native transfer identity differs')
    blocks=p.get('remote_block_ids')
    need(isinstance(blocks,list) and blocks and all(isinstance(g,list) and all(integer(b) for b in g) for g in blocks)
         and any(blocks),'Missing/invalid native KV blocks')
    need(integer(p.get('remote_num_tokens')) and p['remote_num_tokens']==len(prompt)
         and p.get('tp_size')==p.get('pp_size')==1 and p.get('transfer_mode')=='pull'
         and type(p.get('remote_prefill_cached_tokens')) is int and p['remote_prefill_cached_tokens']==0,'Transfer layout/native zero producer prefix-hit accounting differs')
    return p


def parse_sse(raw,prompt):
    need(raw.endswith((b'\n\n',b'\r\n\r\n')),'SSE partial final event')
    fields=[];items=[];done=False
    for line in raw.decode('utf-8').splitlines():
        if line.startswith(':'):continue
        if line.startswith('data:'):
            need(not done,'SSE data after DONE');fields.append(line[5:].lstrip(' '))
        elif not line and fields:
            text='\n'.join(fields);fields=[]
            if text=='[DONE]':need(not done,'Duplicate DONE');done=True
            else:
                value=json.loads(text);need(not done and 'error' not in value,'SSE error/after DONE');items.append(value)
        elif line:need(line.startswith(('event:','id:','retry:')),'Unknown SSE field')
    need(done and not fields and items,'Missing SSE body/DONE')
    chunks=[];prompts=[];finishes=[];usages=[];ids=set()
    for ordinal,item in enumerate(items):
        need(not usages,'SSE JSON event after final usage')
        need(isinstance(item.get('id'),str) and item['id'],'Missing SSE native response ID');ids.add(item['id'])
        need(isinstance(item.get('choices'),list),'Missing SSE choices list')
        if item.get('usage') is not None:
            need(item['choices']==[] and finishes==['length'] and ordinal==len(items)-1
                 and sum(len(x['token_ids']) for x in chunks)==128,'SSE usage must be final empty-choice event after complete output/finish')
            usages.append(item['usage'])
        for choice in item['choices']:
            need(type(choice.get('index')) is int and choice['index']==0,'Unexpected SSE choice index')
            tokens=choice.get('token_ids')
            if tokens is None:need(not choice.get('text'),'SSE text without actual IDs');tokens=[]
            need(isinstance(tokens,list) and all(integer(t) for t in tokens),'Invalid SSE token IDs')
            if tokens:
                need(not finishes,'SSE token content after finish')
                chunks.append({'token_ids':tokens,'text':choice.get('text','')})
            else:need(not choice.get('text'),'SSE text without actual IDs')
            if choice.get('prompt_token_ids') is not None:prompts.append(choice['prompt_token_ids'])
            if choice.get('finish_reason') is not None:finishes.append(choice['finish_reason'])
    need(len(ids)==1 and prompts==[prompt] and finishes==['length'],'SSE ID/prompt/finish inventory differs')
    need(sum(len(x['token_ids']) for x in chunks)==128 and len(usages)==1,'SSE output or usage inventory differs')
    usage=usages[0]
    need(all(type(usage.get(k)) is int for k in ('prompt_tokens','completion_tokens','total_tokens'))
         and (usage['prompt_tokens'],usage['completion_tokens'],usage['total_tokens'])==(1024,128,1152),'Consumer accounting differs')
    cached=(usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    need(type(cached) is int and cached==0,'Expected native producer prefix-hit count zero, not transferred KV coverage')
    return {'id':next(iter(ids)),'chunks':chunks,'usage':usage,'cached_tokens':cached}


def audit_round(bundle,reference,remote,phase,number,prompts,launch_id,seen):
    need(reference['phase']==phase and reference['number']==number and reference['path']==remote,'Round reference identity differs')
    path=bundle.checked_file(remote,reference['sha256']);value=load(path)
    need(value.get('complete') is True and value['phase']==phase and value['round']==number
         and (value['logical_requests'],value['concurrency'],value['client_waves'],value['input_tokens'],value['output_tokens'])
         ==(256,64,4,262144,32768),'Round workload/completion differs')
    start,finish=value['cohort']['start_ns'],value['finished_ns']
    need(integer(start,1) and integer(finish,1) and finish>start and integer(value['cohort']['start_unix_ns'],1)
         and 'barrier_error' not in value['cohort'],'Round clock/barrier differs')
    seconds=(finish-start)/1e9
    close(value['elapsed_seconds'],seconds,'Round wall-clock formula differs');close(reference['elapsed_seconds'],seconds,'Round referenced clock differs')
    close(value['application_tokens_per_second'],294912/seconds,'Application token rate doublecount/formula differs')
    need(value['generation']=='forced128ignoreEOStrue; speed workload only','Fixed output policy differs')
    records=value['records'];need(len(records)==256 and [r['request_index'] for r in records]==list(range(256)),'Missing/duplicate/misordered logical requests')
    native_reports=native_wire.audit(bundle,remote,value,phase,number)
    expected={'round.json','application-round.json'};latencies=[]
    remote_dir=str(PurePosixPath(remote).parent)
    for i,(record,row) in enumerate(zip(records,prompts)):
        need(record.get('complete') is True and record.get('done') is True and 'error' not in record
             and record['producer_status']==record['consumer_status']==200,'Timed request incomplete/error')
        names={s:f'request{i:03d}.'+s for s in ('prefill.request.json','prefill.response.json','decode.request.json','decode.sse')}
        need(set(record['files_sha256'])==set(names.values()),'Request raw inventory differs');expected.update(names.values())
        raw={s:bundle.checked_file(remote_dir+'/'+n,record['files_sha256'][n]).read_bytes() for s,n in names.items()}
        request_id=f'{launch_id}-{phase}-{number}-{i:03d}'
        expected_p=body(row['token_ids'],request_id,1)
        expected_p['kv_transfer_params']={'do_remote_decode':True,'do_remote_prefill':False,'remote_engine_id':None,
                                         'remote_block_ids':None,'remote_host':None,'remote_port':None}
        need(json.loads(raw['prefill.request.json'])==expected_p,'Producer wire controls/prompt differ')
        producer=json.loads(raw['prefill.response.json']);params=transfer(producer,row['token_ids'])
        expected_d=body(row['token_ids'],request_id);expected_d.update(stream=True,stream_options={'include_usage':True},kv_transfer_params=params)
        need(json.loads(raw['decode.request.json'])==expected_d and record['transfer_params']==params,'Consumer wire/actual native params differ')
        parsed=parse_sse(raw['decode.sse'],row['token_ids'])
        for role,identity in (('prefill',producer['id']),('decode',parsed['id'])):
            need(identity=='cmpl-'+request_id and (role,identity) not in seen,'Wrong application response ID or reused native response');seen.add((role,identity))
        chunks=record['chunks']
        need(len(chunks)==len(parsed['chunks']) and [{k:x[k] for k in ('token_ids','text')} for x in chunks]==parsed['chunks']
             and record['usage']==parsed['usage'] and record['cached_prompt_tokens']==parsed['cached_tokens']
             and record['finish_reason']=='length' and record['prompt_ids_verified'] is True,'Recorded chunks/accounting differ from raw SSE')
        times=[record['request_start_ns'],record['producer_finished_ns'],*[x['arrival_ns'] for x in chunks],record['done_ns'],record['finished_ns']]
        need(all(integer(t,1) for t in times) and [start,*times,finish]==sorted([start,*times,finish])
             and record['cohort_start_ns']==start,'Two-leg/chunk timestamp order differs')
        first=chunks[0]['arrival_ns'];ttft=(first-times[0])/1e6;cohort_ttft=(first-start)/1e6;latency=(record['done_ns']-times[0])/1e6
        close(record['ttft_ms'],ttft,'Own-start TTFT formula differs');close(record['cohort_ttft_ms'],cohort_ttft,'Queue-inclusive TTFT formula differs')
        close(record['latency_ms'],latency,'Completion latency formula differs')
        latencies.append({'request_index':i,'own_ttft_ms':ttft,'cohort_ttft_ms':cohort_ttft,'latency_ms':latency,
                          'exploratory_transport_tpot_ms':(chunks[-1]['arrival_ns']-first)/127e6})
    for wave in range(3):
        need(max(r['finished_ns'] for r in records[wave*64:(wave+1)*64])
             <=min(r['request_start_ns'] for r in records[(wave+1)*64:(wave+2)*64]),'Sequential64-request client waves overlap')
    inventory(path.parent,expected)
    return seconds,latencies


def server_command(plan,role):
    kv={'kv_connector':'NixlConnector','kv_role':'kv_producer' if role=='prefill' else 'kv_consumer','kv_load_failure_policy':'fail'}
    return [plan['vllm'],'serve',plan['model'],'--served-model-name','task-model','--host','127.0.0.1',
      '--port',str(8100 if role=='prefill' else 8200),'--data-parallel-rpc-port',str(29550 if role=='prefill' else 29650),
      '--data-parallel-size','4','--tensor-parallel-size','1','--pipeline-parallel-size','1','--enable-expert-parallel',
      '--all2all-backend','deepep_high_throughput','--dtype','bfloat16','--expert-placement-strategy','linear',
      '--compilation-config={"cudagraph_mode":"NONE"}','--max-model-len','8192','--seed','0','--max-num-batched-tokens','16384',
      '--max-num-seqs','256','--no-enable-prefix-caching','--logprobs-mode','raw_logprobs','--enable-prompt-tokens-details',
      '--gpu-memory-utilization',str(plan['gpu_util']),'--kv-transfer-config',encoded(kv).decode()]


def owned_members(launch,role):
    members=launch['owned_group_members_before_cleanup'][role]
    process=launch['closure'][role]['identity']
    need(isinstance(members,list) and members and len({r['pid'] for r in members})==len(members)
         and all(integer(r['pid'],1) and r['pgrp']==process['pgrp'] and r['state']!='Z'
                 and str(r['start_ticks']).isdigit() for r in members),'Owned preclosure role member identity differs')
    return {r['pid'] for r in members}


def pair_contract(bundle,plan,launch,remote_dir,arm,instrumented,seen_processes):
    need(launch.get('complete') is True and launch.get('workload_complete') is True and launch['finalization_errors']==[],'Failed native pair')
    warnings={}
    for role in ('prefill','decode'):
        spec=bundle.json(remote_dir+'/'+role+'-launch.json')
        expected_arm=arm if role=='prefill' else 'stock'
        need(spec['role']==role and spec['arm']==expected_arm and spec['command']==plan['server_commands'][role]
             ==server_command(plan,role),'Physical pair role/arm/CLI differs')
        env={'PYTHONPATH':plan['site_root'],'VLLM_PLUGINS':'placement_aware_router','OMP_NUM_THREADS':'1',
             'NVSHMEM_QP_DEPTH':'4096','CUDA_VISIBLE_DEVICES':'0,1,2,3' if role=='prefill' else '4,5,6,7',
             'VLLM_NIXL_SIDE_CHANNEL_HOST':'127.0.0.1','VLLM_NIXL_SIDE_CHANNEL_PORT':'5600' if role=='prefill' else '5610',
             'UCX_RCACHE_MAX_UNRELEASED':'1024'}
        if expected_arm!='stock':env.update(PLACEMENT_KEEP='6' if arm=='keep6p8' else '4',PLACEMENT_LOAD_PENALTY='8')
        if instrumented:env.update(PLACEMENT_PROBE=plan['phase_probe'],HYBRID_PROBE_DIR=remote_dir+'/'+role+'-probe',HYBRID_ROLE=role,HYBRID_ARM=expected_arm)
        need(spec['environment']==env,'Recorded clean/instrumented target environment differs')
        imports=spec['cpu_import_proof'];names={'placement_router','placement_router.plugin','placement_router.kernel','placement_router.vendor','placement_router.vendor.fused_topk_router'}
        need(imports['entry_points']==[['placement_aware_router','placement_router.plugin:register']]
             and set(imports['module_paths'])==set(imports['module_sha256'])==names,'Actual CPU target entrypoint/module inventory differs')
        for name in names:
            filename=name.replace('.','/')+('/__init__.py' if name in ('placement_router','placement_router.vendor') else '.py')
            need(imports['module_paths'][name]==plan['site_root']+'/'+filename and imports['module_sha256'][name]==SOURCE[filename],'Actual CPU target source differs')
        process=spec['process'];identity=(process['pid'],process['start_ticks'])
        need(integer(process['pid'],1) and process['pid']==process['pgrp'] and identity not in seen_processes,'Native server pair not fresh/owned')
        seen_processes.add(identity)
        cleanup=launch['closure'][role]
        need(cleanup['identity']==process and cleanup['remaining_members']==[],'Owned native pair did not close')
        owned_members(launch,role)
        contract=bundle.json(remote_dir+'/'+role+'-contract.json')
        need(contract.get('complete') is True and contract['role']==role and contract['arm']==expected_arm and contract['ep']==4
             and contract['process']==process and contract['plan_sha256']==launch['plan_sha256']
             and contract['allocation']==plan['allocation'] and contract['source_commit']==COMMIT and contract['checkpoint_revision']==REVISION
             and contract['checkpoint_proof_sha256']==plan['files'][plan['checkpoint_proof']]
             and contract['source_proof_sha256']==plan['files'][plan['source_proof']] and contract['model_path']==plan['model']
             and contract['kv_connector']=='NixlConnector' and contract['kv_load_failure_policy']=='fail' and contract['prefix_caching'] is False
             and contract['native_runner_sources']==plan['native_runner_sources']
             and contract['runner_contract_sha256']==HELPERS['native_runner_contract.py']
             and contract['request_id_sources']==plan['request_id_sources']
             and contract['request_id_contract_sha256']==HELPERS['native_request_id.py'],
             'Owned ready contract model/source/transfer differs')
        log=bundle.checked_file(remote_dir+'/'+role+'-server.log',launch[role+'_server_log_sha256']).read_text(errors='replace')
        need('placement-router: routing left unchanged' not in log and ('placement-router: placement-aware routing active' in log)==(expected_arm!='stock')
             and launch[role+'_activation_contract'] is True,'Actual activation/fallback log differs')
        warnings[role]=[{'line':i+1,'text':line} for i,line in enumerate(log.splitlines()) if any(t in line.lower() for t in ('warn','error','compil','jit','placement-router'))]
    idle=launch['closure']['gpu_idle'];need(idle['returncode']==0 and not idle['stdout'].strip(),'Post-pair whole-host idle missing')
    if not instrumented:
        folder=bundle.resolve(remote_dir+'/launch.json').parent
        expected={r+'-'+suffix for r in ('prefill','decode') for suffix in ('launch.json','contract.json','server.log')} | {'launch.json'} | {f'{phase}{n:02d}'+suffix for phase in ('warmup','measured') for n in range(3) for suffix in ('','-native-metrics')}
        need({p.name for p in folder.iterdir()}==expected and not any(p.is_symlink() for p in folder.iterdir()),'Extra/missing clean timed pair artifacts')
    return warnings


def import_frozen(path,expected,name):
    need(sha(path)==expected,'Independent frozen helper changed')
    previous=list(sys.path);bytecode=sys.dont_write_bytecode;sys.path.insert(0,str(Path(path).parent));sys.dont_write_bytecode=True
    try:
        spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
    finally:sys.path[:]=previous;sys.dont_write_bytecode=bytecode


def recheck_proof(bundle,plan,launch,remote_dir):
    stored=bundle.json(launch['phase_audit']['path'],launch['phase_audit']['sha256'])
    need(stored['complete'] is True,'Instrumented handoff proof incomplete')
    for name,expected in stored['phase_file_sha256'].items():bundle.checked_file(name,expected)
    client=remote_dir+'/client';bundle.checked_file(client+'/result.json',stored['campaign_result_sha256'])
    # Native proof auditor checks all raw bodies, real NIXL,
    # receive/allocation, one-token decode schedule and stock monolithic diagnostic differences.
    module=import_frozen(HERE/'audit_hybrid.py',HELPERS['audit_hybrid.py'],'independent_frozen_hybrid_phase_audit')
    with tempfile.TemporaryDirectory(prefix='hybrid-proof-recheck-') as folder:
        output=Path(folder)/'audit.json'
        module.audit(SimpleNamespace(result=bundle.resolve(client+'/result.json').parent,prepared=bundle.resolve(plan['preflight_inputs']),
            prefill_probe=bundle.resolve(next(k for k in stored['phase_file_sha256'] if '/prefill-probe/' in k)).parent,
            decode_probe=bundle.resolve(next(k for k in stored['phase_file_sha256'] if '/decode-probe/' in k)).parent,output=output))
        actual=load(output)
    actual['phase_file_sha256']={name:actual['phase_file_sha256'][str(bundle.resolve(name))] for name in stored['phase_file_sha256']}
    need(actual==stored,'Independent unchanged native phase re-audit differs')
    for role in ('prefill','decode'):
        need({r['pid'] for r in actual['worker_identity'][role]} <= owned_members(launch,role)
             and set(actual['request_id_mapping_pids'][role]) <= owned_members(launch,role)
             and actual['exact_native_external_internal_mapping'] is True,
             'Actual native phase workers were outside the corresponding owned role process group')
    # Add every underlying preserved proof client file to the final inventory.
    result=bundle.json(client+'/result.json')
    bundle.checked_file(client+'/start.json',result['start_sha256'])
    for name,expected in {**result['raw_files_sha256'],**result['record_files_sha256'],**result.get('stock_reference_files_sha256',{})}.items():
        bundle.checked_file(client+'/'+name,expected)
    return stored


def prior_failure(bundle,reference):
    original=reference['original_path'];previous=bundle.json(reference['path'],reference['sha256'])
    special=original=='/work/results/hybrid-v2/proof/receipt.json'
    need(previous.get('complete') is False and previous.get('cleanup_complete') is True
         and (previous.get('cleanup_errors')==[] or (special and 'cleanup_errors' not in previous))
         and not previous.get('cleanup_error') and not previous.get('active_child')
         and (previous.get('finished_unix') or previous.get('finished_unix_ns')),
         'Preserved failed observer lacks terminal closure')
    if special:
        need(previous['mode']=='proof' and not previous['active_servers'] and len(previous['launches'])==1
             and previous['launches'][0]['complete'] is False
             and previous['emergency_closure']['gpu_idle']['returncode']==0
             and not previous['emergency_closure']['gpu_idle']['stdout'].strip(),'Failed native-ID proof cleanup differs')
        old=previous['launches'][0]
        need(old['receipt']=='/work/results/hybrid-v2/proof/sweep00-position0-stock/launch.json'
             and old['sha256']=='f38e7d01162c32c0ed678969a97e514abc07ac7ab910bbb12711e7d46316745f','Failed stock raw receipt identity differs')
        failed=bundle.json(str(recovery.prior_path(old['receipt'])),old['sha256'])
        need(failed['complete'] is False and failed['finalization_errors']==[]
             and all(failed['closure'][role]['remaining_members']==[] for role in ('prefill','decode')),
             'Failed stock native roles did not close')
    identity=previous.get('controller_process',previous.get('process'))
    need(identity and identity['pid']==identity['pgrp']==reference['controller_process']['pid']
         and str(identity['start_ticks'])==reference['controller_process']['start_ticks'],'Archived failed controller identity differs')
    return previous


def audit_plan(bundle,plan,expected_sha):
    need(plan['schema']=='real-nixl-hybrid-v6' and plan['source_commit']==COMMIT and plan['checkpoint_revision']==REVISION
         and plan['model_id']=='Qwen/Qwen3-30B-A3B-Instruct-2507','Frozen source/model identity differs')
    need(plan['prefill_url']=='http://127.0.0.1:8100' and plan['decode_url']=='http://127.0.0.1:8200'
         and finite(plan['gpu_util']) and 0<plan['gpu_util']<1,'Physical local endpoints/GPU memory fraction differ')
    need((plan['warmup_rounds'],plan['measured_rounds'],plan['output_tokens'],plan['logical_requests'],plan['concurrency'],plan['client_waves'])
         ==(3,3,128,256,64,4) and plan['timed_sweeps']==ORDERS and plan['quality_arms']==list(ARMS),'Frozen design differs')
    for name,expected in plan['files'].items():bundle.checked_file(name,expected)
    need(plan['prospective_policy']==prospective.POLICY and plan['prospective_admission']=={'files_sha256':prospective.FILES,'policy':prospective.POLICY},
         'V6 prospective qualification/baseline or failed diagnostic preservation differs')
    for name,digest in prospective.FILES.items():bundle.checked_file(name,digest)
    admission=bundle.json(prospective.ANALYSIS,prospective.FILES[prospective.ANALYSIS])
    need(admission['complete'] is True and admission['original_v5_campaign_complete'] is False
         and admission['performance_effects'] is None and admission['kv_valid_prefix_pairs']==384
         and admission['all_valid_prefix_sha256_equal'] is True and admission['all_full_prompt_sha256_equal_supplemental'] is True,
         'Complete reviewed prior KV diagnostic required; failed V5 cannot be promoted')
    compatibility=plan['native_accounting_sources']
    need(compatibility['sha256']==accounting.SOURCES and compatibility['root']==plan['native_runner_sources']['root']
         and set(compatibility['files'])==set(accounting.SOURCES),'Exact source-bound native cache/metric definitions required')
    for name,digest in accounting.SOURCES.items():
        need(compatibility['files'][name]==compatibility['root']+'/'+name and plan['files'][compatibility['files'][name]]==digest,
             'Native accounting installed path/plan binding differs')
        bundle.checked_file(compatibility['files'][name],digest)

    need(plan['request_id_sources']['sha256']==REQUEST_ID_SOURCES and set(plan['request_id_sources']['files'])==set(REQUEST_ID_SOURCES), 'Exact native API/UUID/env source map required')
    native=plan['native_runner_sources']
    for name,path in plan['request_id_sources']['files'].items():
        need(path==native['root']+'/'+name and plan['files'][path]==REQUEST_ID_SOURCES[name],'Native request ID path/plan binding differs')
        bundle.checked_file(path,REQUEST_ID_SOURCES[name])
    need(native['sha256']==NATIVE and set(native['files'])==set(NATIVE), 'Actual V2 native source inventory differs')
    for name,expected in NATIVE.items():
        need(native['files'][name]==native['root']+'/'+name and plan['files'][native['files'][name]]==expected,
             'Actual native V2 source path/plan binding differs')
    need({r['original_path']:r['sha256'] for r in plan['failed_prior_receipts']}==FAILED_SHAS,
         'Preserved failed original observer prerequisites differ')
    for reference in plan['failed_prior_receipts']:
        need(reference['path']==str(recovery.prior_path(reference['original_path'])) and reference['closure_scope']==recovery.CLOSURE,
             'Archived prior path/namespace closure differs')
        prior_failure(bundle,reference)
    for filename,expected in SOURCE.items():
        bundle.checked_file(plan['source_root']+'/src/'+filename,expected)
        bundle.checked_file(plan['site_root']+'/'+filename,expected)
    need(plan['files'][plan['preflight_inputs']]=='183888c4a3d0402ab97806d453449dbf740e445901b80ee3d5e156ffc7dd7017'
         and plan['files'][plan['quality_inputs']]=='6c3cdfb53702f5b1e07f8d38a9770b5da45b5d419b5ad81d5b9e81908c0cf63b',
         'Canonical training/full quality task inputs differ')
    for name,expected in HELPERS.items():need(plan['files'][plan['helper_dir']+'/'+name]==expected,'Frozen execution helper differs')
    need(plan['files'][plan['lifecycle_helper']]=='2b1832a73ec1d84ef16746c99a8ad94c82f4300ef7464fc4d2250e63b66b6db3','Frozen lifecycle differs')
    need(plan['server_commands']=={r:server_command(plan,r) for r in ('prefill','decode')},'Frozen physical commands differ')
    source=plan['source_identity'];need(source['commit']==COMMIT and source['proof_sha256']==plan['files'][plan['source_proof']]
        and source['files']=={kind+':src/'+n:d for n,d in SOURCE.items() for kind in ('public','installed')},'Source lineage proof map differs')
    deployment=bundle.json(plan['source_proof'],source['proof_sha256'])
    need(deployment['source_commit']==COMMIT and deployment['source_root']==plan['source_root'] and deployment['site_root']==plan['site_root'],'Deployment paths differ')
    allocation=bundle.json(plan['allocation_proof'],plan['files'][plan['allocation_proof']])
    binding=plan['recovery_binding']
    proof=bundle.json(binding['proof_path'],binding['proof_sha256'])
    try:expected_binding=recovery.verify_value(proof,allocation)
    except RuntimeError as error:raise ValueError(str(error)) from error
    need(binding=={'proof_path':binding['proof_path'],'proof_sha256':binding['proof_sha256'],**expected_binding}
         and plan['files'][binding['proof_path']]==binding['proof_sha256']
         and allocation==plan['allocation']
         and all(deployment['pod_identity'][k]==allocation[k] for k in ('namespace','pod','uid','node','image')),
         'Fresh H200 same-host recovery ownership differs')
    checkpoint=bundle.json(plan['checkpoint_proof'],plan['files'][plan['checkpoint_proof']])
    need(checkpoint['canonical_identity_verified'] is True and checkpoint['model_id']==plan['model_id'] and checkpoint['revision']==REVISION
         and checkpoint['local_model_path']==plan['model']
         and checkpoint['canonical_config_sha256']=='a1ee086a68d0cbfc87316da00ba4b8507bd1292978108e2496201a30a450f438'
         and len(checkpoint['safetensors'])==16
         and len({r['filename'] for r in checkpoint['safetensors']})==16 and checkpoint['weights_verified']==checkpoint['safetensors'],
         'Fresh complete checkpoint proof differs')
    need(plan['checkpoint_identity']['model_id']==plan['model_id'] and plan['checkpoint_identity']['revision']==REVISION
         and plan['checkpoint_identity']['proof_sha256']==plan['files'][plan['checkpoint_proof']]
         and set(plan['checkpoint_identity']['file_stat_signatures'])=={r['filename'] for r in checkpoint['safetensors']}
         and all(plan['checkpoint_identity']['file_stat_signatures'][r['filename']][0]==r['bytes'] for r in checkpoint['safetensors']),
         'Checkpoint frozen shard identities differ')
    runtime=bundle.json(plan['runtime_preflight'],plan['files'][plan['runtime_preflight']])
    need(runtime['complete'] is True and runtime['packages']['vllm'].split('+')[0]=='0.31.0' and runtime['packages']['nixl']=='1.4.1'
         and runtime['nixl_agent_constructed'] is True,'Exact native connector runtime gate differs')
    for record in runtime['module_files'].values():need(plan['files'][record['path']]==record['sha256'],'Actual native connector source not plan-bound')
    prerequisites=[(r,bundle.json(r['path'],r['sha256'])) for r in plan['prerequisite_receipts']]
    need(len(prerequisites)>=3 and all(v.get('complete') is True and v.get('cleanup_complete') is True for _,v in prerequisites),'Prior campaigns incomplete')
    perf=[(r,v) for r,v in prerequisites if len(v.get('launches',[]))==36];profiles=[(r,v) for r,v in prerequisites if 'main_guard' in v]
    need(len(perf)==len(profiles)==1,'Missing exact36timing/3profile prerequisite')
    pref,pre=perf[0];prof,profile=profiles[0];guard=profile['main_guard']
    profile_plan_path=str(PurePosixPath(prof['path']).parent/'plan.json')
    profile_plan=bundle.json(profile_plan_path,profile['plan_sha256'])
    need(profile_plan['schema']=='headline-prefill-profiles-v4', 'Require corrected actual-V2 profiles')
    need(guard['receipt_sha256']==pref['sha256'] and guard['complete_launches']==36 and guard['controller_original_exited'] is True
         and guard['main_lock_held'] is True and len(profile['launches'])==3 and all(r['complete'] for r in profile['launches']),
         'Profile/main ownership linkage differs')
    need(not plan['must_have_exited'],'No archived PID inference in replacement namespace')
    expected_prior_controllers=[{'pid':int(v['controller_process']['pid']),'start_ticks':str(v['controller_process']['start_ticks']),
                                'closure_scope':recovery.CLOSURE} for v in (pre,profile)]
    need(plan['recovered_prior_controllers']==expected_prior_controllers,'Archived prior controller scope differs')
    for reference,value in prerequisites:
        need(reference['path']==str(recovery.prior_path(reference['original_path'])) and reference['closure_scope']==recovery.CLOSURE,
             'Prior successful campaign physical namespace differs')
    for row in profile['launches']:
        launch=bundle.json(str(recovery.prior_path(row['receipt_file'])),row['receipt_sha256'])
        need(launch['complete'] is True and launch['arm']==row['arm'] and launch['plan_sha256']==profile['plan_sha256']
             and launch['finalization_errors']==[] and launch['client_cleanup']['remaining_members']==[]
             and launch['server_cleanup']['remaining_members']==[] and set(launch['actual_runner_startup_contract'])=={'0','1','2','3'},
             'Recovered profile launch source/closure differs')
    main_remote=str(PurePosixPath(pref['path']).parent.parent/'plan.json')
    main=bundle.json(main_remote,pre['plan_sha256']);original=bundle.json(main['prompt_file'],PROMPTS_SHA)
    need(len(original)==64 and [r['request_id'] for r in original]==list(range(64)) and all(len(r['token_ids'])==4096 for r in original),'Original synthetic corpus differs')
    prompts=[{'request_id':i*4+j,'token_ids':r['token_ids'][j*1024:(j+1)*1024]} for i,r in enumerate(original) for j in range(4)]
    need(bundle.json(plan['prompt_file'],plan['files'][plan['prompt_file']])==prompts,'Derived256x1024 workload differs')
    return prompts


def describe(values):
    need(values and all(finite(x) and x>=0 for x in values),'Invalid latency vector')
    ordered=sorted(values)
    return {'n':len(values),'mean':statistics.fmean(values),'p50':ordered[math.ceil(.5*len(values))-1],
            'p95':ordered[math.ceil(.95*len(values))-1],'raw':values}


def audit_campaign(root,expected_plan_sha,bindings,statistics_path,protocol_path):
    root=Path(root).resolve();plan_path=local_file(root,'plan.json');need(sha(plan_path)==expected_plan_sha,'Operational plan SHA differs')
    need(sha(protocol_path)==PROTOCOL_SHA,'Frozen hybrid analysis protocol changed')
    stats=import_frozen(statistics_path,STATS_SHA,'independent_paired_hybrid_statistics')
    plan=load(plan_path);bundle=Bundle(root,plan,bindings);prompts=audit_plan(bundle,plan,expected_plan_sha)
    proof_remote=plan['output']+'/proof';proof=bundle.json(proof_remote+'/receipt.json')
    need(proof==bundle.json(proof_remote+'/status.json') and proof['complete'] is True and proof['cleanup_complete'] is True
         and proof['mode']=='proof' and proof['plan_sha256']==expected_plan_sha and len(proof['launches'])==3 and 'error' not in proof,
         'Native handoff proof campaign incomplete')
    need(not proof.get('active_servers') and proof['emergency_closure']['gpu_idle']['returncode']==0
         and not proof['emergency_closure']['gpu_idle']['stdout'].strip(),'Proof final owned closure/idle missing')
    processes=set();native_proofs=[];expected_proofs=set()
    for position,(item,arm) in enumerate(zip(proof['launches'],ARMS)):
        directory=proof_remote+f'/sweep00-position{position}-{arm}'
        need(item['complete'] is True and (item['sweep'],item['position'],item['arm'])==(0,position,arm)
             and item['receipt']==directory+'/launch.json','Native handoff proof order differs')
        launch=bundle.json(item['receipt'],item['sha256']);expected_proofs.add(bundle.resolve(item['receipt']));need(launch['plan_sha256']==expected_plan_sha,'Native proof plan differs')
        pair_contract(bundle,plan,launch,directory,arm,True,processes)
        native_proofs.append(recheck_proof(bundle,plan,launch,directory))
    need(set(root.glob('proof/sweep*/launch.json'))==expected_proofs,'Extra/missing native proof attempts')
    timed_remote=plan['output']+'/timed';receipt=bundle.json(timed_remote+'/receipt.json')
    need(receipt==bundle.json(timed_remote+'/status.json') and receipt['complete'] is True and receipt['cleanup_complete'] is True
         and receipt['mode']=='timed' and receipt['plan_sha256']==expected_plan_sha and len(receipt['launches'])==9
         and receipt['proof_receipt_sha256']==sha(bundle.resolve(proof_remote+'/receipt.json')) and 'error' not in receipt
         and not receipt.get('active_servers') and receipt['emergency_closure']['gpu_idle']['returncode']==0
         and not receipt['emergency_closure']['gpu_idle']['stdout'].strip(),'Timed fixed nine-pair campaign incomplete')
    normalized=[];descriptive=[];seen=set();warnings=[];round_bounds=[];expected_launches=set()
    for index,item in enumerate(receipt['launches']):
        sweep,position=divmod(index,3);arm=ORDERS[sweep][position];launch_id=f'sweep{sweep:02d}-position{position}-{arm}'
        directory=timed_remote+'/'+launch_id
        need(item['complete'] is True and (item['sweep'],item['position'],item['arm'])==(sweep,position,arm)
             and item['receipt']==directory+'/launch.json','Timed pair order/path differs')
        launch=bundle.json(item['receipt'],item['sha256']);expected_launches.add(bundle.resolve(item['receipt']))
        need((launch['sweep'],launch['position'],launch['arm'])==(sweep,position,arm) and launch['plan_sha256']==expected_plan_sha,'Timed native pair identity differs')
        warnings.append({'launch_id':launch_id,'warnings':pair_contract(bundle,plan,launch,directory,arm,False,processes)})
        spec=[('warmup',n) for n in range(3)]+[('measured',n) for n in range(3)]
        need(len(launch['rounds'])==6,'Fixed3+3 cohort inventory differs')
        seconds=[];latency=[]
        for reference,(phase,number) in zip(launch['rounds'],spec):
            remote=directory+f'/{phase}{number:02d}/round.json'
            elapsed,values=audit_round(bundle,reference,remote,phase,number,prompts,launch_id,seen);seconds.append(elapsed)
            row=bundle.json(remote);round_bounds.append((row['cohort']['start_ns'],row['finished_ns']))
            if phase=='measured':latency.extend(values)
        close(launch['mean_measured_seconds'],statistics.fmean(seconds[3:]),'Launch batch estimator differs')
        normalized.append({'sweep':sweep,'position':position,'arm':arm,'launch_id':launch_id,'warmup_seconds':seconds[:3],'measurement_seconds':seconds[3:]})
        descriptive.append({'sweep':sweep,'arm':arm,'launch_id':launch_id,'metrics':{k:describe([r[k] for r in latency])
            for k in ('own_ttft_ms','cohort_ttft_ms','latency_ms','exploratory_transport_tpot_ms')}})
    need(set(root.glob('timed/sweep*/launch.json'))==expected_launches,'Extra/missing timed native pair attempts')
    need(all(round_bounds[i][1]<=round_bounds[i+1][0] for i in range(len(round_bounds)-1)),'Retained cohorts overlap or reordered')
    result=stats.analyze(normalized);need(result['complete'] is True,'Independent statistics rejected complete normalized matrix')
    differences=[]
    for sweep in range(3):
        rows={r['arm']:r['metrics'] for r in descriptive if r['sweep']==sweep}
        for arm in ARMS[1:]:differences.append({'sweep':sweep,'arm':arm,'treatment_minus_stock':{metric:{endpoint:rows[arm][metric][endpoint]-rows['stock'][metric][endpoint]
            for endpoint in ('mean','p50','p95')} for metric in rows[arm]}})
    result.update(raw_integrity='PASS complete9pairs,54cohorts,13824logicalrequests; every producerJSON/consumerSSE independently rescored',
      normalized_records=normalized,descriptive_latency=descriptive,within_sweep_latency_differences=differences,
      timing_evidence_limits=['HTTPheaders/content-type are runtime source-checked but not independently archived by frozen timed helper; saved200statuses and rawSSE are verified',
       'Client perf-counter arrival timestamps are recorded observations; rawchunk content/order, formulas, interval bounds and sequentialwaves rechecked, not server intertoken times',
       'Actual NIXL scheduling/stockdecode provenance comes from separate instrumented proofs; timed rows require source-correct cached0, off-clock exact per-engine native NIXL transfer count/bytes, exact returned transfer params and clean source-bound server launches',
       'TPOT is exploratory transport-chunk arrival statistic with coalescing; no pertoken or SLO inference'],
      hardware=plan['allocation'],hardware_scope=plan['hardware_scope'],recovery_binding=plan['recovery_binding'],
      native_handoff_proofs=native_proofs,prospective_policy=plan['prospective_policy'],warnings_and_compile_lines=warnings,frozen_plan_sha256=expected_plan_sha,
      timed_receipt_sha256=sha(bundle.resolve(timed_remote+'/receipt.json')),raw_files_verified=len(bundle.checked),verified_artifacts=bundle.checked,
      audit_source_sha256=sha(__file__),statistics_source_sha256=sha(statistics_path),analysis_protocol_sha256=sha(protocol_path),
      bindings_sha256=sha(bindings) if bindings else None)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--bindings',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--statistics',type=Path,default=HERE.parents[1]/'audit/paired_hybrid_statistics_v1.py')
    parser.add_argument('--protocol',type=Path,default=HERE.parents[1]/'audit/frozen-hybrid-analysis-protocol-v1.json')
    args=parser.parse_args()
    try:result=audit_campaign(args.campaign_root,args.plan_sha256,args.bindings,args.statistics,args.protocol)
    except BaseException:
        result={'complete':False,'effects':None,'error':traceback.format_exc(),'interpretation':'Incomplete/invalid raw matrix; no partial effect or selected sweep substituted'}
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    raise SystemExit(0 if result['complete'] else 2)


if __name__=='__main__':main()
