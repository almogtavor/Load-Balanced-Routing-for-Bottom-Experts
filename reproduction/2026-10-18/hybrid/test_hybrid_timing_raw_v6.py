"""Full256-request two-leg raw round and adversarial wire/clock fixtures."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import audit_hybrid_timing_raw_v6 as audit


class RawTests(unittest.TestCase):
    def save(self,path,value):
        path.write_bytes(audit.encoded(value)+b'\n')

    def test_owned_role_members_unique_live_and_original_group(self):
        launch={'closure':{'prefill':{'identity':{'pid':100,'pgrp':100}}},
                'owned_group_members_before_cleanup':{'prefill':[{'pid':101,'pgrp':100,'state':'S','start_ticks':'12'}]}}
        self.assertEqual(audit.owned_members(launch,'prefill'),{101})
        for field,value in (('pgrp',200),('state','Z'),('start_ticks','unknown')):
            changed=copy.deepcopy(launch);changed['owned_group_members_before_cleanup']['prefill'][0][field]=value
            with self.assertRaises(ValueError):audit.owned_members(changed,'prefill')
        changed=copy.deepcopy(launch);changed['owned_group_members_before_cleanup']['prefill']*=2
        with self.assertRaises(ValueError):audit.owned_members(changed,'prefill')

    def test_exact_failed_native_proof_schema_replays_without_invented_cleanup_errors(self):
        here=Path(__file__).resolve().parent
        proof=json.loads((here/'fixtures/hybrid-v3-failed-proof-receipt.json').read_bytes())
        failed=json.loads((here/'fixtures/hybrid-v3-failed-stock-launch.json').read_bytes())
        self.assertNotIn('cleanup_errors',proof)
        original='/work/results/hybrid-v2/proof/receipt.json'
        reference={'original_path':original,'path':str(audit.recovery.prior_path(original)),
                   'sha256':audit.FAILED_SHAS[original],
                   'controller_process':{'pid':40768,'start_ticks':'6385036'}}
        class Bundle:
            def json(self,path,digest):
                if path==reference['path']:
                    self_proof=proof
                    self.assert_digest=digest
                    return self_proof
                assert path==str(audit.recovery.prior_path(proof['launches'][0]['receipt']))
                assert digest==proof['launches'][0]['sha256']
                return failed
        bundle=Bundle();audit.prior_failure(bundle,reference)
        self.assertEqual(bundle.assert_digest,reference['sha256'])
        for change in ('missing_idle','live_role','wrong_receipt','wrong_identity'):
            pcopy=copy.deepcopy(proof);fcopy=copy.deepcopy(failed)
            if change=='missing_idle':proof['emergency_closure']['gpu_idle']['stdout']='still running'
            if change=='live_role':failed['closure']['decode']['remaining_members']=[{'pid':1}]
            if change=='wrong_receipt':proof['launches'][0]['receipt']='/work/results/other/launch.json'
            if change=='wrong_identity':proof['controller_process']['start_ticks']='different'
            with self.assertRaises((ValueError,AssertionError)):audit.prior_failure(bundle,reference)
            proof.clear();proof.update(pcopy);failed.clear();failed.update(fcopy)

    def fixture(self,root):
        remote='/remote/hybrid/timed/sweep00-position0-stock/warmup00/round.json'
        folder=root/'timed/sweep00-position0-stock/warmup00';folder.mkdir(parents=True)
        prompts=[{'request_id':i,'token_ids':[i+1]*1024} for i in range(256)]
        records=[];start=10000;finish=14000
        for i,row in enumerate(prompts):
            request_id=f'sweep00-position0-stock-warmup-0-{i:03d}'
            p=audit.body(row['token_ids'],request_id,1)
            p['kv_transfer_params']={'do_remote_decode':True,'do_remote_prefill':False,'remote_engine_id':None,'remote_block_ids':None,'remote_host':None,'remote_port':None}
            params={'do_remote_prefill':True,'do_remote_decode':False,'remote_engine_id':'engine','remote_request_id':f'cmpl-{request_id}-0-a8477bf8',
                    'remote_host':'127.0.0.1','remote_port':5600+i%4,'remote_block_ids':[[i+1]],'remote_num_tokens':1024,'tp_size':1,'pp_size':1,'transfer_mode':'pull','remote_prefill_cached_tokens':0}
            producer={'id':f'cmpl-{request_id}','choices':[{'index':0,'prompt_token_ids':row['token_ids'],'token_ids':[8],'text':'discard','finish_reason':'length'}],
                      'usage':{'prompt_tokens':1024,'completion_tokens':1,'total_tokens':1025},'kv_transfer_params':params}
            d=audit.body(row['token_ids'],request_id);d.update(stream=True,stream_options={'include_usage':True},kv_transfer_params=params)
            usage={'prompt_tokens':1024,'completion_tokens':128,'total_tokens':1152,'prompt_tokens_details':{'cached_tokens':0}}
            chunks=[{'token_ids':[3]*128,'text':'unit'}]
            item={'id':f'cmpl-{request_id}','choices':[{'index':0,'prompt_token_ids':row['token_ids'],**chunks[0],'finish_reason':'length'}]}
            events=[item,{'id':f'cmpl-{request_id}','choices':[],'usage':usage}]
            sse=b''.join(b'data: '+audit.encoded(x)+b'\n\n' for x in events)+b'data: [DONE]\n\n'
            raw={'prefill.request.json':audit.encoded(p),'prefill.response.json':audit.encoded(producer),'decode.request.json':audit.encoded(d),'decode.sse':sse}
            files={}
            for suffix,data in raw.items():
                name=f'request{i:03d}.'+suffix;(folder/name).write_bytes(data);files[name]=hashlib.sha256(data).hexdigest()
            began=start+(i//64)*1000+i%64
            records.append({'request_index':i,'complete':True,'done':True,'producer_status':200,'consumer_status':200,'files_sha256':files,
                            'transfer_params':params,'chunks':[{'arrival_ns':began+200,**chunks[0]}],'usage':usage,'cached_prompt_tokens':0,
                            'finish_reason':'length','prompt_ids_verified':True,'request_start_ns':began,'producer_finished_ns':began+100,
                            'done_ns':began+300,'finished_ns':began+400,'cohort_start_ns':start,'ttft_ms':.0002,'cohort_ttft_ms':(began+200-start)/1e6,'latency_ms':.0003})
        value={'complete':True,'phase':'warmup','round':0,'logical_requests':256,'concurrency':64,'client_waves':4,'input_tokens':262144,'output_tokens':32768,
               'cohort':{'start_ns':start,'start_unix_ns':1000000},'finished_ns':finish,'elapsed_seconds':.000004,
               'application_tokens_per_second':294912/.000004,'generation':'forced128ignoreEOStrue; speed workload only','records':records}
        self.save(folder/'application-round.json',value)
        self.save(folder/'round.json',value)
        ref={'phase':'warmup','number':0,'path':remote,'sha256':audit.sha(folder/'round.json'),'elapsed_seconds':.000004}
        bundle=audit.Bundle(root,{'output':'/remote/hybrid'},None)
        return bundle,ref,remote,prompts,value,folder

    def run_fixture(self,fixture):
        bundle,ref,remote,prompts,value,folder=fixture
        self.save(folder/'round.json',value);ref['sha256']=audit.sha(folder/'round.json')
        # The native wire/raw counter module is independently exercised by
        # test_accounting_v6 full off-clock fixtures; this inherited test isolates
        # all256 producer/consumer body/SSE/clock validations.
        with patch.object(audit.native_wire,'audit',return_value={'fixture_scope':'wire-only'}):
            return audit.audit_round(bundle,ref,remote,'warmup',0,prompts,'sweep00-position0-stock',set())

    def edit_wire(self,fixture,suffix,edit):
        _,_,_,_,value,folder=fixture
        path=folder/('request000.'+suffix)
        if suffix=='decode.sse':
            items=[json.loads(block[6:]) for block in path.read_bytes().split(b'\n\n') if block and block!=b'data: [DONE]']
            edit(items);path.write_bytes(b''.join(b'data: '+audit.encoded(x)+b'\n\n' for x in items)+b'data: [DONE]\n\n')
        else:
            body=json.loads(path.read_bytes());edit(body);path.write_bytes(audit.encoded(body))
        value['records'][0]['files_sha256'][path.name]=audit.sha(path)

    def test_all256_actual_raw_pairs_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture=self.fixture(Path(folder));seconds,latency=self.run_fixture(fixture)
            self.assertEqual(seconds,.000004);self.assertEqual(len(latency),256)
            self.assertEqual(len(fixture[0].checked),1025)
            self.assertEqual(latency[0]['own_ttft_ms'],.0002)

    def test_producer_id_and_exact_wire_controls_cannot_be_blessed_by_new_hash(self):
        for suffix,edit in [('prefill.response.json',lambda x:x['kv_transfer_params'].update(remote_request_id='invented-0')),
                            ('prefill.request.json',lambda x:x.update(ignore_eos=False)),
                            ('decode.request.json',lambda x:x.update(prompt=[99]*1024))]:
            with self.subTest(suffix=suffix),tempfile.TemporaryDirectory() as folder:
                fixture=self.fixture(Path(folder));self.edit_wire(fixture,suffix,edit)
                with self.assertRaises(ValueError):self.run_fixture(fixture)

    def test_native_cache_count_output_count_and_duplicate_usage_rejected(self):
        mutations=[lambda x:x[1]['usage']['prompt_tokens_details'].update(cached_tokens=1023),
                   lambda x:x[0]['choices'][0].update(token_ids=[3]*127),
                   lambda x:x.append(copy.deepcopy(x[1]))]
        for edit in mutations:
            with tempfile.TemporaryDirectory() as folder:
                fixture=self.fixture(Path(folder));self.edit_wire(fixture,'decode.sse',edit)
                with self.assertRaises(ValueError):self.run_fixture(fixture)

    def test_finish_only_optional_ids_are_valid_and_mixed_ids_fail(self):
        prompt=[1]*1024
        first={'id':'d','choices':[{'index':0,'prompt_token_ids':prompt,'token_ids':[3]*128,'text':'x'}]}
        end={'id':'d','choices':[{'index':0,'finish_reason':'length','text':''}]}
        usage={'id':'d','choices':[],'usage':{'prompt_tokens':1024,'completion_tokens':128,'total_tokens':1152,'prompt_tokens_details':{'cached_tokens':0}}}
        def wire(items):return b''.join(b'data: '+audit.encoded(x)+b'\n\n' for x in items)+b'data: [DONE]\n\n'
        self.assertEqual(len(audit.parse_sse(wire([first,end,usage]),prompt)['chunks']),1)
        usage['id']='changed'
        with self.assertRaises(ValueError):audit.parse_sse(wire([first,end,usage]),prompt)

    def test_usage_before_finish_content_after_finish_and_usage_on_content_rejected(self):
        prompt=[1]*1024
        content={'id':'d','choices':[{'index':0,'prompt_token_ids':prompt,'token_ids':[3]*128,'text':'x','finish_reason':'length'}]}
        usage={'id':'d','choices':[],'usage':{'prompt_tokens':1024,'completion_tokens':128,'total_tokens':1152,'prompt_tokens_details':{'cached_tokens':0}}}
        variants=[[usage,content],[content,{'id':'d','choices':[{'index':0,'token_ids':[9],'text':'late'}]},usage],
                  [{**content,'usage':usage['usage']}]]
        for items in variants:
            with self.subTest(items=items):
                wire=b''.join(b'data: '+audit.encoded(x)+b'\n\n' for x in items)+b'data: [DONE]\n\n'
                with self.assertRaises(ValueError):audit.parse_sse(wire,prompt)

    def test_wave_overlap_and_rate_doublecount_rejected(self):
        for mode in ('wave','rate'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as folder:
                fixture=self.fixture(Path(folder));value=fixture[4]
                if mode=='rate':value['application_tokens_per_second']*=2
                else:
                    row=value['records'][64];row.update(request_start_ns=10001,producer_finished_ns=10101,done_ns=10301,finished_ns=10401)
                    row['chunks'][0]['arrival_ns']=10201;row['cohort_ttft_ms']=.000201
                with self.assertRaises(ValueError):self.run_fixture(fixture)

    def test_extra_raw_file_partial_sse_and_changed_record_chunks_rejected(self):
        for mode in ('extra','partial','chunks'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as folder:
                fixture=self.fixture(Path(folder));folder=fixture[5]
                if mode=='extra':(folder/'unseen.sse').write_text('extra')
                elif mode=='chunks':fixture[4]['records'][0]['chunks'][0]['token_ids'][0]=99
                else:
                    p=folder/'request000.decode.sse';p.write_bytes(p.read_bytes().replace(b'data: [DONE]\n\n',b''))
                    fixture[4]['records'][0]['files_sha256'][p.name]=audit.sha(p)
                with self.assertRaises(ValueError):self.run_fixture(fixture)

    def campaign_fixture(self,root):
        plan={'schema':'real-nixl-hybrid-v4','output':'/remote/hybrid','vllm':'/runtime/vllm','model':'/model','gpu_util':.88,
              'native_runner_sources':{'root':'/native','sha256':audit.NATIVE,'files':{n:'/native/'+n for n in audit.NATIVE}},
              'request_id_sources':{'sha256':audit.REQUEST_ID_SOURCES,'files':{n:'/native/'+n for n in audit.REQUEST_ID_SOURCES}},
              'site_root':'/site','phase_probe':'/helpers/hybrid_phase_probe.py','allocation':{'gpu_count':8},
              'hardware_scope':'Matched same-host H200','recovery_binding':{'test':'synthetic'},'checkpoint_proof':'/checkpoint','source_proof':'/source','files':{'/checkpoint':'checkpoint','/source':'source'}}
        plan['prospective_policy']=audit.prospective.POLICY
        plan['server_commands']={r:audit.server_command(plan,r) for r in ('prefill','decode')}
        self.save(root/'plan.json',plan);plan_sha=audit.sha(root/'plan.json');next_pid=[1000];clock=[100000]
        def pair(mode,sweep,position,arm):
            launch_id=f'sweep{sweep:02d}-position{position}-{arm}';directory=root/mode/launch_id;directory.mkdir(parents=True)
            remote=plan['output']+'/'+mode+'/'+launch_id
            launch={'complete':True,'workload_complete':True,'finalization_errors':[],'sweep':sweep,'position':position,'arm':arm,
                    'plan_sha256':plan_sha,'closure':{'gpu_idle':{'returncode':0,'stdout':''}},'rounds':[],
                    'owned_group_members_before_cleanup':{}}
            for role in ('prefill','decode'):
                expected_arm=arm if role=='prefill' else 'stock';next_pid[0]+=1;pid=next_pid[0]
                process={'pid':pid,'pgrp':pid,'start_ticks':str(pid)}
                env={'PYTHONPATH':'/site','VLLM_PLUGINS':'placement_aware_router','OMP_NUM_THREADS':'1','NVSHMEM_QP_DEPTH':'4096',
                     'CUDA_VISIBLE_DEVICES':'0,1,2,3' if role=='prefill' else '4,5,6,7','VLLM_NIXL_SIDE_CHANNEL_HOST':'127.0.0.1',
                     'VLLM_NIXL_SIDE_CHANNEL_PORT':'5600' if role=='prefill' else '5610','UCX_RCACHE_MAX_UNRELEASED':'1024'}
                if expected_arm!='stock':env.update(PLACEMENT_KEEP='6' if arm=='keep6p8' else '4',PLACEMENT_LOAD_PENALTY='8')
                if mode=='proof':env.update(PLACEMENT_PROBE=plan['phase_probe'],HYBRID_PROBE_DIR=remote+'/'+role+'-probe',HYBRID_ROLE=role,HYBRID_ARM=expected_arm)
                names=('placement_router','placement_router.plugin','placement_router.kernel','placement_router.vendor','placement_router.vendor.fused_topk_router')
                filenames={n:n.replace('.','/')+('/__init__.py' if n in ('placement_router','placement_router.vendor') else '.py') for n in names}
                imports={'entry_points':[['placement_aware_router','placement_router.plugin:register']],
                         'module_paths':{n:'/site/'+f for n,f in filenames.items()},'module_sha256':{n:audit.SOURCE[f] for n,f in filenames.items()}}
                self.save(directory/(role+'-launch.json'),{'role':role,'arm':expected_arm,'command':plan['server_commands'][role],
                          'environment':env,'process':process,'cpu_import_proof':imports})
                self.save(directory/(role+'-contract.json'),{'complete':True,'role':role,'arm':expected_arm,'ep':4,'process':process,
                          'plan_sha256':plan_sha,'allocation':plan['allocation'],'source_commit':audit.COMMIT,'checkpoint_revision':audit.REVISION,
                          'checkpoint_proof_sha256':'checkpoint','source_proof_sha256':'source','model_path':'/model','kv_connector':'NixlConnector',
                          'kv_load_failure_policy':'fail','prefix_caching':False,'native_runner_sources':plan['native_runner_sources'],
                          'runner_contract_sha256':audit.HELPERS['native_runner_contract.py'],
                          'request_id_sources':plan['request_id_sources'],'request_id_contract_sha256':audit.HELPERS['native_request_id.py']})
                log=directory/(role+'-server.log');log.write_text('placement-router: placement-aware routing active\n' if expected_arm!='stock' else 'native\n')
                launch[role+'_server_log_sha256']=audit.sha(log);launch[role+'_activation_contract']=True
                launch['closure'][role]={'identity':process,'remaining_members':[]}
                launch['owned_group_members_before_cleanup'][role]=[{**process,'state':'S'}]
            if mode=='timed':
                duration={'stock':10.,'keep6p8':9.,'keep4p8':8.}[arm]
                for phase in ('warmup','measured'):
                    for number in range(3):
                        folder=directory/f'{phase}{number:02d}';folder.mkdir();folder.with_name(folder.name+'-native-metrics').mkdir();clock[0]+=int(20e9)
                        row={'cohort':{'start_ns':clock[0]},'finished_ns':clock[0]+int(duration*1e9),'elapsed_seconds':duration}
                        self.save(folder/'round.json',row)
                        launch['rounds'].append({'phase':phase,'number':number,'path':remote+f'/{phase}{number:02d}/round.json',
                                                'sha256':audit.sha(folder/'round.json'),'elapsed_seconds':duration})
                launch['mean_measured_seconds']=duration
            self.save(directory/'launch.json',launch)
            return {'complete':True,'sweep':sweep,'position':position,'arm':arm,'receipt':remote+'/launch.json','sha256':audit.sha(directory/'launch.json')}
        proof={'complete':True,'cleanup_complete':True,'mode':'proof','plan_sha256':plan_sha,'launches':[],
               'emergency_closure':{'gpu_idle':{'returncode':0,'stdout':''}}}
        for position,arm in enumerate(audit.ARMS):proof['launches'].append(pair('proof',0,position,arm))
        self.save(root/'proof/receipt.json',proof);self.save(root/'proof/status.json',proof)
        timed={'complete':True,'cleanup_complete':True,'mode':'timed','plan_sha256':plan_sha,'launches':[],
               'proof_receipt_sha256':audit.sha(root/'proof/receipt.json'),'emergency_closure':{'gpu_idle':{'returncode':0,'stdout':''}}}
        for sweep,order in enumerate(audit.ORDERS):
            for position,arm in enumerate(order):timed['launches'].append(pair('timed',sweep,position,arm))
        self.save(root/'timed/receipt.json',timed);self.save(root/'timed/status.json',timed)
        return plan_sha

    def campaign_run(self,root,plan_sha):
        def round_dispatch(bundle,reference,*args):
            row=bundle.json(reference['path'])
            values=[dict(own_ttft_ms=1.,cohort_ttft_ms=2.,latency_ms=3.,exploratory_transport_tpot_ms=0.) for _ in range(256)]
            return row['elapsed_seconds'],values
        # Full wire parsing has a separate complete256request integration test;
        # this fixture tests all9pair metadata/closure/design/statistics dispatch.
        with patch.object(audit,'audit_plan',return_value=[]),patch.object(audit,'recheck_proof',return_value={'complete':True}),patch.object(audit,'audit_round',side_effect=round_dispatch) as rounds:
            result=audit.audit_campaign(root,plan_sha,None,(Path(audit.__file__).parents[4]/'runs/paper-followup-20261018')/'audit/paired_hybrid_statistics_v1.py',
                                       (Path(audit.__file__).parents[4]/'runs/paper-followup-20261018')/'audit/frozen-hybrid-analysis-protocol-v1.json')
            self.assertEqual(rounds.call_count,54)
            return result

    def test_fixed_nine_pair_metadata_closure_and_statistics_dispatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);plan_sha=self.campaign_fixture(root);value=self.campaign_run(root,plan_sha)
            self.assertTrue(value['complete']);self.assertEqual(len(value['normalized_records']),9)
            self.assertEqual([r['metrics']['own_ttft_ms']['n'] for r in value['descriptive_latency']],[768]*9)
            self.assertAlmostEqual(value['effects']['keep6p8_vs_stock']['throughput_change_percent'],100*(10/9-1))

    def test_instrumented_clean_pair_or_extra_attempt_rejects_whole_matrix(self):
        for mode in ('instrumented','extra'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);plan_sha=self.campaign_fixture(root)
                if mode=='instrumented':
                    path=root/'timed/sweep00-position0-stock/prefill-launch.json';row=json.loads(path.read_text());row['environment']['PLACEMENT_PROBE']='/forbidden.py';self.save(path,row)
                else:
                    p=root/'timed/sweep-extra';p.mkdir();(p/'launch.json').write_text('{}')
                with self.assertRaises(ValueError):self.campaign_run(root,plan_sha)

    def test_cli_failure_never_substitutes_partial_effect(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'plan.json').write_text('{}');bindings=root/'bindings.json';bindings.write_text('{}');output=root/'analysis.json'
            result=subprocess.run([sys.executable,str(Path(audit.__file__)),'--campaign-root',str(root),'--plan-sha256','wrong',
                                   '--bindings',str(bindings),'--output',str(output)],capture_output=True)
            self.assertEqual(result.returncode,2);value=json.loads(output.read_text());self.assertFalse(value['complete']);self.assertIsNone(value['effects'])


if __name__=='__main__':unittest.main()
