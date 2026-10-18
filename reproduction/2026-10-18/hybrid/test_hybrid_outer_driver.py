import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import hybrid_outer_driver as h


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class Tests(unittest.TestCase):
    def lib(self):
        return SimpleNamespace(process_record=lambda p:{'pid':p,'pgrp':p,'state':'R','start_ticks':'1'} if p==h.os.getpid() else None,
            save=save,gpu_idle=lambda **k:{'returncode':0,'stdout':''},group_members=lambda p:[])

    def upstream(self,root,final=True,**overrides):
        perf=root/'perf.json';save(perf,{'complete':True})
        launches=[]
        for arm in h.ARMS:
            path=root/(arm+'.json')
            save(path,{'complete':True,'plan_sha256':h.PROFILE_SHA,'arm':arm,'finalization_errors':[],
                'client_cleanup':{'remaining_members':[]},'server_cleanup':{'remaining_members':[]},
                'actual_runner_startup_contract':{str(i):{} for i in range(4)}})
            launches.append({'arm':arm,'complete':True,'receipt_file':str(path),'receipt_sha256':h.sha(path)})
        value={'controller_process':{'pid':38261,'pgrp':38261,'start_ticks':'6345502'},'plan_sha256':h.PROFILE_SHA,
            'complete':True,'cleanup_complete':True,'cleanup_errors':[],'finished_unix_ns':1,'launches':launches,
            'main_guard':{'receipt_sha256':h.sha(perf),'complete_launches':36,'controller_original_exited':True,'main_lock_held':True},**overrides}
        path=root/('receipt.json' if final else 'status.json');save(path,value)
        return path,perf

    def test_exact_stage_sequence_and_primary_plan_flags(self):
        commands=h.stage_commands()
        self.assertEqual([s for s,c in commands],['plan','proof','timed','quality'])
        plan=commands[0][1]
        self.assertNotIn('--must-have-exited',plan)
        self.assertIn('--recovery-proof-sha256',plan)
        self.assertEqual(plan[plan.index('--stop-unix')+1],'1792349474')
        self.assertEqual(plan[plan.index('--preflight-inputs')+1],'/work/results/followup-prepared-v1/preflight.json')

    def test_archived_profile_never_checks_replacement_pid(self):
        with tempfile.TemporaryDirectory() as folder:
            path,profiles=self.upstream(Path(folder))
            real_sha=h.sha
            lib=self.lib();lib.process_record=lambda pid:self.fail("No archived PID lookup on replacement")
            with patch.object(h,'UPSTREAM',path),patch.object(h.recovery,'prior_path',side_effect=lambda p:profiles if p=='/work/results/prefill-e2e-v1/timed/receipt.json' else Path(p)):
                self.assertTrue(h.upstream_ready(lib))

    def test_archived_partial_upstream_never_waits_for_pid_collision(self):
        with tempfile.TemporaryDirectory() as folder:
            path,_=self.upstream(Path(folder),complete=False,cleanup_complete=False,finished_unix_ns=None)
            lib=self.lib();lib.process_record=lambda pid:self.fail("No archived PID lookup")
            with patch.object(h,'UPSTREAM',path),self.assertRaisesRegex(RuntimeError,'incomplete'):
                h.upstream_ready(lib)

    def test_upstream_failure_rejects(self):
        with tempfile.TemporaryDirectory() as folder:
            path,_=self.upstream(Path(folder),error='original capture failed')
            with patch.object(h,'UPSTREAM',path),self.assertRaisesRegex(RuntimeError,'V2 profile failed'):
                h.upstream_ready(self.lib())

    def test_reused_pid_is_never_signaled(self):
        lib=self.lib();lib.process_record=lambda p:{'pid':77,'start_ticks':'different'}
        with patch.object(h.os,'killpg') as kill,self.assertRaisesRegex(RuntimeError,'PID reused'):
            h.cleanup_owned(SimpleNamespace(pid=77),{'pid':77,'pgrp':77,'start_ticks':'old'},lib)
        kill.assert_not_called()

    def test_owned_cleanup_uses720second_grace(self):
        lib=self.lib();child=SimpleNamespace(pid=77,poll=lambda:0,wait=lambda timeout:0)
        result=h.cleanup_owned(child,{'pid':77,'pgrp':77,'start_ticks':'old'},lib)
        self.assertEqual(result['grace_seconds'],720)
        self.assertEqual(result['remaining_members'],[])

    def test_failed_stage_start_keeps_original_error_and_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);instance=object.__new__(h.Outer)
            instance.lib=self.lib();instance.state={'stages':[]};instance.child=instance.child_identity=None
            instance.validate=lambda **kw:None;instance.save=lambda:None
            with patch.object(h,'OUTER',root),patch.object(h.subprocess,'Popen',side_effect=OSError('original native child start failed')):
                with self.assertRaisesRegex(RuntimeError,'Hybrid stage failed'):instance.run_stage('proof',['no-start'])
            result=h.read(root/'proof-receipt.json')
            self.assertFalse(result['complete']);self.assertIn('original native child start failed',result['error'])

    def test_upstream_failure_final_receipt_never_queries_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);lib=self.lib();queries=[];lib.gpu_idle=lambda **k:queries.append(k)
            request=root/'request.json';save(request,{})
            with patch.object(h,'OUTER',root/'outer'),patch.object(h,'FINAL',root/'final.json'):
                instance=h.Outer(request,h.sha(request),lib);instance.validate=lambda **kw:None
                with patch.object(h,'upstream_ready',side_effect=RuntimeError('upstream failed')):
                    with self.assertRaisesRegex(RuntimeError,'upstream failed'):instance.run()
                self.assertFalse(h.read(root/'final.json')['complete']);self.assertEqual(queries,[])

    def test_success_final_schema_has_three_closed_inner_processes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);lib=self.lib();request=root/'request.json';save(request,{})
            upstream=root/'upstream.json';save(upstream,{})
            with patch.object(h,'OUTER',root/'outer'),patch.object(h,'FINAL',root/'final.json'),patch.object(h,'UPSTREAM',upstream):
                instance=h.Outer(request,h.sha(request),lib);instance.validate=lambda **kw:None
                def stage(name,command):
                    if name=='plan':instance.state['hybrid_plan_sha256']='frozen-plan'
                    else:
                        instance.state['hybrid_receipt_sha256'][name]=name+'-receipt'
                        instance.state['hybrid_controller_processes'][name]={'pid':len(name),'start_ticks':'1'}
                instance.run_stage=stage
                real_sha=h.sha
                with patch.object(h,'upstream_ready',return_value=True),patch.object(h,'sha',side_effect=lambda p:'profile-receipt' if str(p)=='/work/results/prefill-e2e-v1/timed/receipt.json' else real_sha(p)):
                    instance.run()
                result=h.read(root/'final.json')
                self.assertTrue(result['complete']);self.assertTrue(result['cleanup_complete'])
                self.assertEqual(set(result['hybrid_receipt_sha256']),{'proof','timed','quality'})
                self.assertEqual(result['hybrid_plan_sha256'],'frozen-plan')
                self.assertIn('controller_process',result);self.assertIn('finished_unix',result)



    def test_actual_profile4_request_gate_source_key_and_process_binding(self):
        request={'schema':'bounded-fresh-hybrid-outer-request-v1','driver_sha256':'driver',
            'stop_unix':h.STOP_UNIX,'tools':str(h.TOOLS),'output':str(h.OUTPUT),'outer_output':str(h.OUTER),
            'final_receipt':str(h.FINAL),'upstream_profile_receipt':str(h.UPSTREAM),
            'upstream_controller':{'pid':38261,'start_ticks':'6345502'},'poll_seconds':10,
            'child_cleanup_grace_seconds':720,'commands':[[k,v] for k,v in h.stage_commands()],
            'files':{str(h.TOOLS/'hybrid_driver.py'):h.HELPER_SHA,
                     str(h.TOOLS/'helper-freeze-fresh-v1.json'):h.FREEZE_SHA,str(h.LIFECYCLE):h.LIFECYCLE_SHA,
                     str(h.RECOVERY):h.RECOVERY_SHA},'recovery_binding':'binding','pod_uid':h.UID,'prospective_admission':'admission'}
        allocation={**h.recovery.OWNER,'verified_live':True,'gpu_count':8,'gpu_name':'NVIDIA H200'}
        profile={'schema':'headline-prefill-profiles-v4','main_plan_sha256':h.MAIN_SHA,
            'profile_files':{'/work/followup/profiles-v4/profile_prefill.py':h.PROFILE_HELPER_SHA}}
        values={'/work/results/allocation-proof.json':allocation,
            '/work/results/provenance/task-source-deployment.json':{'pod_identity':allocation},
            str(h.UPSTREAM_LAUNCH):{'complete':True,'started':True,'process':{'pid':38261,'pgrp':38261,'start_ticks':'6345502'},
                'pod_uid':h.recovery.OLD['uid'],'plan_sha256':h.PROFILE_SHA},str(h.PROFILE_PLAN):profile}
        def sha(path):
            if str(path)=='request':return 'request-sha'
            if str(path)==str(h.PROFILE_PLAN):return h.PROFILE_SHA
            if str(path)==str(h.MAIN_PLAN):return h.MAIN_SHA
            return request['files'].get(str(path),'driver')
        with patch.object(h,'sha',side_effect=sha),patch.object(h,'read',side_effect=lambda p:values[str(p)]),patch.object(h.recovery,'verify',return_value='binding'),patch.object(h,'load_lifecycle',return_value=self.lib()),patch.object(h.prospective,'verify',return_value='admission'):
            h.request_gate(request,'request-sha',Path('request'))
            profile['profile_files']={'/work/followup/profiles-v2/profile_prefill.py':h.PROFILE_HELPER_SHA}
            with self.assertRaises((KeyError,RuntimeError)):h.request_gate(request,'request-sha',Path('request'))

    def test_previous_failed_profile3_controller_cannot_satisfy_profile4(self):
        with tempfile.TemporaryDirectory() as folder:
            path,_=self.upstream(Path(folder),controller_process={'pid':37050,'pgrp':37050,'start_ticks':'6200412'})
            with patch.object(h,'UPSTREAM',path),self.assertRaisesRegex(RuntimeError,'another process'):
                h.upstream_ready(self.lib())

if __name__=='__main__':unittest.main()
