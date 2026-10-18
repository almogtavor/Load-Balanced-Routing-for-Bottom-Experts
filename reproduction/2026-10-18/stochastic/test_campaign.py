#!/usr/bin/env python3
"""Adversarial closed-proof and remote-exit barriers. No model imports."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import campaign as c

class ProofBarrierTests(unittest.TestCase):
 def fixture(self,root):
  files={};private=root/'site';assignment=root/'assignment.json'
  assignment.write_text(json.dumps({'assignments':[{'seed':s,'pod_uid':f'uid{s}'} for s in range(3)]}))
  q={'campaign_kind':'quality','private_site':str(private),'global_assignment':str(assignment),'ledger_helper':'/frozen/audit_stochastic_ledgers_v2.py','training_proofs':[]}
  files[q['ledger_helper']]='helper'
  for n in c.PRIVATE_FILES:files[str(private/'stochastic_tail'/n)]='private-'+n
  for seed in range(3):
   folder=root/f'proof{seed}';folder.mkdir();ref={}
   def add(key,value):
    path=folder/(key+'.json');path.write_text(json.dumps(value,sort_keys=True));digest=c.sha(path)
    files[str(path)]=digest;ref[key]={'path':str(path),'sha256':digest};return digest
   proof={'schema':c.SCHEMA,'campaign_kind':'proof','observer_revision':4,'seeds':[seed],'allocation':{'uid':f'uid{seed}'},'private_freeze':'/freeze','task_adapter':'/adapter','private_site':str(private),'files':{'/freeze':c.SELECTION_FREEZE_SHA,'/adapter':c.ADAPTER_SHA,**{str(private/'stochastic_tail'/n):'private-'+n for n in c.PRIVATE_FILES}}}
   ps=add('plan',proof)
   ls=add('launch',{'complete':True,'workload_complete':True,'campaign_plan_sha256':ps,'finalization_errors':[],'server_cleanup':{'remaining_members':[]}})
   aus=add('audit',{'complete':True,'schema':'closed-stochastic-native-seal-audit-v2','purpose':'proof','seed':seed,'arm':'weighted6','campaign_sha256':ps,'launch_sha256':ls,'adapter_sha256':c.ADAPTER_SHA,'worker_pids':[100,101,102,103],'helper_sha256':'helper'})
   proc={'pid':1000+seed,'pgrp':1000+seed,'start_ticks':str(1234+seed)}
   add('receipt',{'complete':True,'cleanup_complete':True,'closed_ledgers_audited':True,'campaign_kind':'proof','plan_sha256':ps,'finished_unix':10.,'controller_process':proc,'launches':[{'complete':True,'receipt_sha256':ls,'raw_ledger_audit_sha256':aus,'arm':'weighted6','seed':seed,'position':0}]})
   add('exit',{'complete':True,'exited':True,'pod_uid':f'uid{seed}','controller_process':proc,'observed_process':None,'verified_unix':11.})
   q['training_proofs'].append({'seed':seed,**ref})
  return q,files
 def change(self,q,files,seed,key,mutate):
  ref=q['training_proofs'][seed][key];value=c.read(ref['path']);mutate(value);Path(ref['path']).write_text(json.dumps(value,sort_keys=True));ref['sha256']=files[ref['path']]=c.sha(ref['path'])
 def test_three_exact_closed_remote_proofs_pass(self):
  with tempfile.TemporaryDirectory() as t:
   q,files=self.fixture(Path(t));c.training_proof_gate(q,files)
 def test_missing_duplicate_seed_or_unpinned_evidence_rejected(self):
  for kind in ('missing','duplicate','unpinned','mutated'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory() as t:
    q,files=self.fixture(Path(t))
    if kind=='missing':q['training_proofs'].pop()
    elif kind=='duplicate':q['training_proofs'][2]['seed']=1
    elif kind=='unpinned':files.pop(q['training_proofs'][1]['exit']['path'])
    else:Path(q['training_proofs'][2]['audit']['path']).write_text('{}')
    with self.assertRaises(RuntimeError):c.training_proof_gate(q,files)
 def test_forged_observer_source_host_or_seed_rejected(self):
  edits=[lambda p:p.update(observer_revision=3),lambda p:p.update(seeds=[2]),lambda p:p['allocation'].update(uid='oldpod'),lambda p:p['files'].update({'/freeze':'old'}),lambda p:p['files'].update({str(Path(p['private_site'])/'stochastic_tail/plugin.py'):'other'})]
  for edit in edits:
   with self.subTest(edit=edit),tempfile.TemporaryDirectory() as t:
    q,files=self.fixture(Path(t));self.change(q,files,0,'plan',edit)
    with self.assertRaises(RuntimeError):c.training_proof_gate(q,files)
 def test_activation_without_closed_raw_audit_is_not_a_barrier(self):
  edits=[('receipt',lambda r:r.update(closed_ledgers_audited=False)),('receipt',lambda r:r.update(cleanup_complete=False)),('launch',lambda r:r.update(workload_complete=False)),('audit',lambda r:r.update(helper_sha256='another')),('audit',lambda r:r.update(purpose='quality'))]
  for key,edit in edits:
   with self.subTest(key=key),tempfile.TemporaryDirectory() as t:
    q,files=self.fixture(Path(t));self.change(q,files,1,key,edit)
    with self.assertRaises(RuntimeError):c.training_proof_gate(q,files)
 def test_remote_original_pid_exit_not_inferred(self):
  edits=[lambda e:e.update(observed_process={'pid':1000,'start_ticks':'1234'}),lambda e:e['controller_process'].update(start_ticks='reused'),lambda e:e.update(verified_unix=9.),lambda e:e.update(exited=False),lambda e:e.update(pod_uid='different')]
  for edit in edits:
   with self.subTest(edit=edit),tempfile.TemporaryDirectory() as t:
    q,files=self.fixture(Path(t));self.change(q,files,0,'exit',edit)
    with self.assertRaises(RuntimeError):c.training_proof_gate(q,files)
 def test_proof_itself_consumes_no_barrier(self):
  c.training_proof_gate({'campaign_kind':'proof'}, {})
  with self.assertRaises(RuntimeError):c.training_proof_gate({'campaign_kind':'proof','training_proofs':[{}]}, {})

class DeadlineTests(unittest.TestCase):
 def test_stage_stops_before_cleanup_reserve(self):
  class Base:
   def command(self,command,log,env,timeout,server=None):return timeout
  cls=c.make_campaign_class(SimpleNamespace(Campaign=Base),None);obj=object.__new__(cls);obj.plan={'stop_unix':1000+c.CLEANUP_MARGIN+11}
  with patch.object(c.time,'time',return_value=1000):
   self.assertEqual(obj.command([],None,{},600),11);self.assertEqual(obj.command([],None,{},4),4)
  with patch.object(c.time,'time',return_value=1012):
   with self.assertRaises(RuntimeError):obj.command([],None,{},600)
 def test_loopback_management_and_no_forced_runner(self):
  plan={'vllm':'/vllm','model':'/model','private_site':'/private','public_site':'/public'};cmd=c.server_command(plan)
  self.assertEqual(cmd[cmd.index('--host')+1],'127.0.0.1');self.assertEqual(cmd[cmd.index('--worker-extension-cls')+1],'stochastic_tail.worker_extension.CheckpointExtension');self.assertEqual(cmd[cmd.index('--max-model-len')+1],'8192')
  with patch.dict(c.os.environ,{'VLLM_USE_V2_MODEL_RUNNER':'0','HF_TOKEN':'secret'},clear=True):env=c.environment(plan,{'arm':'weighted6','plan':'/plan','plan_sha256':'sha'})
  self.assertEqual(env['VLLM_SERVER_DEV_MODE'],'1');self.assertNotIn('VLLM_USE_V2_MODEL_RUNNER',env);self.assertNotIn('HF_TOKEN',env)

class ClosedAcceptanceTests(unittest.TestCase):
 def setup_launch(self,root,cleanup_fails=False,audit_fails=False):
  from contextlib import nullcontext
  identities=[{'pid':p,'pgrp':200,'start_ticks':'1','state':'S'} for p in [200,101,102,103,104]]
  def stop(child,identity):
   if cleanup_fails:raise RuntimeError('injected owned cleanup failure')
   return {'identity':identity,'remaining_members':[]}
  base=SimpleNamespace(Campaign=object,gpu_idle=lambda **kw:None,process_record=lambda pid:identities[0],
       group_members=lambda pid:identities,stop_owned=stop)
  cls=c.make_campaign_class(base,None);obj=object.__new__(cls)
  obj.root=root;obj.plan_path=root/'plan.json';obj.plan_path.write_text('{}')
  obj.plan={'campaign_kind':'proof','private_site':'/private','public_site':'/public','url':'http://127.0.0.1:8077',
      'server_command':['model'],'boot_timeout':30,'proof_timeout':30,'request_timeout':30,'audit_timeout':30,
      'stop_unix':c.time.time()+1000,'task_adapter':'/adapter','task_harness':'/harness','ledger_helper':'/audit','prepared':'/prepared',
      'files':{'/audit':'helper'}}
  obj.state={'launches':[]};obj.child=obj.child_id=obj.server=obj.server_id=None
  obj.save=lambda:None;obj.validate=lambda **kw:None
  directory=root/'seed0/position0-weighted6'
  cell={'directory':str(directory),'seed':0,'position':0,'arm':'weighted6','plan':'/cellplan','plan_sha256':'cellsha'}
  def command(cmd,log,env,timeout,server=None):
   if '--launch' in cmd:
    target=Path(cmd[cmd.index('--output')+1]);target.write_text(json.dumps({'complete':not audit_fails,'launch_sha256':c.sha(directory/'launch.json'),'campaign_sha256':c.sha(obj.plan_path),'helper_sha256':'helper','purpose':'proof'}))
    if audit_fails:raise RuntimeError('injected strict ledger rejection')
   else:
    (directory/'proof').mkdir();(directory/'proof/result.json').write_text(json.dumps({'complete':True,'experiment_plan_sha256':'cellsha','arm':'weighted6','router_seed':0,'worker_pids':[101,102,103,104],'worker_identity_records':[{}, {}, {}, {}]}))
    (directory/'ledger').mkdir();(directory/'ledger/pid101.jsonl').write_text('{}\n')
   return {'complete':True}
  obj.command=command
  sock=SimpleNamespace(connect_ex=lambda address:1)
  child=SimpleNamespace(pid=200,poll=lambda:None)
  contexts=[patch.object(c.subprocess,'Popen',return_value=child),patch.object(c.socket,'socket',return_value=nullcontext(sock)),
      patch.object(c.urllib.request,'urlopen',return_value=nullcontext(SimpleNamespace(status=200))),patch.object(c,'environment',lambda *a:{})]
  return obj,cell,directory,contexts
 def test_strict_audit_failure_preserves_closed_execution_without_acceptance(self):
  from contextlib import ExitStack
  with tempfile.TemporaryDirectory() as t:
   obj,cell,directory,contexts=self.setup_launch(Path(t),audit_fails=True)
   with ExitStack() as stack:
    for context in contexts:stack.enter_context(context)
    with self.assertRaisesRegex(RuntimeError,'strict ledger rejection'):obj.launch_cell(cell)
   launch=c.read(directory/'launch.json');state=obj.state['launches'][0]
   self.assertTrue(launch['complete']);self.assertEqual(launch['server_cleanup']['remaining_members'],[])
   self.assertFalse(state['complete']);self.assertIn('raw_ledger_audit_error',state)
   self.assertEqual(state['receipt_sha256'],c.sha(directory/'launch.json'))
   self.assertEqual(state['raw_ledger_audit_sha256'],c.sha(directory/'raw-ledger-audit.json'))
 def test_failed_owned_cleanup_keeps_immutable_failure_receipt(self):
  from contextlib import ExitStack
  with tempfile.TemporaryDirectory() as t:
   obj,cell,directory,contexts=self.setup_launch(Path(t),cleanup_fails=True)
   with ExitStack() as stack:
    for context in contexts:stack.enter_context(context)
    with self.assertRaisesRegex(RuntimeError,'Cell failed'):obj.launch_cell(cell)
   launch=c.read(directory/'launch.json')
   self.assertFalse(launch['complete']);self.assertTrue(launch['workload_complete'])
   self.assertEqual(launch['finalization_errors'][0]['step'],'server_cleanup')
   self.assertFalse(obj.state['launches'][0]['complete']);self.assertFalse((directory/'raw-ledger-audit.json').exists())

if __name__=='__main__':unittest.main()
