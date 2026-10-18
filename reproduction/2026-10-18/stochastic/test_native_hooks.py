#!/usr/bin/env python3
"""Functional MRV1/MRV2 context/selection and atomic checkpoint guard fixtures."""
from contextvars import ContextVar
import hashlib
import json
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch

from stochastic_tail import native_hooks,plugin
from stochastic_tail.worker_extension import CheckpointExtension


class NativeHookTests(unittest.TestCase):
    def fixture(self):
        events=[];context=ContextVar('test_native_context',default=None)
        cfg=types.SimpleNamespace(use_v2_model_runner=True)
        def callback():
            active=context.get();self.assertIsNotNone(active)
            active['calls'].append({'layer':0,'call':0,'local_T':17,'experts':128})
        modules={}
        for name in ['vllm','vllm.v1','vllm.v1.worker','vllm.v1.worker.gpu','vllm.v1.worker.gpu_model_runner',
                     'vllm.v1.worker.gpu.model_runner','vllm.v1.worker.gpu_worker']:
            mod=types.ModuleType(name);mod.__path__=[];mod.__file__=__file__;modules[name]=mod
        classes=[]
        for name in ['vllm.v1.worker.gpu_model_runner','vllm.v1.worker.gpu.model_runner']:
            def init(self,config):self.vllm_config=config;self.was_shutdown=False
            def executed(self,scheduler_output,intermediate_tensors=None,dummy_run=False,is_profile=False):
                if dummy_run or scheduler_output.total_num_scheduled_tokens:callback()
                return 'executed'
            def dummied(self,num_tokens,*args,**kwargs):
                scheduler=types.SimpleNamespace(num_scheduled_tokens={'_dummy_req_0':num_tokens},total_num_scheduled_tokens=num_tokens)
                return self.execute_model(scheduler,dummy_run=True,is_profile=kwargs.get('is_profile',False))
            def shutdown(self):self.was_shutdown=True
            for function in (init,executed,dummied,shutdown):function.__module__=name
            cls=type('GPUModelRunner',(),{'__module__':name,'__init__':init,'execute_model':executed,'_dummy_run':dummied,'shutdown':shutdown})
            modules[name].GPUModelRunner=cls;classes.append(cls)
        def load_model(self):self.model_runner=classes[1](cfg);return 'loaded'
        load_model.__module__='vllm.v1.worker.gpu_worker'
        worker=type('Worker',(),{'__module__':'vllm.v1.worker.gpu_worker','load_model':load_model})
        modules['vllm.v1.worker.gpu_worker'].Worker=worker
        native_names=['vllm/v1/worker/gpu_model_runner.py','vllm/v1/worker/gpu/model_runner.py','vllm/v1/worker/gpu_worker.py']
        config={'record_calls':True,'native_source_files':{name:__file__ for name in native_names},
                'native_source_sha256':{name:plugin.sha(__file__) for name in native_names}}
        def emit(kind,**fields):events.append({'kind':kind,**fields})
        control={'snapshot':lambda reason,**fields:emit('counter_snapshot',reason=reason,**fields)}
        with patch.dict(sys.modules,modules):
            native_hooks.install(config,control,context,emit,plugin.callable_identity,plugin.need)
            instance=worker();instance.load_model()
            yield classes,instance,context,events,control

    def test_selected_actual_v2_callback_and_dummy_outer_context_wins(self):
        for classes,worker,context,events,control in self.fixture():
            self.assertEqual(events[0]['kind'],'native_runner_selected')
            self.assertEqual(events[0]['actual_runner']['module'],'vllm.v1.worker.gpu.model_runner')
            worker.model_runner._dummy_run(17,is_profile=True)
            steps=[e for e in events if e['kind']=='native_step_calls']
            self.assertEqual(len(steps),2)
            nested,outer=steps
            self.assertEqual(nested['native_kind'],'explicit_dummy_from_flag')
            self.assertEqual(nested['calls'],[]);self.assertEqual(nested['routing_owner_step'],outer['step'])
            self.assertEqual(outer['native_kind'],'explicit_dummy');self.assertEqual(len(outer['calls']),1)
            self.assertIsNone(context.get())

    def test_real_and_empty_scheduled_classification_and_direct_dummy_flag(self):
        for classes,worker,context,events,control in self.fixture():
            for total in (0,17):
                scheduler=types.SimpleNamespace(num_scheduled_tokens={} if not total else {'actual':total},total_num_scheduled_tokens=total)
                worker.model_runner.execute_model(scheduler)
            scheduler=types.SimpleNamespace(num_scheduled_tokens={'dummy':1},total_num_scheduled_tokens=1)
            worker.model_runner.execute_model(scheduler,dummy_run=True)
            steps=[e for e in events if e['kind']=='native_step_calls']
            self.assertEqual([e['native_kind'] for e in steps],['scheduled_empty','scheduled','explicit_dummy_from_flag'])
            self.assertEqual([len(e['calls']) for e in steps],[0,1,1])

    def test_v1_is_hooked_but_cannot_be_falsely_attested_as_v2(self):
        for classes,worker,context,events,control in self.fixture():
            legacy=classes[0](types.SimpleNamespace(use_v2_model_runner=False));legacy._dummy_run(17)
            self.assertTrue(any(e.get('metadata',{}).get('runner_module')=='vllm.v1.worker.gpu_model_runner' for e in events))
            with self.assertRaises(RuntimeError):control['runner_identity_function'](legacy)

    def test_shutdown_snapshot_precedes_native_teardown(self):
        for classes,worker,context,events,control in self.fixture():
            worker.model_runner.shutdown();self.assertTrue(worker.model_runner.was_shutdown)
            snapshots=[e for e in events if e['kind']=='counter_snapshot']
            self.assertEqual([e['reason'] for e in snapshots],['native_model_runner_shutdown_enter','native_model_runner_shutdown_exit'])


class CheckpointTests(unittest.TestCase):
    def fixture(self,purpose='proof'):
        events=[];runner=object();model=types.SimpleNamespace(actual='runner')
        counters={i:types.SimpleNamespace(_stochastic_call_count=3) for i in range(48)}
        control={'plan_sha256':'plan','route_lock':threading.RLock(),'sealed':False,'seal_nonce':None,
                 'checkpoint_nonces':set(),'runner':runner,'runner_identity':model,
                 'runner_identity_function':lambda r:model,'counters':counters,'routes':{i:{} for i in range(48)}}
        def emit(kind,**fields):
            row={'kind':kind,**fields};events.append(row);return {'row':row,'record_sha256':'record','ledger_path':'ledger'}
        control['emit']=emit
        worker=type('Worker',(CheckpointExtension,),{})();worker.model_runner=runner
        torch=types.ModuleType('torch');torch.cuda=types.SimpleNamespace(synchronize=lambda:events.append({'kind':'synchronize'}))
        return worker,control,events,torch

    def test_terminal_checkpoint_synchronizes_and_atomically_seals_before_snapshot(self):
        worker,control,events,torch=self.fixture()
        payload=json.dumps({'plan_sha256':'plan','phase':'proof','nonce':'a'*32,'sealed':True})
        with patch.dict(sys.modules,{'torch':torch}),patch.object(plugin,'_CONFIG',{'purpose':'proof'}),patch.object(plugin,'_CONTROL',control):
            result=worker.stochastic_tail_checkpoint(payload)
            self.assertTrue(control['sealed']);self.assertEqual(events[0]['kind'],'synchronize')
            self.assertTrue(result['row']['sealed']);self.assertEqual(set(result['row']['counters']),{str(i) for i in range(48)})
            with self.assertRaises(RuntimeError):worker.stochastic_tail_checkpoint(payload)

    def test_unsealed_training_checkpoint_preserves_future_counter_namespace(self):
        worker,control,events,torch=self.fixture('quality')
        with patch.dict(sys.modules,{'torch':torch}),patch.object(plugin,'_CONFIG',{'purpose':'quality'}),patch.object(plugin,'_CONTROL',control):
            worker.stochastic_tail_checkpoint(json.dumps({'plan_sha256':'plan','phase':'preflight','nonce':'b'*32,'sealed':False}))
            self.assertFalse(control['sealed']);self.assertTrue(all(r._stochastic_call_count==3 for r in control['counters'].values()))
            worker.stochastic_tail_checkpoint(json.dumps({'plan_sha256':'plan','phase':'score','nonce':'c'*32,'sealed':True}))
            self.assertTrue(control['sealed'])

    def test_route_guard_rejects_after_seal_before_any_counter_advance(self):
        router=types.SimpleNamespace(_stochastic_layer_index=7,_stochastic_call_count=9)
        events=[];control={'sealed':True,'seal_nonce':'d'*32,'runner_identity':{}}
        emit=lambda kind,**fields:events.append({'kind':kind,**fields})
        with self.assertRaises(RuntimeError):native_hooks.guard_route(control,router,None,True,ContextVar('sealed_ctx',default=None),emit,plugin.need)
        self.assertEqual(router._stochastic_call_count,9);self.assertEqual(events[0]['kind'],'postseal_violation')

    def test_route_without_actual_native_context_cannot_be_published_as_live(self):
        router=types.SimpleNamespace(_stochastic_layer_index=1,_stochastic_call_count=0)
        events=[];control={'sealed':False,'runner_identity':{'actual':True}}
        with self.assertRaises(RuntimeError):native_hooks.guard_route(control,router,types.SimpleNamespace(shape=(17,128)),True,
            ContextVar('missing_ctx',default=None),lambda kind,**fields:events.append({'kind':kind,**fields}),plugin.need)
        self.assertEqual(events[0]['kind'],'orphan_route_call');self.assertEqual(router._stochastic_call_count,0)


if __name__=='__main__':unittest.main(verbosity=2)
