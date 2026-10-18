"""Native collective-RPC checkpoints after an API-owned all-DP pause barrier."""
import json
import re


class CheckpointExtension:
    def stochastic_tail_checkpoint(self,payload):
        import torch
        from . import plugin
        config=plugin._CONFIG;control=plugin._CONTROL;need=plugin.need
        value=json.loads(payload)
        need(value['plan_sha256']==control['plan_sha256'] and value['phase'] in ('proof','preflight','score')
             and re.fullmatch('[0-9a-f]{32}',value['nonce']) is not None,'Checkpoint plan/phase/nonce differs')
        need(value['sealed'] is (value['phase'] in ('proof','score')),'Terminal checkpoint must seal; training preflight resumes')
        need((value['phase']=='proof' and config['purpose']=='proof')
             or (value['phase'] in ('preflight','score') and config['purpose']=='quality'), 'Checkpoint purpose differs')
        with control['route_lock']:
            need(not control['sealed'] and value['nonce'] not in control['checkpoint_nonces'],'Duplicate/sealed checkpoint')
            need(plugin._STEP.get() is None and control.get('runner') is self.model_runner,'Checkpoint insideforward or wrong actual runner')
            need(set(control['counters'])==set(range(48)) and set(control['routes'])==set(range(48)), 'Checkpoint lacks48actualrouters')
            actual=control['runner_identity_function'](self.model_runner)
            need(actual==control['runner_identity'],'Actual runner identity changed')
            # Native pause(mode=wait,clear_cache=false) is attested in the client
            # raw transcript; synchronizing completes work before final counters.
            torch.cuda.synchronize()
            control['sealed']=value['sealed'];control['seal_nonce']=value['nonce'] if value['sealed'] else None
            control['checkpoint_nonces'].add(value['nonce'])
            return control['emit']('native_checkpoint',complete=True,reason='collective_rpc_after_native_pause',
                phase=value['phase'],nonce=value['nonce'],sealed=value['sealed'],active_step=None,
                counters={str(i):r._stochastic_call_count for i,r in sorted(control['counters'].items())},
                actual_runner=actual,cuda_synchronized=True,
                checkpoint_callable=plugin.callable_identity(type(self).stochastic_tail_checkpoint))
