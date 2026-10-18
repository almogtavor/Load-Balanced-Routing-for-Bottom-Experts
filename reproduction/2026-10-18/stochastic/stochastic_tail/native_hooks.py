"""Observer revision4: hook both native runners and prove the selected instance.

No routing/kernel/RNG math is defined here. Explicit dummy scope outranks the
nested MRV2 execute(dummy_run=True). Unknown routing outside native scopes fails
new training proof rather than acquiring an inferred live/dummy label.
"""
from functools import wraps
import inspect
from pathlib import Path


def guard_route(control,router,logits,record_calls,context,emit,need):
    """Called under the same reentrant lock used by terminal checkpoints."""
    if control['sealed']:
        emit('postseal_violation',nonce=control['seal_nonce'],layer=router._stochastic_layer_index,
             call=router._stochastic_call_count)
        raise RuntimeError('Routing attempted after terminal native checkpoint')
    need(control.get('runner_identity') is not None,'Actual selected native runner not attested')
    if record_calls and context.get() is None:
        emit('orphan_route_call',observer_violation=True,layer=router._stochastic_layer_index,
             call=router._stochastic_call_count,local_T=int(logits.shape[0]),experts=int(logits.shape[1]))
        raise RuntimeError('Routing attempted outside actual native runner context')


def install(config, control, step_context, emit, callable_identity, need):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as V1
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as V2
    from vllm.v1.worker.gpu_worker import Worker
    classes=(V1,V2)
    originals={cls:{name:getattr(cls,name) for name in ('execute_model','_dummy_run','shutdown')} for cls in classes}
    sequence=0
    identity_cache={}

    def native_identity(function):
        if function in identity_cache:return identity_cache[function]
        inner=inspect.unwrap(function)
        value=callable_identity(inner)
        need(value['source_sha256']==config['native_source_sha256'][
            'vllm/v1/worker/gpu/model_runner.py' if inner.__module__=='vllm.v1.worker.gpu.model_runner'
            else 'vllm/v1/worker/gpu_model_runner.py'], 'Native runner source differs')
        identity_cache[function]=value
        return value

    def runner_identity(runner):
        cls=type(runner)
        need(cls in classes,'Unsupported actual native runner class')
        path=Path(inspect.getfile(cls)).resolve()
        use_v2=bool(runner.vllm_config.use_v2_model_runner)
        need(use_v2 is True and cls is V2,'This observer revision requires the unchanged default MRV2 recipe')
        record={'module':cls.__module__,'qualname':cls.__qualname__,'source_path':str(path),
                'source_sha256':config['native_source_sha256']['vllm/v1/worker/gpu/model_runner.py'],
                'use_v2_model_runner':use_v2,
                'native_methods':{name:native_identity(fn) for name,fn in originals[cls].items()},
                'actual_patched_methods':{name:callable_identity(getattr(cls,name)) for name in originals[cls]}}
        need(path==Path(config['native_source_files']['vllm/v1/worker/gpu/model_runner.py']).resolve(),
             'Actual selected runner path differs')
        return record

    def step(kind, call, metadata):
        nonlocal sequence
        sequence+=1
        parent=step_context.get()
        value={'step':sequence,'kind':kind,'calls':[]}
        # Keep the outer explicit dummy as the innermost routing owner even
        # when MRV2 calls its scheduled entrypoint inside _dummy_run.
        keep_parent=parent is not None and parent['kind']=='explicit_dummy'
        token=None if keep_parent else step_context.set(value)
        try:return call()
        finally:
            if token is not None:step_context.reset(token)
            emit('native_step_calls',step=value['step'],native_kind=kind,
                 parent_step=parent['step'] if parent else None,calls=value['calls'],metadata=metadata,
                 routing_owner_step=parent['step'] if keep_parent else value['step'])

    for cls in classes:
        native_execute,native_dummy,native_shutdown=(originals[cls][name] for name in ('execute_model','_dummy_run','shutdown'))
        def make_execute(original,runner_cls):
            signature=inspect.signature(original)
            @wraps(original)
            def executed(self,scheduler_output,*args,**kwargs):
                bound=signature.bind(self,scheduler_output,*args,**kwargs);bound.apply_defaults()
                dummy=bool(bound.arguments.get('dummy_run',False));profile=bool(bound.arguments.get('is_profile',False))
                counts=dict(scheduler_output.num_scheduled_tokens)
                total=int(scheduler_output.total_num_scheduled_tokens)
                kind='explicit_dummy_from_flag' if dummy else 'scheduled_empty' if total==0 else 'scheduled'
                metadata={'scheduled_tokens':counts,'total_scheduled_tokens':total,'dummy_run':dummy,'is_profile':profile,
                          'runner_module':runner_cls.__module__,'runner_qualname':runner_cls.__qualname__,
                          'scheduler_output_type':type(scheduler_output).__module__+'.'+type(scheduler_output).__qualname__,
                          'native_execute':native_identity(original)}
                return step(kind,lambda:original(self,scheduler_output,*args,**kwargs),metadata)
            return executed
        def make_dummy(original,runner_cls):
            @wraps(original)
            def dummied(self,num_tokens,*args,**kwargs):
                return step('explicit_dummy',lambda:original(self,num_tokens,*args,**kwargs),
                    {'requested_dummy_tokens':int(num_tokens),'is_profile':bool(kwargs.get('is_profile',False)),
                     'runner_module':runner_cls.__module__,'runner_qualname':runner_cls.__qualname__,
                     'native_dummy':native_identity(original)})
            return dummied
        def make_shutdown(original,runner_cls):
            @wraps(original)
            def shutdown(self,*args,**kwargs):
                # Supplemental evidence precedes native teardown, which can hang.
                control['snapshot']('native_model_runner_shutdown_enter',actual_runner=runner_identity(self))
                try:return original(self,*args,**kwargs)
                finally:control['snapshot']('native_model_runner_shutdown_exit',actual_runner=control.get('runner_identity'))
            return shutdown
        if config['record_calls']:
            cls.execute_model=make_execute(native_execute,cls)
            cls._dummy_run=make_dummy(native_dummy,cls)
        cls.shutdown=make_shutdown(native_shutdown,cls)
    original_load=Worker.load_model
    @wraps(original_load)
    def loaded(self,*args,**kwargs):
        result=original_load(self,*args,**kwargs)
        identity=runner_identity(self.model_runner)
        need(control.get('runner') in (None,self.model_runner),'Native worker runner replaced')
        control['runner']=self.model_runner;control['runner_identity']=identity
        emit('native_runner_selected',complete=True,actual_runner=identity,
             actual_worker_module=type(self).__module__,actual_worker_qualname=type(self).__qualname__,
             actual_worker_load=callable_identity(Worker.load_model),
             native_worker_load=callable_identity(inspect.unwrap(original_load)))
        return result
    Worker.load_model=loaded
    control['runner_identity_function']=runner_identity
