"""Source-bound native external/internal request identity; never change IDs."""
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
import re
import time

SOURCES = {
    'envs.py': '1d7101800f67b803a700fbf7489292cf1f0022b570085166ad407e6b1513dc15',
    'utils/__init__.py': 'c58b3f45deeb98bccca22c945526fd15d483a2f5eb8a3c62cf7df218b081885f',
    'v1/engine/input_processor.py': 'd79601e4a053a7bacb0d6a23e273348435e4ab70c6a315dcccb2d428ed57d96b',
    'entrypoints/openai/completion/serving.py': 'aac889978efb034b95ab81ec59695d83ba7c879ab48cd5be2448c980f109a3bd',
    'v1/engine/output_processor.py': 'c72d0149e18d6da71d53e23b6550d09310af6917523fef6e282b0907d36b8d54',
    'v1/engine/async_llm.py': 'ff933ed78aa3195782cf1c6d18c53897adfa5bf920421bd2bcc330f2adc84d1e',
}

def need(v,m):
    if not v: raise RuntimeError(m)

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def installed_sources(root):
    files={name:str((Path(root)/name).resolve()) for name in SOURCES}
    need(all(digest(path)==SOURCES[name] for name,path in files.items()),'Native API request-ID source changed')
    return {'files':files,'sha256':dict(SOURCES)}

def external_item_id(response_id):
    need(isinstance(response_id,str) and response_id.startswith('cmpl-') and len(response_id)>5,'Invalid completion response ID')
    return response_id+'-0'

def require_internal_id(external,internal):
    need(isinstance(external,str) and external and isinstance(internal,str)
         and re.fullmatch(re.escape(external)+'-[0-9a-f]{8}',internal) is not None,
         'Native internal request ID does not preserve exact external ID and randomized suffix')
    return internal

def producer_internal_id(response):
    params=response.get('kv_transfer_params')
    need(isinstance(params,dict),'Missing native producer transfer parameters')
    return require_internal_id(external_item_id(response.get('id')),params.get('remote_request_id'))

def prompt_identity(prompt):
    need(isinstance(prompt,list) and prompt and all(type(t) is int and t>=0 for t in prompt),'Native assignment requires integer prompt IDs')
    return {'prompt_len':len(prompt),'prompt_sha256':hashlib.sha256(json.dumps(prompt,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),
            'last_prompt_token_id':prompt[-1]}

def install_assignment_hook(processor,envs,sources,emit):
    """Wrap the actual staticmethod without altering its inputs, output or IDs."""
    native=processor.assign_request_id
    filename=str(Path(native.__code__.co_filename).resolve())
    proof={'module':native.__module__,'qualname':native.__qualname__,
           'parameter_names':list(inspect.signature(native).parameters),'code_filename':filename,'source_sha256':digest(filename)}
    need(proof['module']=='vllm.v1.engine.input_processor' and proof['qualname']=='InputProcessor.assign_request_id'
         and proof['parameter_names']==['request'] and filename==sources['files']['v1/engine/input_processor.py']
         and proof['source_sha256']==SOURCES['v1/engine/input_processor.py'],'Actual native ID assignment callable differs')
    uuid=native.__globals__['random_uuid']
    uuid_file=str(Path(uuid.__code__.co_filename).resolve())
    uuid_proof={'module':uuid.__module__,'qualname':uuid.__qualname__,'code_filename':uuid_file,'source_sha256':digest(uuid_file)}
    need(uuid_proof=={'module':'vllm.utils','qualname':'random_uuid','code_filename':sources['files']['utils/__init__.py'],
                     'source_sha256':SOURCES['utils/__init__.py']},'Actual native randomized-ID factory differs')
    proof['random_uuid_callable']=uuid_proof
    need(envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION is False,'Native request-ID randomization must remain enabled')
    @wraps(native)
    def assigned(request):
        started=time.time_ns();external=request.request_id
        need(request.external_req_id is None,'Assignment received already assigned native request')
        before=prompt_identity(request.prompt_token_ids)
        result=native(request)
        need(request.external_req_id==external,'Native external request identity changed unexpectedly')
        internal=require_internal_id(external,request.request_id)
        need(prompt_identity(request.prompt_token_ids)==before,'Native assignment changed prompt IDs')
        emit('request_id_assigned',external_request_id=external,internal_request_id=internal,
             assignment_started_unix_ns=started,assignment_finished_unix_ns=time.time_ns(),
             native_assign_callable=proof,request_id_sources=sources,**before)
        return result
    processor.assign_request_id=staticmethod(assigned)
    return proof

def assignment_index(events,sources):
    need(sources['sha256']==SOURCES and set(sources['files'])==set(SOURCES),'Native request-ID source contract differs')
    index={};internal=set()
    expected={'module':'vllm.v1.engine.input_processor','qualname':'InputProcessor.assign_request_id',
              'parameter_names':['request'],'code_filename':sources['files']['v1/engine/input_processor.py'],
              'source_sha256':SOURCES['v1/engine/input_processor.py'],
              'random_uuid_callable':{'module':'vllm.utils','qualname':'random_uuid','code_filename':sources['files']['utils/__init__.py'],'source_sha256':SOURCES['utils/__init__.py']}}
    for row in events:
        if row['kind']!='request_id_assigned':continue
        external=row['external_request_id'];full=require_internal_id(external,row['internal_request_id'])
        need(external not in index and full not in internal,'Duplicate external/internal native assignment')
        need(row['native_assign_callable']==expected and row['request_id_sources']==sources and type(row['pid']) is int and row['pid']>0,
             'Native request-ID assignment source/PID differs')
        need(all(type(row[k]) is int and row[k]>0 for k in ('assignment_started_unix_ns','assignment_finished_unix_ns','unix_ns'))
             and row['assignment_started_unix_ns']<=row['assignment_finished_unix_ns']<=row['unix_ns'],
             'Native assignment timestamps differ')
        index[external]=row;internal.add(full)
    return index

def resolve_assignment(index,response_id,prompt,transport):
    external=external_item_id(response_id)
    need(external in index,'Missing exact native external/internal assignment')
    row=index[external]
    need(all(row[k]==v for k,v in prompt_identity(prompt).items()),'Native assignment prompt hash/length/last token differs')
    need(transport['submitted_unix_ns']<=row['assignment_started_unix_ns']<=row['assignment_finished_unix_ns']
         <=row['unix_ns']<=transport['returned_unix_ns'],'Native assignment outside corresponding raw HTTP interval')
    return row['internal_request_id']
