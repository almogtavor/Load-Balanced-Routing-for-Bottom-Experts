"""Source-bound native NIXL accounting after actual pilot metric-schema replay.

Prefix-cache hit accounting is independent of native KV transfer byte/count
coverage. Never insert or rewrite a cached-token value in producer/consumer IO.
"""
import hashlib,json,math,re
from pathlib import Path

SOURCES={'v1/core/sched/scheduler.py':'f820bee4f42b878b262a30f3728a97d7413477b2a277748e20f2528ac2ec2f2a',
 'v1/engine/output_processor.py':'c72d0149e18d6da71d53e23b6550d09310af6917523fef6e282b0907d36b8d54',
 'v1/engine/core.py':'f116962c35d9d312b20b0b76c34a8cdaa6d3f1d31f2036db4e123957d2d1555b',
 'distributed/kv_transfer/kv_connector/v1/nixl/stats.py':'5068697da0250b1e10540f037e8648008d656b2173e48fd2da7694b8f87c88ef'}
BYTES='vllm:nixl_bytes_transferred'
ERRORS=['vllm:nixl_num_failed_transfers_total','vllm:nixl_num_failed_notifications_total','vllm:nixl_num_kv_expired_reqs_total']
SELECTED=[BYTES+'_sum',BYTES+'_count',*ERRORS]
LABEL=re.compile(r'([a-zA-Z_][a-zA-Z_0-9]*)=("(?:[^"\\]|\\.)*")')
SAMPLE=re.compile(r'([a-zA-Z_:][a-zA-Z_0-9:]*)(?:\{(.*)\})?\s+([^\s]+)')

def need(v,m):
    if not v:raise ValueError(m)

def zero_cached_transfer(producer_params,producer_request_params,consumer_request_params):
    """Validate the exact native zero without manufacturing transfer coverage."""
    need(isinstance(producer_params,dict) and producer_params.get('do_remote_prefill') is True
         and producer_params.get('do_remote_decode') is False,'Not native pull handoff parameters')
    need(type(producer_params.get('remote_prefill_cached_tokens')) is int
         and producer_params['remote_prefill_cached_tokens']==0,'Disabled producer prefix cache must report native zero')
    need(producer_request_params=={'do_remote_decode':True,'do_remote_prefill':False,'remote_engine_id':None,
         'remote_block_ids':None,'remote_host':None,'remote_port':None},'Producer request injected/changed cached or native handoff fields')
    need(consumer_request_params==producer_params,'Consumer did not preserve entire native producer parameter object')
    return {'native_remote_prefill_cached_tokens':0,'meaning':'Prefix cache hits on remote producer; not transferred KV tokens'}

def zero_cached_usage(usage,prompt_tokens,output_tokens):
    expected={'prompt_tokens':prompt_tokens,'completion_tokens':output_tokens,'total_tokens':prompt_tokens+output_tokens}
    need(all(type(usage.get(k)) is int and usage[k]==v for k,v in expected.items()),'Logical API token accounting differs')
    value=(usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    need(type(value) is int and value==0,'Expected source-correct remote-producer prefix hit count zero')
    return 0

def labels(text):
    result={};position=0
    while position<len(text):
        match=LABEL.match(text,position);need(match is not None,'Invalid Prometheus label encoding')
        name,value=match.groups();need(name not in result,'Duplicate Prometheus label')
        result[name]=json.loads(value);position=match.end()
        if position<len(text):need(text[position]==',','Invalid Prometheus label separator');position+=1
    return tuple(sorted(result.items()))

def metrics(text):
    """Only selected native metrics affect transfer-coverage accounting."""
    need(isinstance(text,str) and text.endswith('\n'),'Partial native metrics exposition')
    result={name:{} for name in SELECTED};types={}
    for line in text.splitlines():
        if line.startswith('# TYPE '):
            parts=line.split();need(len(parts)==4,'Malformed TYPE line')
            if parts[2] in [BYTES,*ERRORS]:
                need(parts[2] not in types,'Duplicate native TYPE');types[parts[2]]=parts[3]
            continue
        if not line or line.startswith('#'):continue
        match=SAMPLE.fullmatch(line)
        need(match is not None,'Malformed metric sample')
        name,labeltext,value=match.groups()
        if name not in result:continue
        key=labels(labeltext or '')
        need(key not in result[name],'Duplicate selected native sample')
        number=float(value);need(math.isfinite(number) and number>=0,'Invalid native cumulative metric')
        need(number.is_integer(),'Native byte sum/count/failure count must be integral')
        result[name][key]=number
    need(types.get(BYTES)=='histogram' and all(types.get(n)=='counter' for n in ERRORS),'Native exporter metric TYPE differs')
    need(all(result[n] for n in SELECTED),'Missing native metric count/sum/failure coverage')
    keys=set(result[BYTES+'_count'])
    need(all(set(result[n])==keys for n in SELECTED),'Native metrics lack consistent per-engine label coverage')
    return result

def delta(before,after,*,expected_labels,expected_requests,expected_bytes,expected_by_engine):
    """Off-clock cumulative native coverage; exact labels supplied after pilot freeze."""
    keys=set(tuple(sorted(x.items())) for x in expected_labels)
    need(len(keys)==len(expected_labels)==4,'Exactly four frozen native engine label sets required')
    need(type(expected_requests) is int and expected_requests>=0 and type(expected_bytes) is int and expected_bytes>=0 and bool(expected_requests)==bool(expected_bytes),'Finite nonnegative expected native transfer budget')
    for snapshot in [before,after]:
        need(all(set(snapshot[name])==keys for name in SELECTED),'Actual engine labels differ from frozen source/owned deployment')
    changes={name:{key:after[name][key]-before[name][key] for key in keys} for name in SELECTED}
    need(all(v>=0 for rows in changes.values() for v in rows.values()),'Native counter reset/regression')
    need(all(v==0 for name in ERRORS for v in after[name].values()),'Fresh server native transfer/notification/expiry failure')
    budget={tuple(sorted(x['labels'].items())):(x['requests'],x['bytes']) for x in expected_by_engine}
    need(len(budget)==len(expected_by_engine)==4 and set(budget)==keys,'Exact four per-engine transfer budgets required')
    need(all(type(n) is int and n>=0 and type(b) is int and b>=0 and bool(n)==bool(b) for n,b in budget.values())
         and sum(n for n,b in budget.values())==expected_requests and sum(b for n,b in budget.values())==expected_bytes,'Per-engine budget sum/type differs')
    need(all(changes[BYTES+'_count'][key]==n and changes[BYTES+'_sum'][key]==b for key,(n,b) in budget.items()),'Per-engine transfer coverage differs')
    count=sum(changes[BYTES+'_count'].values());size=sum(changes[BYTES+'_sum'].values())
    need(count==expected_requests and size==expected_bytes,'Native transfer count/bytes coverage differs from all fixed application handoffs')
    return {'complete':True,'native_transfer_count':int(count),'native_transfer_bytes':int(size),
            'frozen_engine_labels':expected_labels,'native_failure_counters_zero':True,
            'scope':'Source-bound off-clock native KV handoff bytes/count, not prefix-cache hits or DeepEP expert-dispatch traffic; no request-level identity inferred from counters alone'}


def source_contract(files):
    """Rehash actual pinned source bytes before accepting their accounting semantics."""
    need(set(files)==set(SOURCES),'Exact native accounting source inventory required')
    for name,path in files.items():
        path=Path(path)
        need(path.is_file() and not path.is_symlink()
             and hashlib.sha256(path.read_bytes()).hexdigest()==SOURCES[name],'Native accounting source changed: '+name)
    return dict(SOURCES)

def block_transfer_bytes(prompt_tokens):
    """Fixed current BF16 TP/DCP1, 48 layers, H4, packedC256, block16."""
    need(type(prompt_tokens) is int and prompt_tokens>0,'Positive native producer prompt length required')
    return ((prompt_tokens+15)//16)*16*48*4*256*2


def installed_sources(root):
    root=Path(root).resolve();files={name:str(root/name) for name in SOURCES}
    source_contract(files)
    return {'root':str(root),'files':files,'sha256':dict(SOURCES)}

# Default native exporter stays enabled on all arms. These GETs execute only
# outside the application completion clock; no worker/profiler hook is installed.
ENGINE_LABELS=[{'engine':str(i),'model_name':'task-model'} for i in range(4)]

def cumulative_budget(rounds,role):
    need(type(rounds) is int and 0<=rounds<=6 and role in ('prefill','decode'),'Fixed round/role budget required')
    n=64*rounds if role=='decode' else 0
    return [{'labels':x,'requests':n,'bytes':n*block_transfer_bytes(1024)} for x in ENGINE_LABELS]

def covered(parsed,budget,previous=None):
    target={tuple(sorted(x['labels'].items())):(x['requests'],x['bytes']) for x in budget}
    need(len(target)==4 and all(set(parsed[n])==set(target) for n in SELECTED),'Native exact engine label coverage differs')
    need(all(v==0 for n in ERRORS for v in parsed[n].values()),'Fresh native transfer/notification/expiry failure')
    for k,(n,b) in target.items():
        need(parsed[BYTES+'_count'][k]<=n and parsed[BYTES+'_sum'][k]<=b,'Unexpected extra native transfer/count bytes')
    if previous is not None:
        need(all(parsed[n][k]>=previous[n][k] for n in SELECTED for k in target),'Native cumulative telemetry reset/regression')
    return all(parsed[BYTES+'_count'][k]==n and parsed[BYTES+'_sum'][k]==b for k,(n,b) in target.items())

def _write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')

def capture(url,stem,stop_unix):
    import http.client,time,urllib.request,urllib.error
    stem=Path(stem);path=stem.with_suffix('.txt');transport_path=stem.with_suffix('.transport.json')
    started=time.time_ns();buffer=bytearray();headers=[];status=None;complete=False;error=None;response=None
    try:
        remaining=stop_unix-time.time();need(remaining>0,'Native metrics absolute deadline')
        try:response=urllib.request.urlopen(url+'/metrics',timeout=min(1.0,remaining))
        except urllib.error.HTTPError as failure:
            response=failure;error=repr(failure)
        status=response.status;headers=list(response.headers.items())
        lengths=[v for k,v in headers if k.lower()=='content-length']
        need(len(lengths)==1 and lengths[0].isdigit() and 0<int(lengths[0])<=4<<20
             and not any(k.lower()=='transfer-encoding' for k,v in headers),
             'Source-bound native metrics require finite Content-Length framing')
        target=int(lengths[0]);underlying=response.fp if isinstance(response,urllib.error.HTTPError) else response
        while len(buffer)<target:
            remaining=stop_unix-time.time();need(remaining>0,'Native metrics absolute read deadline')
            need(underlying.fp is not None,'Native metrics premature framing EOF')
            underlying.fp.raw._sock.settimeout(min(1.0,remaining))
            part=underlying.read1(min(65536,target-len(buffer)))
            need(part,'Native metrics premature Content-Length EOF')
            buffer.extend(part)
        need(len(buffer)==target and getattr(underlying,'length',None)==0,'Native metrics framing incomplete')
        complete=True
    except http.client.IncompleteRead as failure:buffer.extend(failure.partial);error=repr(failure)
    except BaseException as failure:error=repr(failure)
    finally:
        if response is not None:response.close()
    raw=bytes(buffer);finished=time.time_ns()
    with path.open('xb') as f:f.write(raw)
    meta={'url':url+'/metrics','started_unix_ns':started,'finished_unix_ns':finished,'http_status':status,
          'response_complete':complete,'headers':headers,'error':error,'response_bytes':len(raw),
          'response_sha256':hashlib.sha256(raw).hexdigest()}
    _write(transport_path,meta)
    need(error is None and complete and status==200 and raw,'Native metrics GET failed; all bytes/transport retained')
    types=[v for k,v in headers if k.lower()=='content-type']
    need(types==['text/plain; version=1.0.0; charset=utf-8'],'Source-bound native metrics content type differs')
    return {'path':str(path),'sha256':meta['response_sha256'],'transport_path':str(transport_path),
            'transport_sha256':hashlib.sha256(transport_path.read_bytes()).hexdigest(),'transport':meta},metrics(raw.decode('utf-8'))


def begin_round(plan,phase,number,folder):
    need(phase in ('warmup','measured') and type(number) is int and 0<=number<3,'Fixed3+3 cohort required')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    completed=number+(3 if phase=='measured' else 0);refs={};values={}
    for role in ('prefill','decode'):
        refs[role],values[role]=capture(plan[role+'_url'],folder/('before-'+role),plan['stop_unix'])
        need(covered(values[role],cumulative_budget(completed,role)),'Initial or previous drained cumulative native coverage differs')
    return {'folder':folder,'completed_rounds':completed,'before':refs,'before_values':values}

def finish_round(plan,state,completion_finished_unix_ns):
    import time
    folder=state['folder'];start=time.time();deadline=min(plan['stop_unix'],start+5.0)
    polls=[];previous=state['before_values'];target_rounds=state['completed_rounds']+1
    for index in range(10):
        due=start+index*.5
        need(due<deadline,'Native transfer coverage not complete within fixed5s drain')
        delay=due-time.time()
        if delay>0:time.sleep(delay)
        need(time.time()<deadline,'Native metrics fixed5s drain elapsed')
        refs={};values={};all_covered=True
        for role in ('prefill','decode'):
            refs[role],values[role]=capture(plan[role+'_url'],folder/(f'after-poll{index:02d}-'+role),deadline)
            need(refs[role]['transport']['started_unix_ns']>=completion_finished_unix_ns,'Native metrics GET overlaps completion clock')
            all_covered=covered(values[role],cumulative_budget(target_rounds,role),previous[role]) and all_covered
        polls.append({'index':index,'scheduled_unix':due,'roles':refs,'coverage_complete':all_covered})
        _write(folder/f'poll{index:02d}.json',polls[-1]);previous=values
        if all_covered:
            need(time.time()<=deadline,'Native metrics coverage completed after fixed5s drain')
            reports={}
            for role in ('prefill','decode'):
                budget=cumulative_budget(1,role)
                reports[role]=delta(state['before_values'][role],values[role],expected_labels=ENGINE_LABELS,
                  expected_requests=256 if role=='decode' else 0,expected_bytes=256*block_transfer_bytes(1024) if role=='decode' else 0,
                  expected_by_engine=budget)
            receipt={'complete':True,'completed_rounds_before':state['completed_rounds'],'before':state['before'],
                     'after_polls':polls,'native_coverage':reports,'completion_finished_unix_ns':completion_finished_unix_ns,
                     'drain_started_unix':start,'drain_deadline_unix':deadline,'finished_unix_ns':time.time_ns(),
                     'poll_policy':'Immediate then deterministic0.5s schedule, at most10polls/5seconds; every rawGET retained, no cohort retry/filter',
                     'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}}
            _write(folder/'receipt.json',receipt)
            return {'path':str(folder/'receipt.json'),'sha256':hashlib.sha256((folder/'receipt.json').read_bytes()).hexdigest(),
                    'complete':True,'native_coverage':reports}
    raise ValueError('Native transfer coverage not complete within fixed5s drain; all polls retained')
