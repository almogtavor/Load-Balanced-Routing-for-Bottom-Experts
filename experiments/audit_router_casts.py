#!/usr/bin/env python3
"""Read-only trace audit of int32->int64 routing-ID copies inside MoE scopes."""
import argparse
from collections import defaultdict,Counter
import gzip,json,bisect
from pathlib import Path
from analyze_profiles import find_raw,distribution


def audit(path,rank,arm,windows):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt') as f: trace=json.load(f)
    scopes=defaultdict(list);copies=[];apis={};kernels=[]; to_copy=[]
    for e in trace['traceEvents']:
        if e.get('ph')!='X':continue
        cat=e.get('cat');name=e.get('name','');a=e.get('args',{})
        if cat=='cpu_op' and name=='vllm::moe_forward':
            dims=a.get('Input Dims',[])
            scopes[(e['pid'],e['tid'])].append((e['ts'],e['ts']+e['dur'],dims[0][0] if dims and dims[0] else None))
        if cat=='cpu_op' and name in ('aten::copy_','aten::_to_copy','aten::to'):
            dims=a.get('Input Dims',[]);types=a.get('Input type',[])
            if dims and len(dims[0])==2 and dims[0][1]==8:
                if name=='aten::copy_' and len(types)>=2 and types[0]=='long int' and types[1]=='int':copies.append(e)
                elif name=='aten::copy_' and len(types)>=2 and 'long' in types[0].lower() and types[1].lower() in ('int','int32','int32_t'):copies.append(e)
                if name=='aten::_to_copy' and types and types[0].lower() in ('int','int32','int32_t'):to_copy.append(e)
        if cat in ('cuda_runtime','cuda_driver') and a.get('correlation') is not None:
            apis[a['correlation']]=e
        if cat=='kernel':kernels.append(e)
    starts={}
    for thread,v in scopes.items():v.sort();starts[thread]=[x[0] for x in v]
    def in_moe(e):
        thread=(e['pid'],e['tid']);v=scopes.get(thread,[])
        i=bisect.bisect_right(starts.get(thread,[]),e['ts'])-1
        return v[i] if i>=0 and e['ts']+e['dur']<=v[i][1]+.01 else None
    copies=[e for e in copies if in_moe(e)];to_copy=[e for e in to_copy if in_moe(e)]
    by_external={e['args'].get('External id'):e for e in copies}
    copies_by_thread=defaultdict(list)
    for copy in copies:copies_by_thread[(copy['pid'],copy['tid'])].append(copy)
    for values in copies_by_thread.values():values.sort(key=lambda x:x['ts'])
    copy_starts={key:[x['ts'] for x in values] for key,values in copies_by_thread.items()}
    gpu=[]
    for e in kernels:
        a=e.get('args',{});copy=by_external.get(a.get('External id'))
        if copy:
            gpu.append((e,copy))
        else:
            launch=apis.get(a.get('correlation'))
            if launch:
                thread=(launch['pid'],launch['tid'])
                position=bisect.bisect_right(copy_starts.get(thread,[]),launch['ts'])-1
                if position>=0:
                    copy=copies_by_thread[thread][position]
                    if copy['ts']<=launch['ts']<=copy['ts']+copy['dur']:
                        gpu.append((e,copy))
    out={'arm':arm,'rank':rank,'source':str(path),'moe_calls':sum(len(v) for v in scopes.values()),
         'int32_to_int64_T8_copy_cpu_calls':len(copies),'int32_T8_to_copy_parent_cpu_calls':len(to_copy),
         'correlated_cast_gpu_kernels':len(gpu),'cast_gpu_kernel_names':dict(Counter(e['name'] for e,c in gpu)),
         'CPU_copy_inclusive_us':{'sum':sum(e['dur'] for e in copies),**(distribution([e['dur'] for e in copies]) or {})},
         'CPU_to_copy_parent_inclusive_us':{'sum':sum(e['dur'] for e in to_copy),**(distribution([e['dur'] for e in to_copy]) or {})},
         'GPU_cast_us':{'sum':sum(e['dur'] for e,c in gpu),**(distribution([e['dur'] for e,c in gpu]) or {})},
         'by_source_T':{},'workloads':{},
         'example_copy_events':copies[:2],
         'example_to_copy_events':to_copy[:2]}
    byT=defaultdict(lambda:{'cpu':[],'gpu':[],'parent':[]})
    for e in copies:byT[e['args']['Input Dims'][0][0]]['cpu'].append(e['dur'])
    for e in to_copy:byT[e['args']['Input Dims'][0][0]]['parent'].append(e['dur'])
    for e,c in gpu:byT[c['args']['Input Dims'][0][0]]['gpu'].append(e['dur'])
    for t,d in byT.items():out['by_source_T'][str(t)]={k:{'count':len(v),'sum_us':sum(v),'distribution_us':distribution(v)} for k,v in d.items()}
    base=trace['baseTimeNanoseconds']
    for name,w in windows.items():
        lo,hi=((w[k]-base)/1000 for k in ('started_unix_ns','finished_unix_ns'))
        out['workloads'][name]={'profile_wall_us':hi-lo,
            'CPU_to_copy_parent_us':sum(e['dur'] for e in to_copy if lo<=e['ts']<hi),
            'CPU_copy_us':sum(e['dur'] for e in copies if lo<=e['ts']<hi),
            'GPU_cast_us':sum(e['dur'] for e,c in gpu if lo<=e['ts']<hi),
            'GPU_cast_calls':sum(lo<=e['ts']<hi for e,c in gpu)}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path,nargs='+');p.add_argument('--trace-root',type=Path,action='append',default=[]);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    output={'limits':'Instrumented CPU times include profiler overhead. aten::_to_copy contains copy_ and allocation; do not add parent/child durations. GPU copies must correlate to typed CPU copy inside MoE, not merely match kernel name.','ranks':[]}
    if a.resume and a.output.exists():output=json.loads(a.output.read_text())
    for result in a.results:
        r=json.load(result.open())
        for s in r['trace_summary']['traces']:
            if s.get('dp_rank') is None:continue
            if any(x['arm']==r['arm'] and x['rank']==s['dp_rank'] for x in output['ranks']):continue
            path=find_raw(s['path'],[result.parent,*a.trace_root])
            if path is None:raise FileNotFoundError(s['path'])
            output['ranks'].append(audit(path,s['dp_rank'],r['arm'],r['shapes']))
            a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,indent=2)+'\n')
            print(r['arm'],s['dp_rank'],output['ranks'][-1]['int32_to_int64_T8_copy_cpu_calls'],output['ranks'][-1]['correlated_cast_gpu_kernels'],flush=True)
if __name__=='__main__':main()
