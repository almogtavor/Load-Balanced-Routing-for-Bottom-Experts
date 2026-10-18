#!/usr/bin/env python3
"""Stage-0 proof of DeepEP legacy fixed-receive CUDA graph correctness.

  OMP_NUM_THREADS=1 timeout 300 torchrun --standalone --nproc_per_node=4 \
    experiments/replay_deepep_graph.py --output /results/deepep_graph_probe.json

No vLLM/runtime patch and no expert GEMM. Proves a GPU actual-receive-count
formula against known asymmetric all-rank routes before using it inside capture.
Uses fixed receive capacity R*Tcap, poisons padding with NaNs/valid-looking IDs,
masks IDs/weights/activations from the proven GPU count, and verifies weighted
identity after changing input tensors in place across graph replays. Source
padding, zero-receive destination ranks and concentrated/diffuse routes are tested.
Failure is a failed feasibility gate, not permission to force serving graphs.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def case_routes(case, source, cap, experts, world=4):
    active_vectors = [[cap, cap-3, cap//2, 0], [0, cap//3, cap, cap-1],
                      [cap]*world, [1, cap-2, 0, cap//2]]
    active = active_vectors[case][source]
    block = experts // world
    ids, sends = [], [0]*world
    for row in range(cap):
        if row >= active:
            ids.append([-1]*8)
            continue
        if case == 0:
            destinations = [(source + row % 3) % world]
        elif case == 1:
            destinations = [source, (source + 1 + row % 2) % world]
        elif case == 2:
            destinations = list(range(world))
        else:
            destinations = [0]
        picks = []
        for destination in destinations:
            sends[destination] += 1
            for slot in range(8 // len(destinations)):
                picks.append(destination*block + (source*5 + row + slot) % block)
        assert len(set(picks)) == 8
        ids.append(picks)
    return ids, sends, active


def matching_prefix_formulas(raw, sends, rank):
    """Prove every source prefix in the destination rank's owned slice.

    The legacy dispatch only initializes this destination's column of its
    rank-prefix allocation. Other columns may contain allocator garbage and
    are not a replicated all-destination matrix. Test the transposed possibility
    as well, and require the same unique interpretation across asymmetric cases.
    """
    world = len(sends)
    if not 0 <= rank < world or len(raw) != world or any(
            len(row) != world for row in raw + sends):
        raise ValueError('Expected square all-rank matrices and a valid rank')
    expected, total = [], 0
    for source in range(world):
        total += sends[source][rank]
        expected.append(total)
    matches = []
    if [row[rank] for row in raw] == expected:
        matches.append('source_prefix_rows')
    if raw[rank] == expected:
        matches.append('destination_prefix_columns')
    return matches, expected


def main():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--capacity',type=int,default=16)
    p.add_argument('--hidden',type=int,default=2048)
    p.add_argument('--experts',type=int,default=128)
    p.add_argument('--buffer-mb',type=int,default=1024)
    p.add_argument('--replays',type=int,default=8)
    p.add_argument('--timeout-seconds',type=int,default=180)
    a=p.parse_args()
    if a.capacity<4 or a.hidden%256 or a.experts%4 or a.experts//4<8:
        p.error('Require capacity>=4, BF16 hidden divisible by256, experts divisible by4 and >=8/rank')
    if min(a.buffer_mb,a.replays,a.timeout_seconds)<1:p.error('buffer/replays/timeout must be positive')
    import torch
    import torch.distributed as dist
    rank,world,local_rank=(int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK'))
    if world!=4:p.error('Exactly four NVLink ranks are required')
    def timed_out(signum, frame):raise TimeoutError('Graph feasibility probe host watchdog expired')
    signal.signal(signal.SIGALRM,timed_out);signal.alarm(a.timeout_seconds)
    device=torch.device('cuda',local_rank);torch.cuda.set_device(device);torch.set_num_threads(1)
    dist.init_process_group('nccl',device_id=device,timeout=datetime.timedelta(seconds=a.timeout_seconds))
    cpu_group=dist.new_group(backend='gloo',timeout=datetime.timedelta(seconds=a.timeout_seconds))
    import deep_ep
    if deep_ep.__version__!='2.0.0':raise RuntimeError('Only installed/verified DeepEP2.0.0 legacy API supported')
    buffer=deep_ep.Buffer(group=cpu_group,num_nvl_bytes=a.buffer_mb*1024*1024,
        num_rdma_bytes=0,low_latency_mode=False,num_qps_per_rank=1,explicitly_destroy=True)
    dc,cc=deep_ep.Buffer.get_dispatch_config(world),deep_ep.Buffer.get_combine_config(world)
    cap=a.capacity*world
    result={'complete':False,'stage':'initialize','kind':'stage0_fixed_receive_graph_correctness_only',
            'configuration':vars(a)|{'output':str(a.output),'world':world,'receive_capacity':cap,'deep_ep':deep_ep.__version__},
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'discovery':[],'eager_checks':[],'graph_checks':[],
            'limits':'No vLLM patch, expert GEMM, graph guard change, model quality evaluation or serving performance claim. Prefix formula is accepted only after every source prefix in the rank-owned slice matches all asymmetric cases/ranks; other columns are not assumed initialized.'}
    def checkpoint(stage):
        result['stage']=stage
        if rank==0:save(a.output,result)
    def gather(value):
        values=[None]*world;dist.all_gather_object(values,value,group=cpu_group);return values
    stream=torch.cuda.Stream(device=device)
    try:
        with torch.cuda.stream(stream):
            cases=[]
            for case in range(4):
                all_sources=[case_routes(case,s,a.capacity,a.experts,world) for s in range(world)]
                cpu_ids,sends,active=all_sources[rank]
                ids=torch.tensor(cpu_ids,device=device,dtype=deep_ep.topk_idx_t)
                weights=torch.where(ids>=0,1/8,0).float()
                generator=torch.Generator(device=device).manual_seed(20261017+rank*1009+case*37)
                x=torch.randn((a.capacity,a.hidden),generator=generator,device=device,dtype=torch.float32).to(torch.bfloat16)
                expected=x.clone();expected[active:]=0
                cases.append({'ids':ids,'weights':weights,'x':x,'expected':expected,
                    'send_matrix':[s[1] for s in all_sources],
                    'expected_received':sum(s[1][rank] for s in all_sources),'active_source_rows':active})
            ids=torch.empty_like(cases[0]['ids']);weights=torch.empty_like(cases[0]['weights']);x=torch.empty_like(cases[0]['x'])
            def apply_case(case):
                ids.copy_(cases[case]['ids']);weights.copy_(cases[case]['weights']);x.copy_(cases[case]['x'])
            def dispatch_fixed():
                previous=buffer.capture()
                n_rank,n_rdma,n_expert,membership,_=buffer.get_dispatch_layout(
                    topk_idx=ids,num_experts=a.experts,previous_event=previous,
                    async_finish=False,allocate_on_comm_stream=False)
                received,ri,rw,counts,handle,event=buffer.dispatch(
                    x=x,handle=None,num_tokens_per_rank=n_rank,num_tokens_per_rdma_rank=n_rdma,
                    is_token_in_rank=membership,num_tokens_per_expert=n_expert,topk_idx=ids,topk_weights=weights,
                    expert_alignment=1,num_worst_tokens=cap,config=dc,previous_event=previous,
                    async_finish=True,allocate_on_comm_stream=False)
                if event.event is not None:event.current_stream_wait()
                if counts:raise AssertionError('Fixed-receive returned CPU expert counts; no-CPU-sync contract absent')
                if received.shape!=(cap,a.hidden) or ri.shape!=(cap,8) or rw.shape!=(cap,8):
                    raise AssertionError(f'Fixed-shape contract absent: {received.shape}/{ri.shape}/{rw.shape}')
                return received,ri,rw,handle
            def combine(value,handle):
                output,_,event=buffer.combine(x=value,handle=handle,topk_weights=None,config=cc,
                    previous_event=buffer.capture(),async_finish=True,allocate_on_comm_stream=False)
                if event.event is not None:event.current_stream_wait()
                return output
            candidates={'source_prefix_rows','destination_prefix_columns'}
            for case,c in enumerate(cases):
                checkpoint(f'discover_case_{case}');apply_case(case);stream.synchronize();dist.barrier(group=cpu_group)
                received,ri,rw,handle=dispatch_fixed();stream.synchronize()
                matrix=handle[0]
                if tuple(matrix.shape)!=(world,world):raise AssertionError(f'Unknown prefix matrix shape {matrix.shape}')
                raw=matrix.cpu().tolist()
                valid,expected_prefix=matching_prefix_formulas(raw,c['send_matrix'],rank)
                proof={'case':case,'rank':rank,'known_send_matrix':c['send_matrix'],'raw_rank_prefix_matrix':raw,
                    'expected_received':c['expected_received'],'matching_matrix_formulas':valid,
                    'expected_owned_source_prefix':expected_prefix,
                    'proof_scope':'All source entries in this destination-owned slice; unrelated matrix entries may be uninitialized.'}
                proofs=gather(proof);result['discovery'].extend(proofs)
                for q in proofs:candidates.intersection_update(q['matching_matrix_formulas'])
                # Complete the handle's cycle before issuing another dispatch.
                combine(torch.zeros_like(received),handle);stream.synchronize()
            if len(candidates)!=1:
                checkpoint('unsupported_or_ambiguous_gpu_count_formula')
                raise AssertionError(f'Could not uniquely prove prefix semantics: {sorted(candidates)}; inspect discovery matrices')
            formula=next(iter(candidates));result['proven_formula']=formula
            result['gpu_count_expression']='handle[0][-1, rank]' if formula=='source_prefix_rows' else 'handle[0][rank, -1]'
            rows=torch.arange(cap,device=device)
            def identity_roundtrip():
                received,ri,rw,handle=dispatch_fixed()
                count=handle[0][-1,rank] if formula=='source_prefix_rows' else handle[0][rank,-1]
                valid=rows<count
                # Poison precisely the device-identified padding region. Invalid
                # tails deliberately look like real local experts and carry NaNs.
                received.masked_fill_(~valid[:,None],float('nan'))
                ri.masked_fill_(~valid[:,None],0)
                rw.masked_fill_(~valid[:,None],99.)
                safe_ids=torch.where(valid[:,None]&(ri>=0)&(ri<a.experts//world),ri,-1)
                safe_weights=torch.where(safe_ids>=0,rw,0.)
                safe_x=torch.where(valid[:,None],received,0.)
                identity=(safe_x.float()*safe_weights.sum(-1,keepdim=True)).to(torch.bfloat16)
                output=combine(identity,handle)
                return output,count,safe_ids,safe_weights,safe_x
            def validate(tensors,case,phase,index):
                output,count,safe_ids,safe_weights,safe_x=tensors
                stream.synchronize()
                actual=int(count.item());expected=cases[case]['expected_received']
                assert actual==expected,(phase,case,rank,actual,expected)
                assert bool(torch.isfinite(safe_x).all()),'NaN poisoning escaped activation mask'
                assert bool((safe_ids[actual:]==-1).all()),'Padding IDs look valid'
                assert bool((safe_weights[actual:]==0).all()),'Padding weights nonzero'
                assert bool((safe_x[actual:]==0).all()),'Padding activations nonzero'
                torch.testing.assert_close(output,cases[case]['expected'],atol=.04,rtol=.02)
                checks=gather({'rank':rank,'case':case,'index':index,'actual_received':actual,
                    'expected_received':expected,'padding_rows':cap-actual,'padding_poison_masked':True,
                    'identity_max_abs_error':float((output.float()-cases[case]['expected'].float()).abs().max())})
                result[phase+'_checks'].extend(checks)
                checkpoint(f'{phase}_validated_{index}')
            for case in range(4):
                apply_case(case);stream.synchronize();dist.barrier(group=cpu_group)
                validate(identity_roundtrip(),case,'eager',case)
            apply_case(0);stream.synchronize();dist.barrier(group=cpu_group)
            checkpoint('capture_started')
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):captured=identity_roundtrip()
            checkpoint('capture_finished')
            order=[0,3,1,2,0,2,3,1]
            for index in range(a.replays):
                case=order[index%len(order)];apply_case(case);stream.synchronize();dist.barrier(group=cpu_group)
                checkpoint(f'replay_started_{index}_case_{case}')
                graph.replay();validate(captured,case,'graph',index)
            result['complete']=True;checkpoint('passed')
    except BaseException as e:
        result['error']=f'{type(e).__name__}: {e}'
        if rank==0:save(a.output,result)
        raise
    finally:
        signal.alarm(0)
        torch.cuda.synchronize(device);buffer.destroy()
        dist.destroy_process_group(cpu_group);dist.destroy_process_group()

if __name__=='__main__':main()
