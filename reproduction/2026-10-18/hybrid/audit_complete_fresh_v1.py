#!/usr/bin/env python3
"""One full3+9+3 offline acceptance report; no partial effects substituted."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
QUALITY_SHA='0859ca8aad1498785ce0431cc3662b4800ced02281d08522053032b1a73c72bd'
ACCEPTANCE_SHA='9b1094cd0d3521efc595cb51c45976228e3d08c53f7de8861a1bdfba7a1a73ab'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def need(value,message):
 if not value:raise ValueError(message)
def quality_helper():
 path=HERE/'audit_quality_fresh_v1.py';need(sha(path)==QUALITY_SHA,'Frozen full-quality reader differs')
 spec=importlib.util.spec_from_file_location('fresh_hybrid_full_quality_gate',path)
 value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
def audit(root,plan_sha,bindings,statistics,protocol,acceptance):
 """No effect routine before all15 closed pairs and full raw quality pass."""
 q=quality_helper();root=Path(root).resolve()
 need(sha(acceptance)==ACCEPTANCE_SHA,'Prospective complete-matrix acceptance differs')
 need(sha(root/'plan.json')==plan_sha,'Operational plan bytes differ');plan=q.load(root/'plan.json')
 with q.frozen() as (raw,scorer):
  bundle=raw.Bundle(root,plan,bindings);raw.audit_plan(bundle,plan,plan_sha)
  need(plan['files'][plan['quality_inputs']]==q.FULL_SHA,'Full canonical GSM input differs')
  modes,launches=q.closed_modes(bundle,plan,plan_sha,raw)
  rows=bundle.json(plan['quality_inputs']);gsm=[r for r in rows if r['task']=='gsm8k']
  need(len(gsm)==1319 and all(r['split']=='test' for r in gsm)
       and [r['id'] for r in gsm]==['gsm8k:test:'+str(i) for i in range(1319)],'Full canonical held-out set required')
  # All15 owned pairs/source/exit and all6 raw native phases, then every
  # raw full1319 score pass before the timing reader can compute effects.
  proofs,phases,scores=q.replay_all(bundle,plan,launches,raw,scorer)
  timing=raw.audit_campaign(root,plan_sha,bindings,statistics,protocol)
  need(timing['complete'] is True,'Full54-cohort timing raw replay incomplete')
  contrasts=q.observed_contrasts(scores)
  for name,item in list(bundle.checked.items()):bundle.checked_file(name,item['sha256'])
  need(sha(HERE/'audit_quality_fresh_v1.py')==QUALITY_SHA and sha(acceptance)==ACCEPTANCE_SHA,'Frozen acceptance source changed during replay')
  return {'schema':'complete-fresh-hybrid-acceptance-v1','complete':True,
   'all_fifteen_closed_pairs_and_full_raw_before_comparative_effects':True,
   'plan_sha256':plan_sha,'bindings_sha256':sha(bindings),'acceptance_protocol_sha256':ACCEPTANCE_SHA,
   'quality_reader_sha256':QUALITY_SHA,'reader_sha256':sha(__file__),'frozen_core_sha256':q.CORE_SHA,
   'native_training_proofs':proofs,'native_quality_phase':phases,'quality_scores':scores,
   'quality_observed_paired_contrasts':contrasts,'timing':timing,'checked_quality_inputs':bundle.checked,
   'qualifications':[
    'All timing contrasts use fresh stock-prefill/stock-decode handoff on this allocation, with stock decode in every arm.',
    'Three paired sweep units on one H200 host; inherited timing t intervals have two degrees of freedom and are descriptive.',
    'One fresh full1319 GSM8K server pair per arm; item counts and percentage-point differences are finite-corpus observations, without a launch-population confidence interval.',
    'No pooling with old partial V6, public monolithic quality or H100 prefill results; failed V5 output-equivalence remains failed.',
    'No monolithic numerical equivalence, monolithic accuracy safety, SLO or clean-run KV tensor-integrity claim.',
    'Native phase proof and off-clock timed transfer-count/byte coverage have distinct observer scopes; synchronized KV checks do not rule out an uninstrumented visibility race.',
    'All raw rounds, errors, length caps, empty outputs and parsing failures are retained by the unchanged frozen readers.']}
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--root',type=Path,required=True);p.add_argument('--plan-sha256',required=True)
 p.add_argument('--bindings',type=Path,required=True);p.add_argument('--bindings-sha256',required=True)
 p.add_argument('--statistics',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
 p.add_argument('--acceptance',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 need(not a.output.resolve().is_relative_to(a.root.resolve()),'Output must be outside immutable raw campaign')
 try:
  need(sha(a.bindings)==a.bindings_sha256,'Explicit external bindings changed')
  result=audit(a.root,a.plan_sha256,a.bindings,a.statistics,a.protocol,a.acceptance)
 except BaseException:
  result={'schema':'complete-fresh-hybrid-acceptance-v1','complete':False,'comparative_effects':None,
   'error':traceback.format_exc(),'reader_sha256':sha(__file__),
   'interpretation':'No partial performance or quality contrast accepted; complete fixed matrix and all raw gates required.'}
 with a.output.open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
 raise SystemExit(0 if result['complete'] else 2)
if __name__=='__main__':main()
