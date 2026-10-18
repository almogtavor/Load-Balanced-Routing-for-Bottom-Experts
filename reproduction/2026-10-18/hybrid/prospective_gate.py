"""Prospective V6 admission: preserve failure and require closed native diagnostic."""
from pathlib import Path
from common_v2 import read, require, file_sha
import archived_h200_closure as predecessor

ANALYSIS='/work/results/provenance/stock-kv-numerics-raw-analysis-v1.json'
FILES={
 ANALYSIS:'7a5c5d1e9e2ecbd865649c21375311343295a805528213af2e13005daf17ae2d',
 '/work/results/hybrid-v5/proof/receipt.json':'045c2c82d44c85a97c8f5157b355d5d04cae7d8097eb4f16501dac0c5ac58685',
 '/work/results/hybrid-outer-driver-v5.json':'c93882ae4a37abe5b64817310e970289b9763945598cabe4d1404b0b6c29afa7',
 '/work/results/hybrid-v5/proof/sweep00-position0-stock/launch.json':'cd4b5b327b9f8d643d3bd2517b8cfa32c2031b98b2141d56bcb717826d336ce4',
 '/work/results/hybrid-stock-numerics-v1/plan.json':'8de14ef658ce14f322d2e5beff3138707708ccc21725625880603209f6d8770c',
 '/work/results/hybrid-stock-numerics-v1/diagnostic/receipt.json':'af27fde722e3d389a8bed8912185d35653cb16a10c2a8844f1f8ebc2d4484b8f',
 '/work/results/hybrid-stock-numerics-v1/diagnostic/stock-pair/launch.json':'dcf52f8d0dd812f774eab3acfc8d6c3959ef07a1d78fa4c16fd8f6b9b64f24ab'}
FILES.update(predecessor.FILES)
FILES['/work/gap/hybrid/archived_h200_closure.py']='72515b110e24f0f7e9633a8c101c36ef9ef2b4c4957172848c5d2ef6558c39a1'
POLICY={'revision':6,'stock_monolithic_reference':'Diagnostic only; retain every token/finish difference, no exact-output equivalence or benign-noise claim',
 'comparison_baseline':'Actual stock-prefill plus stock-decode NIXL handoff on same eight H200 GPUs',
 'failed_v5_preserved':True,'all_three_heldout_arms_required':True,
 'native_cached_tokens':'Remote producer prefix-cache hits, exact integer zero; not transferred prefix coverage',
 'metrics':'Off-clock baseline and deterministic0.5s polls for at most5s after each fixed cohort; exact per-engine transfer sum/count and zero failures'}

def verify(lib):
    # Prior PIDs belong to the released predecessor namespace. Never query them here.
    require(predecessor.verify()==predecessor.SUMMARY,'Preserved predecessor closure required')
    for name,digest in FILES.items():
        p=Path(name);require(p.is_file() and not p.is_symlink() and file_sha(p)==digest,'Required preserved diagnostic/failure changed: '+name)
    analysis=read(ANALYSIS)
    require(analysis['complete'] is True and analysis['diagnostic_only'] is True
      and analysis['original_v5_campaign_complete'] is False and analysis['performance_effects'] is None
      and analysis['all_five_cohorts_raw_replayed'] is True and analysis['all_eight_handoffs_all48_layers'] is True
      and analysis['all_valid_prefix_sha256_equal'] is True and analysis['all_full_prompt_sha256_equal_supplemental'] is True
      and analysis['kv_valid_prefix_pairs']==384 and analysis['application_requests']==20
      and analysis['completion_http_requests']==28 and analysis['native_transfer_bytes']==600834048
      and analysis['plan_sha256']==FILES['/work/results/hybrid-stock-numerics-v1/plan.json'],
      'Complete reviewed native KV/numerical diagnostic required')
    for path in ['/work/results/hybrid-v5/proof/receipt.json','/work/results/hybrid-outer-driver-v5.json']:
        receipt=read(path)
        require(receipt['complete'] is False and receipt['cleanup_complete'] is True ,
                'Failed V5 must remain failed/closed in the released predecessor archive')
    receipt=read('/work/results/hybrid-stock-numerics-v1/diagnostic/receipt.json')
    require(receipt['complete'] is True and receipt['cleanup_complete'] is True and receipt['cleanup_errors']==[]
        and receipt['original_v5_campaign_complete'] is False and receipt['performance_effects'] is None
        and not receipt['active_servers'] 
        and len(receipt['launches'])==1 and receipt['launches'][0]['complete'] is True,
        'Stock numerical diagnostic must remain closed in the released predecessor archive')
    return {'files_sha256':dict(FILES),'policy':dict(POLICY)}
