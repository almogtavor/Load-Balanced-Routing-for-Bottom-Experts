"""Fresh owner plus preserved predecessor closure; no archived PID lookups."""
import hashlib,json
from pathlib import Path,PurePosixPath
_ARCHIVED_SHA='72515b110e24f0f7e9633a8c101c36ef9ef2b4c4957172848c5d2ef6558c39a1'
if hashlib.sha256(Path(__file__).with_name('archived_h200_closure.py').read_bytes()).hexdigest()!=_ARCHIVED_SHA:
    raise RuntimeError('Archived H200 closure helper changed before import')
import archived_h200_closure as predecessor

OLD={'namespace':'itay','pod':'par-followup-h100-20261018','uid':'5f650599-18e1-42f5-8847-fbdeca1f12b1'}
OWNER={'namespace':'default','pod':'par-gap-hybrid-h200-20261018','uid':'8e098972-bdeb-4589-b79f-324bfd601784','node':'g134dfa',
       'image':'vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b'}
RESULTS='/work/recovered-primary-results'
SOURCE='/work/recovered-primary-source'
ARCHIVES={'results':{'sha256':'262b0b836104e0c1241b8a046ff45baa1d7b838555fc07933951663b9e375460','bytes':1139058934},
          'source':{'sha256':'a35151418510ec2fa913ddd14ffd2c4beac89811a9fdcad2cd21af6da70f027b','bytes':33536950}}
CLOSURE='Exact archived cleanup plus source pod NotFound; no replacement-pod PID inference'

def need(v,m):
    if not v:raise RuntimeError(m)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())

def prior_path(original):
    p=PurePosixPath(str(original))
    need(str(p)==str(original) and p.is_relative_to('/work/results') and '..' not in p.parts,'Only original results paths can bind to recovered evidence')
    return Path(RESULTS).joinpath(*p.relative_to('/work/results').parts)

def original_path(recovered):
    p=PurePosixPath(str(recovered))
    need(str(p)==str(recovered) and p.is_relative_to(RESULTS) and '..' not in p.parts,'Not a recovered results path')
    return str(PurePosixPath('/work/results').joinpath(*p.relative_to(RESULTS).parts))

def verify_value(proof,allocation):
    need(OWNER['uid'] != 'PENDING_FRESH_ALLOCATION' and OWNER['uid'] != predecessor.OLD['uid'], 'Fresh exact owner must be frozen before planning')
    need(proof.get('previous_h200_closure') == predecessor.SUMMARY, 'Fresh owner must preserve and exclude previous partial H200 matrix')
    need(proof.get('complete') is True and proof['schema']=='hybrid-primary-owner-recovery-v1'
         and proof['original_pod']==OLD and proof['recovered_results_root']==RESULTS and proof['recovered_source_root']==SOURCE,
         'Recovery source identity/roots differ')
    for name,expected in ARCHIVES.items():
        actual=proof[name+'_archive']
        need(all(actual[k]==v for k,v in expected.items()) and actual['verified'] is True,'Actual verified recovery archive differs')
    absent=proof['old_pod_not_found'];command=absent['command']
    need(isinstance(command,list) and len(command) in (7,8) and command[0].endswith('/.local/bin/oc-llmd-roce')
         and command[1:6]==['-n','itay','get','pod',OLD['pod']]
         and command[6:] in (['-ojson'],['-o=json'],['-o','json'])
         and type(absent['returncode']) is int and absent['returncode']!=0
         and absent['stdout']=='' and absent['stderr'].strip()=='Error from server (NotFound): pods "'+OLD['pod']+'" not found'
         and type(absent['observed_unix_ns']) is int and absent['observed_unix_ns']>0
         and absent['expected_uid']==OLD['uid'],'Original exact owned pod absence is not established; transport/auth errors never imply exit')
    owner=proof['replacement_pod'];hardware=proof['replacement_hardware']
    need(owner==OWNER and allocation['verified_live'] is True and all(allocation[k]==v for k,v in owner.items())
         and set(hardware)=={'gpu_count','gpu_name'} and hardware['gpu_count']==8
         and isinstance(hardware['gpu_name'],str) and 'H200' in hardware['gpu_name']
         and all(allocation[k]==v for k,v in hardware.items()),'Fresh current H200 allocation differs from frozen recovery owner/hardware')
    return {'original_pod':OLD,'replacement_pod':owner,'replacement_hardware':hardware,
            'prior_controller_closure':CLOSURE,'recovered_results_root':RESULTS,'recovered_source_root':SOURCE,'archives':ARCHIVES,'previous_h200_closure':predecessor.SUMMARY}

def verify(path,expected_sha,allocation):
    need(predecessor.verify()==predecessor.SUMMARY,'Preserved H200 closure changed')
    need(sha(path)==expected_sha,'Frozen owner/recovery receipt changed')
    return {'proof_path':str(Path(path).resolve()),'proof_sha256':expected_sha,**verify_value(read(path),allocation)}
