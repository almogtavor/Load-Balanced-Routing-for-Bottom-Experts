#!/usr/bin/env python3
"""Standalone archive-only external bindings; no outcomes interpreted.

Copies only required external bytes into SHA-named objects. Current canonical
paths, actual native source, and the opaque11 historical inputs retain their
original remote names. No local predecessor archive is a runtime fallback.
"""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,shutil,time
HERE=Path(__file__).resolve().parent
UID='8e098972-bdeb-4589-b79f-324bfd601784'
PLAN_SHA='2e9e5b7f03946381a8d59fbd80535ef5b29d5d63b1520e985ce2df3dc1c1f4b5'
REQUEST_SHA='5d66fa4d9ccbac46f223d6e81d35301eb64638e68a9122157245f8731e99ba54'
FINALIZER_SHA='e65584df654bf9a956322ae3e13f65d657b26775ab4a79156bce9db968fba4d2'
PRESERVER_SHA='52b500ffff4468e45d7f9cc1f6d429ccabafe954f783580f21e17e773aed2de0'
REMOTE='/work/results/gap-hybrid-v1'
CAPSULE='/work/results/provenance/hybrid-historical-bytecode-v2'
def need(v,m):
 if not v:raise ValueError(m)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def load(path):return json.loads(Path(path).read_bytes())
def regular(path,root):
 path=Path(path);root=Path(root).resolve()
 need(path.is_file() and path.resolve().is_relative_to(root) and not any(p.is_symlink()for p in [path,*path.parents]),'Missing/escaping/nonregular source')
 return path
def relative(name):
 p=PurePosixPath(name);need(p.is_absolute()and'..'not in p.parts and str(p)==name,'Unsafe remote path');return p

def resolver(files,freeze,capsule):
 files=Path(files).resolve();rows={r['source']:r for r in freeze['files']}
 need(len(rows)==len(freeze['files']),'Duplicate frozen source rows')
 need(capsule['complete'] is True and capsule['schema']=='hybrid-historical-bytecode-capsule-v2'
      and capsule['pod_uid']==UID and capsule['helper_sha256']==PRESERVER_SHA
      and capsule['request']=={'path':'/work/gap/hybrid/hybrid-outer-request-fresh-v1.json','sha256':REQUEST_SHA}
      and len(capsule['files'])==11,'Exact opaque11 capsule absent/different')
 regular(files/'provenance/hybrid-historical-bytecode-v2/preserver-source.py',files)
 need(sha(files/'provenance/hybrid-historical-bytecode-v2/preserver-source.py')==PRESERVER_SHA,'Actual preserver source changed')
 def resolve(name):
  p=relative(name)
  if p.is_relative_to('/work/results'):out=files/p.relative_to('/work/results')
  elif name in capsule['files']:
   row=capsule['files'][name];copy=relative(row['copy'])
   need(name.startswith('/work/recovered-primary-source/')and name.endswith('.pyc')and '/__pycache__/'in name
        and copy.parent==PurePosixPath(CAPSULE+'/objects')and copy.name==row['sha256'],'Unsafe opaque11 mapping')
   out=files/copy.relative_to('/work/results');need(sha(regular(out,files))==row['sha256'],'Opaque copy hash differs')
  else:
   need(name in rows,'Missing actual frozen source mapping: '+name);r=rows[name];copy=Path(r['copy'])
   need(not copy.is_absolute()and'..'not in copy.parts,'Unsafe frozen copy path');out=files/copy
   need(sha(regular(out,files))==r['sha256']and out.stat().st_size==r['bytes'],'Frozen source copy differs')
  return regular(out,files)
 return resolve

def build(marker_path,output):
 marker_path=Path(marker_path).resolve();m=load(marker_path);source=m['source']
 need(m['final_verified'] is True and m['writers_stopped_asserted'] is True and m['remote_verification']['ok'] is True
      and source['pod_uid']==UID and source['pod']=='par-gap-hybrid-h200-20261018' and source['namespace']=='default'
      and source['root']=='/work/results' and source['container']=='experiment','Wrong/incomplete whole capture')
 archive=regular(marker_path.parent/'results.tar.gz',marker_path.parent)
 need(archive.stat().st_size==m['archive_bytes']and sha(archive)==m['archive_sha256'],'Whole archive changed')
 extraction=Path(m['extracted_directory']).resolve();need(extraction.parent==marker_path.parent,'Unexpected extraction root')
 files=extraction/'files';freeze=load(regular(files/'provenance/final-inputs-freeze-v1.json',files))
 need(freeze['complete'] is True and freeze['pod_uid']==UID and freeze['finalizer_sha256']==FINALIZER_SHA
      and freeze['writers_stopped'] is True and freeze['gpu_idle'] is True and freeze['allocation']['gpu_count']==8,'Final current input freeze required')
 capsule=load(regular(files/'provenance/hybrid-historical-bytecode-v2/receipt.json',files));resolve=resolver(files,freeze,capsule)
 root=files/'gap-hybrid-v1';need(sha(regular(root/'plan.json',files))==PLAN_SHA,'Exact new operational plan required');plan=load(root/'plan.json')
 need(plan['output']==REMOTE and plan['allocation']['uid']==UID,'Plan target differs')
 request_path=resolve('/work/gap/hybrid/hybrid-outer-request-fresh-v1.json');need(sha(request_path)==REQUEST_SHA,'Outer source inventory differs')
 request=load(request_path);need(len(request['files'])==365,'Full365 inventory required')
 expected={**request['files'],**plan['files']}
 for name,digest in request['files'].items():need(expected[name]==digest,'Conflicting immutable source binding')
 for row in freeze['files']:
  name,digest=row['source'],row['sha256']
  need(name not in expected or expected[name]==digest,'Conflicting frozen source alias: '+name)
  expected[name]=digest
 need({n:d for n,d in request['files'].items()if n.endswith('.pyc')}=={n:r['sha256']for n,r in capsule['files'].items()},'Exact11 historical inventory differs')
 for name,digest in expected.items():need(sha(resolve(name))==digest,'Pinned input differs: '+name)
 # All current closure and provenance files outside the output are available
 # without parsing task scores or opening any remote connection.
 selected={n:resolve(n) for n in expected if not relative(n).is_relative_to(REMOTE)}
 for p in files.rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(files)
  if rel.is_relative_to('gap-hybrid-v1'):continue
  name='/work/results/'+rel.as_posix();selected[name]=regular(p,files)
 selected['/work/gap/hybrid/hybrid-outer-request-fresh-v1.json']=request_path
 output=Path(output).resolve();need(not output.exists() and not output.is_relative_to(files),'Fresh local binding destination required');output.mkdir(parents=True)
 objects=output/'objects';objects.mkdir();bindings={};catalog={}
 for name,path in sorted(selected.items()):
  before=path.stat();digest=sha(path);after=path.stat()
  need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'Source changed during binding copy')
  target=objects/digest
  if not target.exists():
   with path.open('rb')as src,target.open('xb')as dst:shutil.copyfileobj(src,dst,4<<20)
  need(sha(target)==digest and sha(path)==digest,'Binding object/source drift')
  bindings[name]='objects/'+digest;catalog[name]={'sha256':digest,'bytes':before.st_size,'archive_source':str(path.relative_to(files))}
 bind=output/'bindings.json';bind.write_text(json.dumps(bindings,indent=2,sort_keys=True)+'\n')
 receipt={'schema':'fresh-hybrid-archive-only-bindings-v1','complete':True,'source_marker':str(marker_path),'source_marker_sha256':sha(marker_path),
  'archive_sha256':m['archive_sha256'],'raw_manifest_sha256':m['raw_manifest_sha256'],'pod_uid':UID,'plan_sha256':PLAN_SHA,
  'request_sha256':REQUEST_SHA,'bindings_sha256':sha(bind),'builder_sha256':sha(__file__),'catalog':catalog,'created_unix':time.time(),
  'scope':'Opaque archive-only source/closure binding inventory; no performance/accuracy effects read and no release authority.'}
 (output/'build-receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n');return receipt
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--marker',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 v=build(a.marker,a.output);print(json.dumps({'complete':True,'bindings':len(v['catalog']),'bindings_sha256':v['bindings_sha256']}))
