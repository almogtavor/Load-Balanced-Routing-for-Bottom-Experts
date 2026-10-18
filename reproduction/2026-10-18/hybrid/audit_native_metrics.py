"""Independent closed raw native-metrics replay; no HTTP/GPU calls."""
from pathlib import PurePosixPath
import math
import native_accounting as accounting

def need(value,message):
    if not value:raise ValueError(message)

def audit(bundle,remote,summary,phase,number):
    directory=str(PurePosixPath(remote).parent)
    folder=directory+'-native-metrics';ref=summary['native_metrics']
    need(ref['complete'] is True and ref['path']==folder+'/receipt.json','Native metrics receipt path/completeness differs')
    receipt=bundle.json(ref['path'],ref['sha256'])
    application=bundle.json(directory+'/application-round.json',summary['application_round_sha256'])
    need(application=={k:v for k,v in summary.items() if k not in ('application_round_sha256','native_metrics','cached_token_scope')},
         'Application clock/records differ from final off-clock accounting round')
    completed=number+(3 if phase=='measured' else 0)
    need(receipt['complete'] is True and receipt['completed_rounds_before']==completed
         and receipt['completion_finished_unix_ns']==summary['finished_unix_ns']
         and summary['finished_unix_ns']>=summary['cohort']['start_unix_ns'],
         'Native metrics round/completion clock identity differs')
    started=receipt['drain_started_unix'];deadline=receipt['drain_deadline_unix']
    need(type(started) in (int,float) and type(deadline) in (int,float)
         and math.isfinite(started) and math.isfinite(deadline) and 0<deadline-started<=5.000001
         and started*1e9>=summary['finished_unix_ns'],'Bounded off-clock drain window differs')
    need(receipt['finished_unix_ns']>=started*1e9,'Native drain completion timestamp differs')
    files=receipt['files_sha256'];path=bundle.checked_file(ref['path'],ref['sha256']).parent
    entries=list(path.iterdir())
    need(all(p.is_file() and not p.is_symlink() for p in entries)
         and {p.name for p in entries}==set(files)|{'receipt.json'},'Native metrics flat complete inventory differs')
    for name,digest in files.items():
        need(PurePosixPath(name).name==name and name not in ('.','..'),'Unsafe native metrics filename')
        bundle.checked_file(folder+'/'+name,digest)
    expected=set();before={};latest_http_finished=0
    def snapshot(row,stem):
        nonlocal latest_http_finished
        raw=folder+'/'+stem+'.txt';transport=folder+'/'+stem+'.transport.json'
        need(row['path']==raw and row['transport_path']==transport,'Native snapshot canonical path differs')
        payload=bundle.checked_file(raw,row['sha256']).read_bytes();meta=bundle.json(transport,row['transport_sha256'])
        need(meta==row['transport'] and meta['response_complete'] is True and meta['http_status']==200
             and meta['error'] is None and meta['response_bytes']==len(payload) and meta['response_sha256']==row['sha256']
             and latest_http_finished<=meta['started_unix_ns']<=meta['finished_unix_ns'],'Native metrics raw HTTP incomplete/changed/overlapping')
        latest_http_finished=meta['finished_unix_ns']
        role='prefill' if stem.endswith('prefill') else 'decode'
        need(meta['url']==('http://127.0.0.1:8100' if role=='prefill' else 'http://127.0.0.1:8200')+'/metrics',
             'Native metrics role/URL differs')
        header={}
        for name,value in meta['headers']:header.setdefault(name.lower(),[]).append(value)
        need(header.get('content-type')==['text/plain; version=1.0.0; charset=utf-8']
             and header.get('content-length')==[str(len(payload))] and 'transfer-encoding' not in header,
             'Native metrics source/framing headers differ')
        expected.update((stem+'.txt',stem+'.transport.json'))
        return accounting.metrics(payload.decode('utf-8')),meta
    need(set(receipt['before'])=={'prefill','decode'},'Native baseline role inventory differs')
    for role in ('prefill','decode'):
        before[role],meta=snapshot(receipt['before'][role],'before-'+role)
        need(meta['finished_unix_ns']<=summary['cohort']['start_unix_ns']
             and accounting.covered(before[role],accounting.cumulative_budget(completed,role)),
             'Native baseline overlaps clock or differs from prior drained cumulative budget')
    polls=receipt['after_polls'];need(1<=len(polls)<=10,'Finite native drain poll inventory differs')
    previous=before;reports={}
    for index,poll in enumerate(polls):
        need(poll['index']==index and poll['scheduled_unix']==started+index*.5
             and set(poll['roles'])=={'prefill','decode'},'Native deterministic poll schedule/role order differs')
        need(bundle.json(folder+f'/poll{index:02d}.json')==poll,'Native retained nominal poll differs');expected.add(f'poll{index:02d}.json')
        values={};complete=True
        for role in ('prefill','decode'):
            values[role],meta=snapshot(poll['roles'][role],f'after-poll{index:02d}-'+role)
            need(max(summary['finished_unix_ns'],poll['scheduled_unix']*1e9)<=meta['started_unix_ns']
                 <=meta['finished_unix_ns']<=deadline*1e9,'Native poll overlaps completion clock or exceeds due/deadline')
            complete=accounting.covered(values[role],accounting.cumulative_budget(completed+1,role),previous[role]) and complete
        need(poll['coverage_complete'] is complete and complete is (index==len(polls)-1),
             'Native polling continued after target or stopped before exact coverage')
        previous=values
    need(receipt['finished_unix_ns']>=latest_http_finished,'Native receipt precedes final selected GET')
    for role in ('prefill','decode'):
        reports[role]=accounting.delta(before[role],previous[role],expected_labels=accounting.ENGINE_LABELS,
          expected_requests=256 if role=='decode' else 0,
          expected_bytes=256*accounting.block_transfer_bytes(1024) if role=='decode' else 0,
          expected_by_engine=accounting.cumulative_budget(1,role))
    need(set(files)==expected and receipt['native_coverage']==ref['native_coverage']==reports,
         'Native per-engine count/byte coverage or poll artifact inventory differs')
    return reports
