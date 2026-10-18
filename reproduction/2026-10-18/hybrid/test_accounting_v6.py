import copy,hashlib,json,sys,tempfile,threading,time,unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
from unittest.mock import patch
import native_accounting as a
import audit_native_metrics as raw
import hybrid_client as client
from common_v2 import write_new,read,file_sha

class Tests(unittest.TestCase):
 def text(self,rounds,role='decode'):
  n=64*rounds if role=='decode' else 0
  lines=['# TYPE '+a.BYTES+' histogram']+['# TYPE '+x+' counter' for x in a.ERRORS]
  for i in range(4):
   for name,value in [(a.BYTES+'_count',n),(a.BYTES+'_sum',n*a.block_transfer_bytes(1024)),*[(x,0)for x in a.ERRORS]]:
    lines.append(name+'{engine="'+str(i)+'",model_name="task-model"} '+str(value))
  return ('\n'.join(lines)+'\n').encode()
 def test_exact_zero_not_transfer_coverage_and_no_injection(self):
  p={'do_remote_prefill':True,'do_remote_decode':False,'remote_prefill_cached_tokens':0,'native_other_field':'preserved'}
  self.assertEqual(a.zero_cached_transfer(p,client.REMOTE_DECODE,p)['native_remote_prefill_cached_tokens'],0)
  for changed in [{**p,'remote_prefill_cached_tokens':True},{**p,'remote_prefill_cached_tokens':1023}]:
   with self.assertRaises(ValueError):a.zero_cached_transfer(changed,client.REMOTE_DECODE,changed)
  with self.assertRaises(ValueError):a.zero_cached_transfer(p,{**client.REMOTE_DECODE,'remote_prefill_cached_tokens':0},p)
  with self.assertRaises(ValueError):a.zero_cached_transfer(p,client.REMOTE_DECODE,{**p,'native_other_field':'changed'})
 def test_per_engine_count_bytes_not_merely_total(self):
  before=a.metrics(self.text(0).decode());after=a.metrics(self.text(1).decode());budget=a.cumulative_budget(1,'decode')
  value=a.delta(before,after,expected_labels=a.ENGINE_LABELS,expected_requests=256,expected_bytes=24*2**30,expected_by_engine=budget)
  self.assertEqual(value['native_transfer_bytes'],24*2**30)
  keys=list(after[a.BYTES+'_sum']);after[a.BYTES+'_sum'][keys[0]]+=1;after[a.BYTES+'_sum'][keys[1]]-=1
  with self.assertRaises(ValueError):a.delta(before,after,expected_labels=a.ENGINE_LABELS,expected_requests=256,expected_bytes=24*2**30,expected_by_engine=budget)
 def test_parser_types_duplicate_nonfinite_reset_and_missing_labels_rejected(self):
  source=self.text(1).decode()
  for s in [source.rstrip(),source.replace(' histogram',' gauge'),source.replace(' 64\n',' nan\n',1),source+source.splitlines()[-1]+'\n']:
   with self.assertRaises(ValueError):a.metrics(s)
  with self.assertRaises(ValueError):a.covered(a.metrics(source),a.cumulative_budget(0,'decode'))
  current=a.metrics(source);prev=a.metrics(self.text(2).decode())
  with self.assertRaises(ValueError):a.covered(current,a.cumulative_budget(2,'decode'),prev)
 def test_reference_difference_retains_text_tokens_and_first_divergence_without_threshold(self):
  p={'token_ids':[1,2,4],'text':'a b d','finish_reason':'length'};d={'token_ids':[1,2,3],'text':'a b c','finish_reason':'stop'}
  value=client.reference_difference('training',d,p)
  self.assertFalse(value['equal_text']);self.assertEqual(value['first_differing_generated_position'],2)
  self.assertEqual(value['handoff_choice'],d);self.assertEqual(value['monolithic_choice'],p)
 def server(self,kind):
  payload=self.text(0)
  class Handler(BaseHTTPRequestHandler):
   def log_message(self,*args):pass
   def do_GET(self):
    self.send_response(500 if kind=='error' else 200)
    self.send_header('Content-Type','text/plain; version=1.0.0; charset=utf-8')
    self.send_header('Content-Length',str(len(payload)+(17 if kind=='short' else 0)));self.end_headers()
    self.wfile.write(payload);self.wfile.flush();self.close_connection=True
  server=HTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  self.addCleanup(lambda:(server.shutdown(),server.server_close(),thread.join(timeout=2)))
  return f'http://127.0.0.1:{server.server_address[1]}'
 def test_real_content_length_success_full_raw_retained(self):
  with tempfile.TemporaryDirectory()as t:
   ref,value=a.capture(self.server('success'),Path(t)/'snapshot',time.time()+2)
   self.assertTrue(ref['transport']['response_complete']);self.assertEqual(Path(ref['path']).read_bytes(),self.text(0))
 def test_real_premature_eof_keeps_partial_bytes_and_false_complete(self):
  with tempfile.TemporaryDirectory()as t:
   with self.assertRaises(ValueError):a.capture(self.server('short'),Path(t)/'snapshot',time.time()+2)
   self.assertEqual((Path(t)/'snapshot.txt').read_bytes(),self.text(0));self.assertFalse(read(Path(t)/'snapshot.transport.json')['response_complete'])
 def test_http_error_body_bounded_retained_without_replay(self):
  with tempfile.TemporaryDirectory()as t:
   with self.assertRaises(ValueError):a.capture(self.server('error'),Path(t)/'snapshot',time.time()+2)
   meta=read(Path(t)/'snapshot.transport.json');self.assertEqual(meta['http_status'],500);self.assertEqual((Path(t)/'snapshot.txt').read_bytes(),self.text(0))
 def synthetic_capture(self,url,stem,stop_unix):
  # Synthetic counters exercise exact all-role HTTP metadata/poll raw replay;
  # this fixture does not claim CUDA/native transfer execution.
  role='prefill' if '8100' in url else 'decode';after='after-'in Path(stem).name
  rounds=1 if after and (not self.lag or 'poll00'not in Path(stem).name)else 0
  payload=self.text(rounds,role);path=Path(stem).with_suffix('.txt');path.write_bytes(payload)
  now=time.time_ns();meta={'url':url+'/metrics','started_unix_ns':now,'finished_unix_ns':now,'http_status':200,'response_complete':True,
   'headers':[['content-type','text/plain; version=1.0.0; charset=utf-8'],['content-length',str(len(payload))]],'error':None,'response_bytes':len(payload),'response_sha256':hashlib.sha256(payload).hexdigest()}
  tp=Path(stem).with_suffix('.transport.json');write_new(tp,meta)
  return {'path':str(path),'sha256':meta['response_sha256'],'transport_path':str(tp),'transport_sha256':file_sha(tp),'transport':meta},a.metrics(payload.decode())
 def fixture(self,t,lag=False):
  self.lag=lag;directory=Path(t)/'warmup00';folder=directory.with_name('warmup00-native-metrics')
  plan={'prefill_url':'http://127.0.0.1:8100','decode_url':'http://127.0.0.1:8200','stop_unix':time.time()+30}
  with patch.object(a,'capture',side_effect=self.synthetic_capture):
   state=a.begin_round(plan,'warmup',0,folder);directory.mkdir();start=time.time_ns();finish=time.time_ns()
   app={'complete':True,'phase':'warmup','round':0,'cohort':{'start_unix_ns':start,'start_ns':1},'finished_unix_ns':finish,'finished_ns':2,'records':[]}
   write_new(directory/'application-round.json',app);ref=a.finish_round(plan,state,finish)
  summary={**app,'application_round_sha256':file_sha(directory/'application-round.json'),'native_metrics':ref,'cached_token_scope':'Native zero producer hits'}
  write_new(directory/'round.json',summary)
  class Bundle:
   def checked_file(self,name,expected=None):
    p=Path(name)
    if expected is not None:raw.need(file_sha(p)==expected,'hash changed')
    return p
   def json(self,name,expected=None):return read(self.checked_file(name,expected))
  return Bundle(),str(directory/'round.json'),summary,folder
 def test_full_raw_offclock_fixture_and_lag_all_polls_preserved(self):
  with tempfile.TemporaryDirectory()as t:
   b,path,summary,folder=self.fixture(t,True);reports=raw.audit(b,path,summary,'warmup',0)
   self.assertEqual(reports['decode']['native_transfer_count'],256);self.assertEqual(len(read(folder/'receipt.json')['after_polls']),2)
 def test_raw_changed_counter_gap_clock_due_or_unknown_file_rejected(self):
  for attack in ['counter','due','clock','extra','application']:
   with self.subTest(attack=attack),tempfile.TemporaryDirectory()as t:
    b,path,s,folder=self.fixture(t);receipt=read(folder/'receipt.json')
    if attack=='extra':(folder/'unknown.txt').write_text('extra')
    elif attack=='application':s['finished_ns']=33
    elif attack=='counter':receipt['native_coverage']['decode']['native_transfer_count']=255
    elif attack=='due':receipt['after_polls'][0]['scheduled_unix']+=1
    else:receipt['completion_finished_unix_ns']-=1
    if attack in ('counter','due','clock'):
     (folder/'receipt.json').write_text(json.dumps(receipt));s['native_metrics']['sha256']=file_sha(folder/'receipt.json')
    with self.assertRaises(ValueError):raw.audit(b,path,s,'warmup',0)

if __name__=='__main__':unittest.main()
