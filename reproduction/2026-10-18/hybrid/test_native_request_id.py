"""CPU native-source identity, exact mapping and real HTTP regression fixtures."""
import ast,copy,hashlib,json,tempfile,threading,time,unittest
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import native_request_id as ids
import hybrid_client as client
import hybrid_driver as driver
import hybrid_timing as timing
from common_v2 import base_body,encoded

HERE=Path(__file__).resolve().parent
CAPTURE=(HERE.parents[3]/'runs/paper-followup-20261018')/'audit/hybrid-native-request-id-source-v1'
ASYNC=(HERE.parents[3]/'runs/paper-followup-20261018')/'audit/profile-aux-native-source-v1/async_llm.py'

class NativeTests(unittest.TestCase):
 def tree(self,root):
  for name,digest in ids.SOURCES.items():
   source=ASYNC if name=='v1/engine/async_llm.py' else CAPTURE/name
   raw=source.read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),digest)
   p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
  return ids.installed_sources(root)
 def actual(self,root,disabled=False):
  sources=self.tree(root);env=SimpleNamespace(VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=disabled)
  uuid_node=next(n for n in ast.parse((root/'utils/__init__.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='random_uuid')
  g={'__name__':'vllm.utils','uuid':SimpleNamespace(uuid4=lambda:SimpleNamespace(int=0xa8477bf800000000)),'MASK_64_BITS':(1<<64)-1}
  exec(compile(ast.Module(body=[uuid_node],type_ignores=[]),str(root/'utils/__init__.py'),'exec'),g)
  cls=next(n for n in ast.parse((root/'v1/engine/input_processor.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='InputProcessor')
  method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='assign_request_id')
  cls.body=[method];cls.bases=[];cls.keywords=[];cls.decorator_list=[]
  namespace={'__name__':'vllm.v1.engine.input_processor','EngineCoreRequest':object,'envs':env,'logger':SimpleNamespace(warning_once=lambda *a:None),'random_uuid':g['random_uuid']}
  exec(compile(ast.Module(body=[cls],type_ignores=[]),str(root/'v1/engine/input_processor.py'),'exec'),namespace)
  return namespace['InputProcessor'],env,sources
 def test_actual_all_six_source_bytes_and_disabled_default(self):
  with tempfile.TemporaryDirectory() as t:self.tree(Path(t))
  env=(CAPTURE/'envs.py').read_text()
  self.assertIn('int(os.getenv("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "0"))',env)
 def test_actual_native_staticmethod_and_uuid_body_unchanged_by_observer(self):
  with tempfile.TemporaryDirectory() as t:
   p,e,s=self.actual(Path(t));events=[]
   ids.install_assignment_hook(p,e,s,lambda kind,**v:events.append({'kind':kind,'pid':99,'unix_ns':time.time_ns(),**v}))
   request=SimpleNamespace(request_id='cmpl-pilot-0',external_req_id=None,prompt_token_ids=[11,22,33])
   before=time.time_ns();self.assertIsNone(p.assign_request_id(request));after=time.time_ns()
   self.assertEqual(request.request_id,'cmpl-pilot-0-a8477bf8');self.assertEqual(request.external_req_id,'cmpl-pilot-0')
   index=ids.assignment_index(events,s)
   self.assertEqual(ids.resolve_assignment(index,'cmpl-pilot',[11,22,33],{'submitted_unix_ns':before,'returned_unix_ns':after}),request.request_id)
 def test_native_randomization_disabled_rejected_without_installing_wrapper(self):
  with tempfile.TemporaryDirectory() as t:
   p,e,s=self.actual(Path(t),True);original=p.assign_request_id
   with self.assertRaisesRegex(RuntimeError,'remain enabled'):ids.install_assignment_hook(p,e,s,lambda *a,**k:None)
   self.assertIs(p.assign_request_id,original)
 def test_changed_actual_native_factory_source_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   p,e,s=self.actual(Path(t));p.assign_request_id.__globals__['random_uuid']=lambda:'a8477bf8'
   with self.assertRaisesRegex(RuntimeError,'factory differs'):ids.install_assignment_hook(p,e,s,lambda *a,**k:None)
 def test_actual_captured_full_producer_id_is_preserved(self):
  capture=json.loads(((HERE.parents[3]/'runs/paper-followup-20261018')/'audit/hybrid-v3-client-failure.json').read_bytes())
  response=capture['client/raw/000000-prefill.response.json']['text']
  native=response['kv_transfer_params']['remote_request_id'];self.assertEqual(ids.producer_internal_id(response),native)
  self.assertNotEqual(native,response['id']+'-0')
 def test_prefix_only_unsuffixed_wrong_suffix_and_wrong_parent_rejected(self):
  for value in ('cmpl-job-0','cmpl-job-0-a8477bf','cmpl-job-0-A8477BF8','cmpl-other-0-a8477bf8','cmpl-job-0-a8477bf8-extra'):
   with self.subTest(value=value),self.assertRaises(RuntimeError):ids.require_internal_id('cmpl-job-0',value)
 def fixture(self):
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);p,e,s=self.actual(Path(t.name));events=[]
  ids.install_assignment_hook(p,e,s,lambda kind,**v:events.append({'kind':kind,'pid':99,'unix_ns':time.time_ns(),**v}))
  p.assign_request_id(SimpleNamespace(request_id='cmpl-job-0',external_req_id=None,prompt_token_ids=[11,22]))
  return events,s
 def test_duplicate_source_or_factory_identity_tampering_rejected(self):
  events,s=self.fixture()
  for attack in ('duplicate','source','factory','time','prompt'):
   changed=copy.deepcopy(events)
   if attack=='duplicate':changed.append(changed[0])
   elif attack=='source':changed[0]['request_id_sources']['sha256']['envs.py']='other'
   elif attack=='factory':changed[0]['native_assign_callable']['random_uuid_callable']['qualname']='other'
   elif attack=='time':changed[0]['assignment_finished_unix_ns']=0
   else:changed[0]['prompt_sha256']='other'
   with self.subTest(attack=attack),self.assertRaises(RuntimeError):
    index=ids.assignment_index(changed,s);ids.resolve_assignment(index,'cmpl-job',[11,22],{'submitted_unix_ns':1,'returned_unix_ns':time.time_ns()+1})
 def test_wrong_http_interval_and_missing_exact_mapping_rejected(self):
  events,s=self.fixture();index=ids.assignment_index(events,s)
  for response,start,end in [('cmpl-other',1,time.time_ns()+1),('cmpl-job',time.time_ns()+1,time.time_ns()+10),('cmpl-job',1,2)]:
   with self.subTest(response=response,start=start),self.assertRaises(RuntimeError):ids.resolve_assignment(index,response,[11,22],{'submitted_unix_ns':start,'returned_unix_ns':end})
 def test_invalid_prompt_ids_not_hashed_as_valid_native_tokens(self):
  for prompt in ([],[True],[1.5],[-1]):
   with self.subTest(prompt=prompt),self.assertRaises(RuntimeError):ids.prompt_identity(prompt)
 def test_launch_scrubs_inherited_request_randomization_disable_without_override(self):
  plan={"site_root":"/pristine","phase_probe":"/probe"}
  with patch.dict("os.environ",{"VLLM_DISABLE_REQUEST_ID_RANDOMIZATION":"1"}):
   for role in ("prefill","decode"):
    for probe in (None,Path("/evidence")):
     env=driver.environment(plan,role,"stock",probe)
     self.assertNotIn("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION",env)
 def test_real_content_length_producer_and_sse_consumer_full_id_roundtrip(self):
  prompt=[33]*1024;identity='cmpl-cpu-stream';params={'do_remote_prefill':True,'do_remote_decode':False,'remote_engine_id':'p','remote_request_id':identity+'-0-a8477bf8','remote_host':'127.0.0.1','remote_port':5600,'remote_block_ids':[[1,2]],'remote_num_tokens':1024,'tp_size':1,'pp_size':1,'transfer_mode':'pull','remote_prefill_cached_tokens':0}
  producer=encoded({'id':identity,'choices':[{'index':0,'prompt_token_ids':prompt,'token_ids':[99],'text':'x','finish_reason':'length'}],'kv_transfer_params':params,'usage':{'prompt_tokens':1024,'completion_tokens':1,'total_tokens':1025}})
  frames=[]
  for i in range(128):
   choice={'index':0,'token_ids':[55],'text':'x','finish_reason':'length' if i==127 else None}
   if i==0:choice['prompt_token_ids']=prompt
   frames.append({'id':identity,'choices':[choice]})
  frames.append({'id':identity,'choices':[],'usage':{'prompt_tokens':1024,'completion_tokens':128,'total_tokens':1152,'prompt_tokens_details':{'cached_tokens':0}}})
  consumer=b''.join(b'data: '+encoded(f)+b'\n\n' for f in frames)+b'data: [DONE]\n\n';calls=[]
  class Handler(BaseHTTPRequestHandler):
   def log_message(self,*a):pass
   def do_POST(self):
    calls.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))));p=self.path.startswith('/p/');raw=producer if p else consumer
    self.send_response(200);self.send_header('Content-Type','application/json' if p else 'text/event-stream');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
  server=HTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  try:
   parser=driver.lifecycle((HERE.parents[3]/'runs/paper-followup-20261018')/'performance/prefill_e2e.py').parse_event
   row={'request_id':0,'token_ids':prompt};body=base_body('task-model',prompt,'cpu-stream',128);body['ignore_eos']=True;pre,_=client.paired_body(body,None)
   url=f'http://127.0.0.1:{server.server_address[1]}';plan={'prefill_url':url+'/p','decode_url':url+'/d','request_timeout':5,'stop_unix':time.time()+60,'output_tokens':128}
   result,raw_p,raw_d,wire=timing.stream_handoff(plan,row,body,encoded(pre),parser,threading.Barrier(1),{'start_ns':time.perf_counter_ns()})
   self.assertTrue(result['complete'],result.get('error'));self.assertEqual(result['transfer_params']['remote_request_id'],params['remote_request_id']);self.assertEqual(calls[1]['kv_transfer_params'],params);self.assertEqual(raw_p,producer);self.assertEqual(raw_d,consumer)
  finally:server.shutdown();server.server_close();thread.join(timeout=2)

if __name__=='__main__':unittest.main()
