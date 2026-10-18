import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import prospective_gate as p
import native_accounting as a
import hybrid_driver as driver
from common_v2 import read,file_sha

HERE=Path(__file__).resolve().parent;STUDY=HERE.parents[1]
class Tests(unittest.TestCase):
 def mapping(self):
  old=STUDY/'remote-capture/h200-v5-closed/archive-captures/20261018T084854Z-439b023329f9/extracted-20261018T084904Z-4f51b4efb804/files'
  new=STUDY/'remote-capture/h200-stock-numerics-closed/archive-captures/20261018T092007Z-e626bcb30bf7/extracted-20261018T092042Z-3c94d4fa5f89/files'
  return {name:(STUDY/'audit/stock-kv-numerics-raw-analysis-v1.json' if name==p.ANALYSIS else
   (new if '/hybrid-stock-numerics-v1/'in name else old)/name.removeprefix('/work/results/'))for name in p.FILES}
 def test_actual_reviewed_closed_metadata_and_preserved_failed_v5_admission(self):
  mapping=self.mapping()
  for n,d in p.FILES.items():self.assertEqual(file_sha(mapping[n]),d)
  with patch.object(p,'Path',side_effect=lambda n:mapping[str(n)]),patch.object(p,'read',side_effect=lambda n:read(mapping[str(n)])):
   result=p.verify(SimpleNamespace(process_record=lambda pid:None))
   self.assertEqual(result['files_sha256'],p.FILES);self.assertFalse(read(mapping['/work/results/hybrid-v5/proof/receipt.json'])['complete'])
   original=read(mapping['/work/results/hybrid-stock-numerics-v1/diagnostic/receipt.json'])['controller_process']
   with self.assertRaises(RuntimeError):p.verify(SimpleNamespace(process_record=lambda pid:{**original,'state':'S'} if pid==original['pid'] else None))
 def test_actual_source_definition_bytes_and_changed_source_rejection(self):
  root=HERE.parent/'native-cached-nixl-source-v1'
  files={'v1/core/sched/scheduler.py':root/'v1/core/sched/scheduler.py',
   'distributed/kv_transfer/kv_connector/v1/nixl/stats.py':root/'distributed/kv_transfer/kv_connector/v1/nixl/stats.py',
   'v1/engine/output_processor.py':STUDY/'audit/hybrid-native-request-id-source-v1/v1/engine/output_processor.py',
   'v1/engine/core.py':STUDY/'audit/h200-native-engine-core-source-v1/core.py'}
  self.assertEqual(a.source_contract(files),a.SOURCES)
  with tempfile.TemporaryDirectory()as t:
   bad=Path(t)/'scheduler.py';bad.write_bytes(files['v1/core/sched/scheduler.py'].read_bytes()+b'\n')
   with self.assertRaises(ValueError):a.source_contract({**files,'v1/core/sched/scheduler.py':bad})
 def test_same_scientific_server_commands_and_all_new_runtime_assets_pinned(self):
  # AST comparison is independent of imports/model/runtime availability.
  import ast
  old=ast.parse((HERE.with_name('hybrid-v5')/'hybrid_driver.py').read_text());new=ast.parse((HERE/'hybrid_driver.py').read_text())
  for name in ('commands','environment'):
   f=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
   self.assertEqual(ast.dump(f(old),include_attributes=False),ast.dump(f(new),include_attributes=False))
  function=next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name=='make_plan')
  strings={n.value for n in ast.walk(function)if isinstance(n,ast.Constant)and isinstance(n.value,str)}
  self.assertTrue({'native_accounting.py','prospective_gate.py','audit_native_metrics.py','audit_hybrid_timing_raw_v6.py'}<=strings)

if __name__=='__main__':unittest.main()
