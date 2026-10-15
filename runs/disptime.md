DeepEP dispatch time inside a running vLLM 0.31 engine on the DeepEP high-throughput all-to-all, timed with CUDA events around `DeepEPHTPrepareAndFinalize.prepare_async` and the wait on its completion event - the all-to-all the rule shrinks, and reduced to a median per token-count bucket. Each arm is divided by the base arms of its own launch; `keep=8` is the zero-substitution control. Positive is faster. The floor is that model's own widest base-against-base spread, halved, over the launches in its row.

| model | stock ms / deepep dispatch | tokens / forward | floor | keep=8 (control: no substitution) | keep=6 | keep=6 +load balance | keep=4 | keep=4 +load balance |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B  E=128  ep=4 | 1.96 | 16384 | +/-0.36% | **+0.0%** x4 | **+0.8%** x4 | **+0.8%** x4 | **+4.0%** x1 | - |
| Qwen3.5-35B  E=256  ep=4 | 1.93 | 16384 | +/-0.16% | **-0.0%** x2 | - | **+0.0%** x1 | - | **+2.7%** x1 |
| Gemma-4-26B-A4B  E=128  ep=4 | 2.13 | 16384 | +/-0.02% | **-0.2%** x1 | **+0.7%** x1 | - | **+3.9%** x1 | - |
| Qwen3-30B-A3B  E=128  ep=4  (host B) | 1.81 | 16384 | +/-0.02% | **+0.1%** x3 | **+1.0%** x3 | **+0.9%** x3 | **+4.2%** x2 | **+4.2%** x1 |
| Qwen3-30B-A3B  E=128  ep=8  (host B) | 2.23 | 16384 | +/-0.05% | **+0.0%** x3 | **+3.9%** x3 | **+4.0%** x5 | **+10.7%** x2 | **+10.4%** x2 |
