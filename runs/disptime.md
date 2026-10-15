DeepEP dispatch time inside a running vLLM 0.31 engine on the DeepEP high-throughput all-to-all, timed with CUDA events around `DeepEPHTPrepareAndFinalize.prepare_async` and the wait on its completion event - the all-to-all the rule shrinks, and reduced to a median per token-count bucket. Each arm is divided by the base arms of its own launch; `keep=8` is the zero-substitution control. Positive is faster. Launch-to-launch floor +/-0.06%.

| model | stock ms / deepep dispatch | tokens / forward | keep=8 (control: no substitution) | keep=6 | keep=6 +load balance | keep=4 | keep=4 +load balance |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B  E=128  ep=4 | 1.96 | 16384 | **+0.1%** x2 | **+0.9%** x2 | **+0.7%** x1 | **+4.0%** x1 | - |
| Qwen3-30B-A3B  E=128  ep=8 | 2.23 | 16384 | **+0.0%** x3 | **+3.9%** x3 | **+4.0%** x4 | **+10.5%** x1 | **+10.4%** x1 |
