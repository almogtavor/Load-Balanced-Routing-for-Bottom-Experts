DeepEP combine time inside a running vLLM 0.31 engine on the DeepEP high-throughput all-to-all, timed with CUDA events around `DeepEPHTPrepareAndFinalize._finalize` and the wait on its completion event, and reduced to a median per token-count bucket. Each setting is divided by the base settings of its own launch; `keep=8` is the zero-substitution control. Positive is faster, so in every percentage column higher is better (marked ↑ in the header; ↓ marks the one column where lower is better). The floor is that model's own widest base-against-base spread, halved, over the launches in its row.

| model | default ms / deepep combine ↓ | tokens / forward | floor | keep=8 (control: no substitution) ↑ | keep=6 ↑ | keep=6 +load balance ↑ | keep=4 ↑ | keep=4 +load balance ↑ |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B  E=128  ep=4 | 2.23 | 16384 | +/-0.46% | **+0.1%** x5 | **-7.8%** x5 | **+9.1%** x5 | **-21.9%** x4 | **-7.7%** x3 |
| Qwen3.5-35B  E=256  ep=4 | 2.17 | 16384 | +/-0.18% | **+0.1%** x3 | **-4.6%** x2 | **+4.9%** x3 | **-14.6%** x2 | **-8.7%** x3 |
| Gemma-4-26B-A4B  E=128  ep=4 | 3.04 | 16384 | +/-0.46% | **+0.4%** x2 | **-6.2%** x3 | **+27.2%** x2 | **-18.4%** x2 | **+11.6%** x3 |
| Qwen3-30B-A3B  E=128  ep=4  (host B) | 2.14 | 16384 | +/-0.44% | **-0.0%** x4 | **-9.8%** x4 | **+9.2%** x4 | **-23.8%** x4 | **-5.7%** x4 |
| Qwen3-30B-A3B  E=128  ep=8  (host B) | 5.18 | 16384 | +/-0.41% | **+0.5%** x7 | **-9.9%** x7 | **-1.8%** x9 | **-21.8%** x6 | **-13.6%** x6 |
