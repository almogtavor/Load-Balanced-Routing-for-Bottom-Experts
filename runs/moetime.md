MoE layer time inside a running vLLM 0.31 engine on the DeepEP high-throughput all-to-all, timed with CUDA events around `MoERunner._forward_impl` - routing, dispatch, expert GEMM, combine, and reduced to a median per token-count bucket. Each arm is divided by the base arms of its own launch; `keep=8` is the zero-substitution control. Positive is faster. The floor is that model's own widest base-against-base spread, halved, over the launches in its row.

| model | stock ms / moe layer | tokens / forward | floor | keep=8 (control: no substitution) | keep=6 | keep=6 +load balance | keep=4 | keep=4 +load balance |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B  E=128  ep=4 | 8.08 | 16384 | +/-0.20% | **-0.6%** x7 | **-2.1%** x7 | **+2.0%** x5 | **-5.1%** x6 | **-1.5%** x4 |
| Qwen3.5-35B  E=256  ep=4 | 7.00 | 16384 | +/-0.22% | **-0.4%** x5 | **-1.8%** x4 | **+0.5%** x3 | **-2.0%** x4 | **-1.1%** x3 |
| Gemma-4-26B-A4B  E=128  ep=4 | 9.72 | 16384 | +/-0.14% | **-0.3%** x3 | **-1.9%** x4 | **+3.2%** x2 | **-5.4%** x3 | **-0.3%** x3 |
| Qwen3-30B-A3B  E=128  ep=4  (host B) | 7.56 | 16384 | +/-0.08% | **-0.6%** x4 | **-2.3%** x4 | **+1.9%** x4 | **-5.3%** x4 | **-1.9%** x4 |
| Qwen3-30B-A3B  E=128  ep=8  (host B) | 11.17 | 16384 | +/-0.52% | **-0.1%** x8 | **-2.7%** x7 | **+0.4%** x10 | **-6.4%** x6 | **-1.8%** x7 |
