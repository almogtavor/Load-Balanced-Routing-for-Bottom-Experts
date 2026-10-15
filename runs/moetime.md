MoE layer time inside a running vLLM 0.31 engine on the DeepEP high-throughput all-to-all, timed with CUDA events around `MoERunner._forward_impl` - routing, dispatch, expert GEMM, combine, and reduced to a median per token-count bucket. Each arm is divided by the base arms of its own launch; `keep=8` is the zero-substitution control. Positive is faster. Launch-to-launch floor +/-0.52%.

| model | stock ms / moe layer | tokens / forward | keep=8 (control: no substitution) | keep=6 | keep=6 +load balance | keep=4 | keep=4 +load balance |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B  E=128  ep=4 | 8.08 | 16384 | **-0.7%** x5 | **-2.0%** x4 | **+1.9%** x2 | **-5.2%** x3 | **-1.6%** x1 |
| Qwen3.5-35B  E=256  ep=4 | 7.06 | 16384 | **-0.2%** x2 | **-1.7%** x2 | - | **-1.8%** x2 | - |
| Qwen3-30B-A3B  E=128  ep=8 | 11.17 | 16384 | **-0.1%** x4 | **-3.1%** x3 | **+0.6%** x5 | **-6.2%** x1 | **-1.6%** x2 |
