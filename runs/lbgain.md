% faster with the load term than the plain rule at the same keep, paired within launch: mean +/- s.e., x launches. Bold: faster at p < 0.05, two-sided t-test over launches.

| model | dispatch keep=6 ↑ | dispatch keep=4 ↑ | combine keep=6 ↑ | combine keep=4 ↑ | MoE layer keep=6 ↑ | MoE layer keep=4 ↑ |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B  E=128  ep=4 | +0.1 +/-0.1 x5 | **+0.2 +/-0.0 x3** | **+18.4 +/-0.0 x5** | **+18.4 +/-0.4 x3** | **+4.1 +/-0.1 x5** | **+3.8 +/-0.2 x3** |
| Qwen3.5-35B  E=256  ep=4 | -0.1 +/-0.0 x2 | -0.2 +/-0.0 x2 | **+9.6 +/-0.2 x2** | +6.4 +/-0.6 x2 | **+2.1 +/-0.1 x2** | +0.9 +/-0.1 x2 |
| Gemma-4-26B-A4B  E=128  ep=4 | -0.0 +/-0.1 x2 | -0.1 x1 | **+35.7 +/-0.1 x2** | +37.5 x1 | **+5.2 +/-0.1 x2** | +5.4 x1 |
| Qwen3-30B-A3B  E=128  ep=4  (host B) | -0.1 +/-0.1 x4 | +0.1 +/-0.0 x4 | **+21.1 +/-0.2 x4** | **+23.8 +/-0.8 x4** | **+4.3 +/-0.0 x4** | **+3.7 +/-0.1 x4** |
| Qwen3-30B-A3B  E=128  ep=8  (host B) | -0.2 +/-0.1 x7 | -0.1 +/-0.2 x6 | **+9.0 +/-0.9 x7** | **+10.4 +/-1.0 x6** | **+3.0 +/-0.5 x7** | **+4.8 +/-0.3 x6** |
