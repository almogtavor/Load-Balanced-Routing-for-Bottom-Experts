# LBaR: Load-Balance Aware Router for the MoE Tail

A vLLM 0.31 plugin that changes which experts a top-k MoE router picks under expert parallelism (EP).
Each token keeps its top `keep` experts (the head). Its remaining `k - keep` experts (the tail) are replaced, one pick at a time, by the highest-scoring experts after a load penalty. The penalty counts how many experts a tile of 16 consecutive tokens has already placed on each GPU. Every token still runs `k` experts, so expert compute is unchanged; what changes is how evenly the experts are spread over the GPUs, and the MoE layer ends when the busiest GPU ends.

Two tail settings:

* **any-rank** (`PLACEMENT_ANY_RANK=1`, the paper's main setting): the tail may come from any GPU.
* **placement-aware**: the tail only comes from GPUs the head already visits, so each token is sent to fewer GPUs and the all-to-all moves less data.

The **mass rule** sizes the head per token: the smallest head (at least `keep`) that holds a share `m` of the top-k gate weight.

The rule runs inside a copy of vLLM's fused top-k Triton kernel, replacing it rather than running after it, so it adds no kernel launch, no memory traffic and no collective.

## Install and serve

```bash
pip install --no-deps -e .   # registers the vllm.general_plugins entry point
PLACEMENT_KEEP=4 PLACEMENT_LOAD_PENALTY=8 PLACEMENT_ANY_RANK=1 \
  vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --data-parallel-size 4 --enable-expert-parallel \
  --all2all-backend deepep_high_throughput
```

With `PLACEMENT_KEEP` unset the plugin patches nothing, which is the default vLLM every result is compared against. vLLM 0.31 uses an all-to-all only with data parallelism, so expert parallelism must run as DP=4, TP=1 (see the header of `experiments/serve_run.py`). `docker/` builds the serving image and `helm/` deploys it.

| variable | meaning |
|---|---|
| `PLACEMENT_KEEP` | head size; unset turns the plugin off |
| `PLACEMENT_LOAD_PENALTY` | load penalty per 16-token tile (λ); 0 is no penalty |
| `PLACEMENT_ANY_RANK=1` | tail from any GPU instead of the GPUs the head visits |
| `PLACEMENT_MASS`, `PLACEMENT_MASS_NORM=1` | mass rule; share of the renormalized top-k weight |
| `PLACEMENT_PREFILL_ONLY=1` | decode tokens keep default routing |
| `PLACEMENT_TAU`, `PLACEMENT_RANGE` | sampled tail as in Expert-Sample (control) |
| `PLACEMENT_COACT`, `PLACEMENT_COACT_LOAD=1` | co-activation expert placement from `experiments/coact_place.py` |
| `PLACEMENT_EPLB=1` | lets Gemma-4 report its expert load to EPLB, see below |

The plugin passes the selection through unchanged, with one warning, wherever the rule does not apply: grouped top-k, score-correction bias, routing without renormalization, EP size 1. EPLB replication is supported only with the any-rank tail. The full list is `_unsupported` in `src/placement_router/plugin.py`.

**Gemma-4 and EPLB.** In vLLM 0.31, Gemma-4's custom router does not report expert load to EPLB, so EPLB never rebalances it. With `PLACEMENT_EPLB=1` the plugin wires Gemma-4's MoE layers into EPLB at runtime (`_gemma_eplb`); the EPLB baseline for Gemma-4 in the paper uses this fix.

## Results

End-to-end prefill on 4xH100 (one host, NVLink), EP=4 as DP=4, DeepEP high-throughput, any-rank tail, λ=8 (mass: `keep=1`). Change in total throughput against default vLLM, with the 95% interval over 10 paired launches (5 for `keep=6` and Qwen3-30B mass 0.7):

| model | EPLB | keep=6 | keep=4 | mass 0.7 | mass 0.6 |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | +7.8 [6.9, 8.8] | +5.8 [4.3, 7.2] | +11.0 [9.8, 12.1] | +8.9 [7.0, 10.9] | +9.9 [8.4, 11.4] |
| Qwen3.5-35B-A3B | -2.7 [-3.4, -1.9] | +3.9 [3.7, 4.1] | +4.5 [3.9, 5.1] | +3.8 [2.0, 5.7] | +3.8 [2.6, 5.0] |
| Gemma-4-26B-A4B | +4.5 [3.5, 5.5] | +7.0 [5.4, 8.7] | +9.9 [8.8, 11.1] | +9.1 [8.5, 9.7] | +9.9 [8.9, 11.0] |

Accuracy, wall-clock time and the comparison against EPLB are in the paper.

### In-engine timing

CUDA events around DeepEP dispatch and around the whole MoE layer (`MoERunner._forward_impl`), at 16,384 tokens per forward, each setting paired with default vLLM in the same launch. `keep=8` runs the kernel with no replacement and is the control for its own cost. `xN` is the number of launches; the floor is the row's widest default-against-default spread, halved. These runs use the placement-aware tail and the earlier per-token load penalty at λ=1 ("+load balance"), from commit 2d8f519, before the 16-token tile.

![In-engine DeepEP dispatch time](runs/disptime.png)

<!-- disptime -->
| model | default ms / deepep dispatch ↓ | tokens / forward | floor | keep=8 (control: no substitution) ↑ | keep=6 ↑ | keep=6 +load balance ↑ | keep=4 ↑ | keep=4 +load balance ↑ |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B  E=128  ep=4 | 1.95 | 16384 | +/-0.36% | **+0.1%** x5 | **+0.7%** x5 | **+0.9%** x5 | **+3.8%** x4 | **+3.9%** x3 |
| Qwen3.5-35B  E=256  ep=4 | 1.93 | 16384 | +/-0.16% | **-0.0%** x3 | **-0.3%** x2 | **-0.2%** x3 | **+2.7%** x2 | **+2.6%** x3 |
| Gemma-4-26B-A4B  E=128  ep=4 | 2.15 | 16384 | +/-0.16% | **-0.0%** x2 | **+0.6%** x3 | **+0.6%** x2 | **+3.8%** x2 | **+3.6%** x3 |
| Qwen3-30B-A3B  E=128  ep=4  (host B) | 1.81 | 16384 | +/-0.05% | **+0.0%** x4 | **+1.0%** x4 | **+0.9%** x4 | **+4.1%** x4 | **+4.2%** x4 |
| Qwen3-30B-A3B  E=128  ep=8  (host B) | 2.23 | 16384 | +/-0.11% | **+0.0%** x7 | **+4.1%** x7 | **+4.0%** x9 | **+10.4%** x6 | **+10.4%** x6 |
<!-- /disptime -->

![In-engine MoE layer time](runs/moetime.png)

<!-- moetime -->
| model | default ms / moe layer ↓ | tokens / forward | floor | keep=8 (control: no substitution) ↑ | keep=6 ↑ | keep=6 +load balance ↑ | keep=4 ↑ | keep=4 +load balance ↑ |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B  E=128  ep=4 | 8.08 | 16384 | +/-0.20% | **-0.6%** x7 | **-2.1%** x7 | **+2.0%** x5 | **-5.1%** x6 | **-1.5%** x4 |
| Qwen3.5-35B  E=256  ep=4 | 7.00 | 16384 | +/-0.22% | **-0.4%** x5 | **-1.8%** x4 | **+0.5%** x3 | **-2.0%** x4 | **-1.1%** x3 |
| Gemma-4-26B-A4B  E=128  ep=4 | 9.72 | 16384 | +/-0.14% | **-0.3%** x3 | **-1.9%** x4 | **+3.2%** x2 | **-5.4%** x3 | **-0.3%** x3 |
| Qwen3-30B-A3B  E=128  ep=4  (host B) | 7.56 | 16384 | +/-0.08% | **-0.6%** x4 | **-2.3%** x4 | **+1.9%** x4 | **-5.3%** x4 | **-1.9%** x4 |
| Qwen3-30B-A3B  E=128  ep=8  (host B) | 11.17 | 16384 | +/-0.52% | **-0.1%** x8 | **-2.7%** x7 | **+0.4%** x10 | **-6.4%** x6 | **-1.8%** x7 |
<!-- /moetime -->

## Reproducing

All runs use vLLM 0.31.0 (`vllm/vllm-openai:v0.31.0`) on one host with 4xH100. `experiments/run.sh` installs the plugin, runs the kernel self-check on the GPU, then starts one `vllm serve` per setting and sweeps all settings `REPS` times, writing to `runs/`. A setting name such as `keep1-mass0.6-penalty8-any-rank` encodes the plugin variables (`setting_env` in `experiments/serve_run.py`); `base` is default vLLM and `eplb` is default vLLM with EPLB.

Prefill throughput and wall-clock (Table 2, Table 3 and the stacking table):

```bash
MODEL=/models/qwen3-30b TAG=prefill-qwen30b DP=4 A2A=deepep_high_throughput GPU_UTIL=0.90 \
  N_REQ=64 PROMPT_LEN=4096 GEN_LEN=1 BATCH_TOK=16384 REPEATS=6 WARMUP=12 REPS=10 \
  EPLB_STEPS=40 EPLB_REDUNDANT=16 \
  SETTINGS=base,eplb,keep6-penalty8-any-rank,keep4-penalty8-any-rank,keep1-mass0.7-penalty8-any-rank,keep1-mass0.6-penalty8-any-rank,keep4-penalty8-any-rank-with-eplb,keep1-mass0.6-penalty8-any-rank-with-eplb \
  experiments/run.sh
python experiments/paired.py prefill-qwen30b        # against default vLLM
python experiments/paired.py prefill-qwen30b eplb   # against EPLB
```

Use `GPU_UTIL=0.75` for Qwen3.5-35B and Gemma-4.

Accuracy (Table 2: GSM8K over three runs, HellaSwag, MMLU, every token routed):

```bash
MODEL=/models/qwen3-30b TAG=accuracy-qwen30b DP=4 A2A=deepep_high_throughput GPU_UTIL=0.90 \
  SCRIPT=experiments/evalrun.py REPS=3 \
  SETTINGS=base,keep6-penalty8-any-rank,keep4-penalty8-any-rank,keep1-mass0.7-penalty8-any-rank,keep1-mass0.6-penalty8-any-rank \
  experiments/run.sh
python experiments/paired.py accuracy-qwen30b
```

For Gemma-4 add `EVAL_CHAT=1 EVAL_TASKS=gsm8k`; `EVAL_TASKS` selects lm-eval tasks.

Placement-aware tail accuracy and WikiText NLL (Table 5): the same command with `SCRIPT=experiments/evalrun.py,experiments/nllserve.py` and `SETTINGS=base,base2,keep6-penalty8,keep4-penalty8,keep6-sampled-from-top32,keep4-sampled-from-top32,keep1-mass0.6,keep1-mass0.7`.

In-engine timing (the tables above; check out 2d8f519 first for the per-token penalty):

```bash
MODEL=/models/qwen3-30b TAG=layer-qwen30b DP=4 A2A=deepep_high_throughput SETTINGS=base,base2,keep8,keep6,keep6-penalty1,keep4,keep4-penalty1,base3 \
  experiments/run.sh
for phase in moe disp comb; do PHASE=$phase PYTHONPATH=experiments python experiments/moeplot.py; done
python3 -I experiments/readme_tables.py
```

On Kubernetes, `experiments/k8s/launch.sh NAME=... TAG=... MODEL=... SETTINGS=...` runs the same job with the same variables; rename the model claims and the `hf-token` secret in `experiments/k8s/job.yaml` to your own.

## Code

| path | contents |
|---|---|
| `src/placement_router/kernel.py` | the rule: Triton kernel, torch reference, self-check (`python -m placement_router.kernel --cpu` on a laptop) |
| `src/placement_router/plugin.py` | the vLLM hook, the plugin variables, the Gemma-4 EPLB fix |
| `src/placement_router/vendor/fused_topk_router.py` | vLLM 0.31's fused top-k router with the rule inserted |
| `experiments/serve_run.py` | one setting under a real `vllm serve`; throughput and wall-clock |
| `experiments/evalrun.py`, `experiments/nllserve.py` | lm-eval accuracy and WikiText NLL against the served model |
| `experiments/paired.py` | paired changes and intervals from `runs/` |
| `experiments/moetime.py`, `moeplot.py`, `lbgain.py`, `imbalance.py` | in-engine timers, their paired tables, per-rank load statistics |
| `experiments/accuracy.py`, `coact_place.py` | offline paired NLL; co-activation placement |

## Related work

Replacing the lowest-ranked experts with ones that are cheaper to reach: arXiv 2412.00099, 2504.01337, 2505.13345. The tail of a fine-grained router's selection is its uncertain part: arXiv 2602.02443.
