# Placement-Aware MoE Router

A vLLM plugin that re-routes the bottom of each token's expert selection onto the GPUs the
token is already being sent to, and the experiments that measure what that costs and buys on
four MoE families.

A token routed to top-8 of a few hundred experts in an expert-parallel deployment usually has
to be sent to *every* GPU, because eight experts drawn from that many rarely share a device.
The all-to-all dispatch costs one copy of the token per destination GPU. Published ablations
find the selection is not uniformly important - the top few experts carry the result and the
bottom ones can be substituted with little accuracy loss - so the bottom slots are free
parameters, and there is an obvious thing to spend them on:

> Keep the top `keep` experts by router score. The *allowed set* is every expert living on a
> GPU that those `keep` already require. Fill the remaining `top_k - keep` slots from the
> allowed set, by score. Per-token GPU fan-out drops to `|GPUs of the top keep|`; the number of
> experts per token, and so the expert GEMM work, is unchanged.

The saving is purely communication, which is why it has to be measured rather than counted.

![the result](runs/vllm.png)

## Using it

`PAR_KEEP` is the only knob. Unset, the plugin patches nothing and the image is stock vLLM -
which is also the arm every claim here is compared against.

```bash
docker build -t par-vllm:v0.31.0 -f docker/Dockerfile .
docker run --gpus all -e PAR_KEEP=2 par-vllm:v0.31.0 \
  --model Qwen/Qwen3-30B-A3B --tensor-parallel-size 4 --enable-expert-parallel
```

```bash
helm install par helm/par --set keep=2 --set gpus=4 --set model=Qwen/Qwen3-30B-A3B
```

Or into an existing environment, against any vLLM 0.31 install:

```bash
pip install --no-deps -e .   # registers the `vllm.general_plugins` entry point
PAR_KEEP=2 vllm serve <model> --tensor-parallel-size 4 --enable-expert-parallel
```

## How it hooks in

vLLM v0.31 routes every fused-MoE layer through `BaseRouter._select_experts`, which validates
EPLB state, delegates to a subclass `_compute_routing`, maps logical expert ids to physical and
fixes the index dtype. `src/par/plugin.py` wraps that one method: the model's own router
computes its own top-k, then the Triton kernel in `src/par/kernel.py` rewrites the bottom
`k - keep` ids and the gate weights are regathered at the new indices. The dispatch, the
combine, the expert GEMMs and every other routing variant are untouched, which is the only
reason this is safe to drop into a stock image.

Expert-to-rank needs no map tensor: vLLM shards experts in contiguous blocks, so rank is
`expert * ep_size // num_experts`, computed in the kernel.

The wrapper **passes the selection through unchanged, with one warning**, wherever the rule's
premise does not hold: EPLB replication (expert id no longer implies a rank), grouped top-k or
score-correction bias or zero-expert routing (the ranking is not the gate logits, so "the top
`keep` by logit" is not the head the router chose), a rank count of 1 (no all-to-all to
shrink), and a non-divisible or too-small expert block (no substitute is guaranteed to exist).
A routing rule that silently applied where its premise fails would be measured as free and
serve garbage.

## What it measures out at

**The throughput numbers are withdrawn and being re-measured.** The arms that produced them
ran a kernel that mishandled vLLM's padded rows, and the mishandling fell on the rule's side
of the comparison only, so the cost it reported is an upper bound on the real one and not an
estimate of it.

vLLM pads a batch for cudagraph and data-parallel capture and marks the padding by forcing
those tokens' expert ids to `-1`, so the dispatch and the experts drop them
(`VLLM_MOE_SKIP_PADDING`, on by default since 0.31). A `-1` is not a selection, and the first
kernel treated it as one: its `argmax` ran over an all-`-inf` block, fewer than `top_k` output
slots got written, and the row came out holding whatever `torch.empty` had left there. Real
expert ids, for tokens the unmodified engine never dispatches. The baseline arms skipped that
padding and the rule's arms paid full dispatch and expert GEMM for it - at decode batch sizes,
where padding is a large fraction of the captured batch, that is most of a measurement.

It also explains the one failure that was not a quiet bias: a `-1` reaching Gemma's index
scatter asserted out of bounds mid-run (`ScatterGatherKernel.cu:203`).

The kernel now declines any row that is not `top_k` distinct in-range experts and returns it
byte-for-byte, sentinel included; `src/par/kernel.py` asserts that on GPU before any arm runs.
The structural claim below is unaffected - it is a property of the dispatch, not of the rule's
per-layer cost - and so are the quality numbers: vLLM's fused MoE imposes no expert capacity,
so a padded token routed by mistake costs time and takes nothing from a real token's output.

The reason there is nothing on the other side of it is the dispatch. vLLM 0.31 defaults to
`allgather_reducescatter`, which gathers every token onto every rank: the volume it moves does
not depend on per-token GPU fan-out, so this rule has nothing to shrink and its extra work is
pure cost.

### How much there is to shrink, before anything is measured

Dispatch volume is per-token GPU fan-out, and for uniform logits that is analytic: a rank
holds `E/G` of the `E` experts, so a top-k drawn without replacement misses it with
probability `prod_i (E-E/G-i)/(E-i)`, and a token touches `G (1 - p(k))` ranks. The rewrite
makes a token's rank set its top-`keep`'s rank set, so the same expression at `n = keep` is
the fan-out afterwards. `expected_fanout` in `src/par/kernel.py` is that formula, asserted
against the measured fan-out in the self-check and printed by it:

| `ep_size` | top-8 touches | `keep=6` | `keep=4` | `keep=2` |
| --- | --- | --- | --- | --- |
| 2 | 1.99 / 2 | 1.1% | 5.7% | 24.6% |
| 4 | 3.63 / 4 | 8.6% | 24.1% | 51.6% |
| 8 | 5.34 / 8 | 16.3% | 37.4% | 64.8% |
| 16 | 6.60 / 16 | 20.8% | 44.2% | 70.5% |
| 32 | 7.36 / 32 | 23.2% | 47.6% | 73.1% |

Two things follow, and both are about what these runs can and cannot show. The headroom is
governed by the rank count: at `ep_size=4`, which is the width these experiments fit on,
`keep=4` removes a quarter of the dispatch volume, and dispatch is a fraction of a decode
step - so a few percent is the ceiling on the whole arm, and an arm that comes back flat here
has not tested the rule, only this width. And the headroom does not grow with how many tokens
are in flight: throughput is a rate, every decode step pays the same dispatch and the same
per-layer cost, so the arm-over-base ratio is invariant in sequence length. A saving that
looks small per step does not accumulate into a large one over a long generation. What makes
it large is a wider deployment.

### What the rule costs in quality

Paired `delta`NLL on 32 windows of held-out wikitext, same windows across arms, mean +- SE of
the paired difference; **bold** is resolved at 3 sigma:

| | `base2` (repeat) | `keep=6` | `keep=4` | `keep=2` |
| --- | --- | --- | --- | --- |
| Qwen3-30B-A3B | +0.0004 +-0.0003 | -0.0006 +-0.0008 | **+0.0100 +-0.0016** | **+0.1594 +-0.0047** |
| Qwen3.5-35B | -0.0009 +-0.0007 | +0.0022 +-0.0013 | **+0.0180 +-0.0024** | **+0.1051 +-0.0058** |
| Gemma-4-26B-A4B | -0.0000 +-0.0000 | +0.032 +-0.070 | +0.126 +-0.113 | +0.420 +-0.154 |

This is the axis with the statistical power, and it says `keep=6` is free on both Qwen families
and **`keep=4` is not**: +0.0100 nats at t=6.3 and +0.0180 at t=7.6, both replicated at the same
size by a second independent pass of all five arms (+0.0096 +-0.0016, +0.0185 +-0.0023). The
common claim that the bottom of a top-8 selection is substitutable holds here down to `keep=6`
and starts to cost at `keep=4`, well before it becomes visible on a benchmark.

Gemma is a weaker instrument on this axis, not a better result: its paired per-window spread is
two orders of magnitude wider than the Qwens' (+-0.07 against +-0.0008), so nothing below
`keep=2` resolves (t=2.7 there). Its `base2` arm reproduces `base` to five decimals, so the
pairing is sound - the spread is real per-window variance under re-routing.

Gemma's absolute NLL is high (5.66 nats) and that is the checkpoint, not the harness: the
offline HF path gets 5.85 for it on a different corpus. Prepending a `<bos>` to these
mid-corpus windows - the obvious suspect - was measured and makes it worse (9.73 nats, and
`base` stops reproducing `base2`), so the windows are scored without one. See `nll()`.

### What the benchmark can and cannot see

Accuracy is GPQA, 198 items, scored by option likelihood and paired per item. It resolves one
arm in the whole sweep - Qwen3.5 at `keep=2`, -5.05 points at `p=0.041` - and that is the axis'
ceiling, not evidence the other arms are free:

* All three baselines sit at or near chance (0.2727, 0.3283, 0.2222 against 0.25), so there is
  almost no headroom for a degradation to show up in.
* The stock engine, loaded twice, disagrees with itself on 0-12 of the 198 items, which is
  +-2 accuracy points of instrument noise before any arm is applied.
* Qwen3-30B at `keep=4` reads nominally **positive** (+4.04, `p=0.077`) on the same arm the NLL
  axis resolves as a real loss. At near-chance accuracy that is noise with a sign, and reporting
  it as a gain would be the easiest way to cheat here.

### The backend where the saving exists

`allgather_reducescatter` has nothing to save, so the test that matters is DeepEP, whose
dispatch volume *is* per-token fan-out. vLLM 0.31 only builds it under data parallelism, which
needs one process per rank - so this arm runs through a real `vllm serve`
(`experiments/serve_run.py`, the `dp` tags). It does not pay there either:

| | `base2` (repeat) | `keep=6` | `keep=4` | `keep=2` | instrument |
| --- | --- | --- | --- | --- | --- |
| Qwen3-30B-A3B | +12.46% | +3.16% | -1.43% | +5.11% | ±2-2.5%, baseline shifted +12% |
| Qwen3.5-35B | +3.78% | -4.04% | -6.04% | -4.14% | ±0.3-0.7% |
| Gemma-4-26B-A4B | -1.16% | -3.22% | -5.21% | -2.85% | ±2.9-3.8% |

Read the `base2` column first. On Qwen3-30B the stock engine, loaded and measured a second
time, is **12.5% faster than itself** at `p=0.008` - ten times the allgather band - so that row
resolves nothing and the three keep arms scattered around zero are noise, not a result. Gemma's
repeats drift monotonically downward inside every engine load (2949 → 2540 out-tok/s across
five warm repeats), a clock or neighbour effect, and no arm there resolves either.

Qwen3.5 is the one serve row with a usable instrument, and it is negative: every keep arm loses
4-6% against its own base and 8-10 points against the repeat. That is the honest answer to the
question the offline microbenchmark raised - the dispatch saving is real (+8-9% on the
all-to-all alone, measured in isolation) and smaller than what the rewrite costs around it.

One confound to state rather than hide: DeepEP HT with `data_parallel_size > 1` disables CUDA
graphs (`vllm/config/compilation.py`), which is why serve-mode absolute throughput is ~2.4K
out-tok/s against ~9K offline. It applies to every arm equally, so the comparison is fair - but
this kernel's cost is per-layer Python work, and the one backend where its saving exists is
also the backend that cannot graph that work away. A fused implementation of the rewrite, or a
DeepEP path that keeps CUDA graphs, is the experiment that could still move this.

## The measurements

Two independent ones, because they fail in different ways.

**In vLLM - `experiments/vllm_run.py`, `experiments/compare.py`.** A stock engine, four GPUs,
expert-parallel, nothing swapped but the router. One process per arm, since `PAR_KEEP` is read
when the engine loads. Throughput is wall clock for a fixed decode workload with `ignore_eos`
and a fixed token batch, so no arm can look fast by doing less; accuracy is GPQA scored by
option likelihood, paired per item. The honest resolution of the y axis is `base2` - the stock
engine loaded and measured a second time - and the figure draws that band.

Three families, not four: vLLM 0.31 has no native Hunyuan-MoE implementation and serves
`HunYuanMoEV1ForCausalLM` through its Transformers backend, which executes the checkpoint's own
modelling file - and that file still selects attention from a hard-coded class dict, so it
rejects the `vllm` attention implementation the backend installs. Hunyuan therefore appears in
the offline sweep only. Making it load would mean rewriting someone's modelling code and then
measuring that, which is not the same experiment.

**Offline, isolating the all-to-all - `experiments/{accuracy,bench,latency}.py`.** The vLLM
number is end-to-end, so a real saving in the dispatch shows up diluted by everything else in
a forward pass. The offline sweep replays recorded routing through one hand-built
expert-parallel layer over NCCL `all_to_all_single` and times the collective itself across a
batch sweep, which is the only way to see the mechanism. It is a model of serving, not serving;
the pair is the point.

## What keeps it honest

* **A no-op arm.** `keep = top_k` makes the allowed set the one the baseline already visits, so
  it must reproduce the baseline exactly. A non-zero delta is a hard stop, not a warning.
* **A repeat baseline.** `base2` is the stock arm run again in its own process. Every number
  here is wall clock on a shared cluster; a gap smaller than what one configuration reports
  twice is not a speedup.
* **A random control** (offline) making the same number of substitutions with no placement
  benefit, which separates "the bottom experts do not matter" from "our replacement is good".
* **Discordant counts** next to every McNemar p. Below a handful of discordant pairs the honest
  reading is "not resolved", not "no difference".
* **A kernel self-check** that runs on the GPU before anything is measured: the Triton path
  against the torch reference on a shuffled selection, and the mean fan-out against the exact
  hypergeometric closed form.
* **Two things that bias against the offline result.** No compute is overlapped with the
  collectives, and the local experts run as a capacity-padded batched GEMM, so re-routing -
  which concentrates tokens onto already-hot GPUs - is charged padding the baseline does not
  pay. Both inflate compute, understating the communication fraction this shrinks. The padding
  factor and the collective-only time are reported per arm.

## Reproducing

```bash
# the real measurement: one vLLM engine per arm
MODEL=/models/qwen3-30b-a3b TAG=q30rh ARMS=base,base2,6,4,2 experiments/run.sh
python experiments/compare.py

# the offline sweep: needs a cached token stream at $STREAMS/$STREAM.npy
MODEL=/models/qwen3-30b-a3b TAG=q30rh experiments/run.sh
python experiments/plot.py
```

On OpenShift, one Job per model (`ARMS` switches a job to the vLLM path):

```bash
experiments/k8s/launch.sh NAME=q30 TAG=q30rh MODEL=/models/bf16      ARMS=base,base2,6,4,2
experiments/k8s/launch.sh NAME=q35 TAG=q35   MODEL=/models/q35_bf16  ARMS=base,base2,6,4,2
experiments/k8s/launch.sh NAME=gm  TAG=gm26b MODEL=/models/gm_bf16   ARMS=base,base2,6,4,2
# Hunyuan has no vLLM arm - see "The measurements" - so it runs the offline sweep only:
experiments/k8s/launch.sh NAME=hy  TAG=hy13b MODEL=/models/hy_bf16   TF4=1 BATCH=1
```

`SERVE=1 DP=2 A2A=deepep_high_throughput` runs the same arms against a real `vllm serve` on the
DeepEP dispatch instead of the offline API. Give those runs their own `TAG`; the backend is a
measurement condition, not an implementation detail.

## Files

| | |
| --- | --- |
| `src/par/kernel.py` | the rule: Triton kernel, torch reference, fan-out metric, self-check |
| `src/par/plugin.py` | the vLLM hook and the guards that decline to apply it |
| `docker/`, `helm/` | the stock serving image with the router swapped, and a chart for it |
| `experiments/vllm_run.py` | one arm in real vLLM: GPQA and decode throughput |
| `experiments/compare.py` | the vLLM figure and the paired statistics |
| `experiments/accuracy.py` | offline `delta-NLL` per arm, plus the routing traces |
| `experiments/bench.py` | offline GPQA per arm, McNemar's exact paired test |
| `experiments/latency.py` | offline expert-parallel MoE layer time under those traces |
| `experiments/plot.py` | the offline figure |
| `experiments/hunyuan_capacity.py` | Hunyuan's gate drops tokens past a capacity equal to the mean load; this raises it to the true maximum |
| `experiments/k8s/` | one Job per model |
