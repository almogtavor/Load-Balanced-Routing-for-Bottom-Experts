# Placement-Aware MoE Router

A vLLM plugin that re-routes the bottom of each token's expert selection onto the GPUs the
token is already being sent to, and the experiments that measure what that costs and buys on
four MoE families - three of them inside vLLM, the fourth offline only.

A token routed to top-8 of a few hundred experts in an expert-parallel deployment usually has
to be sent to *every* GPU, because eight experts drawn from that many rarely share a device.
The all-to-all dispatch costs one copy of the token per destination GPU. The selection is not
uniformly important - the top few experts carry the result and the bottom ones can be
substituted with little accuracy loss (arXiv 2412.00099, which also does the restricted-set
reroute this rule reuses; see [the literature](#where-this-sits-in-the-literature)) - so the
bottom slots are free parameters, and there is an obvious thing to spend them on:

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
fixes the index dtype. `src/par/vendor/fused_topk_router.py` is that subclass vendored
verbatim from v0.31.0 with two imports and four lines changed: the Triton kernel in
`src/par/kernel.py` *replaces* the fused top-k rather than running after it, so an arm spends
the one launch per MoE layer stock vLLM also spends. Gemma routes through `CustomRoutingRouter`
instead, so `src/par/plugin.py` wraps that method and substitutes the kernel for the model's
own routing function, folding in its per-expert scale. The dispatch, the combine, the expert
GEMMs and every other routing variant are untouched, which is the only reason this is safe to
drop into a stock image.

Expert-to-rank needs no map tensor: vLLM shards experts in contiguous blocks, so rank is
`expert * ep_size // num_experts`, computed in the kernel.

The router **passes the selection through unchanged, with one warning**, wherever the rule's
premise does not hold. `_unsupported` in `src/par/plugin.py` is the whole list: a router class
or custom routing function that does not rank by gate logits alone (grouped top-k, score
correction, zero-expert routing - "the top `keep` by logit" would not be the head the router
chose), a non-softmax `scoring_func` or routing without renormalization (the kernel's softmax
is scoped to the selection), EPLB replication (expert id no longer implies a rank), `keep >
top_k`, a logit width that disagrees with the expert count, a rank count of 1 (no all-to-all
to shrink), a non-divisible or too-small expert block (no substitute is guaranteed to exist),
and `ep_size > 32` (the kernel tracks the head's ranks in one int32 lane). `keep == top_k` is
allowed through on purpose: that arm is the control. A routing rule that silently applied
where its premise fails would be measured as free and serve garbage.

## What it measures out at

Three results, and they separate cleanly. **The plugin is free to run**: `keep = top_k` asks
for no substitution, and across two engine loads it comes out at +0.22% throughput. **The rule
costs real accuracy** below `keep=6`, resolved at 3 sigma on both Qwen families. **Nothing is
faster in any configuration measured so far** - every arm lands between 0 and -5%, and
[why a smaller payload is not a faster step](#why-a-smaller-payload-is-not-a-faster-step) is
what the evidence says about that. That claim is scoped to one measurement condition: under
data parallelism vLLM caps the scheduler at 256 tokens per step, not the usual 2048
(`DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP`, applied in `arg_utils.py` whenever the flag
is unset), so every serve run in the table below chunked its prompts into 256-token steps and
the dispatch payload per all-to-all stayed small however many requests were in flight. A rule
that shrinks payload and not per-collective latency cannot show there. `BATCH_TOK` now passes
the budget explicitly and every artefact records it, and the re-measurement at a real budget
is in: `q30pf`, 32 requests of 4096 prompt tokens at `--max-num-batched-tokens 8192`, ep=4 on
one host, five base launches over two sweeps. It did not change the sign. The arms come out
monotone - `keep=8` **+0.33%**, `keep=6` **-0.49%**, `keep=4` **-3.16%**, `keep=2` **-5.43%** -
against a base spread of 0.64% and 0.22% in the two sweeps, which is a tighter instrument than
the capped runs ever gave. The no-op control costing nothing is what makes the rest of that
ladder readable: the loss at `keep=4` and below is the routing rule and not the plugin. The
budget was a real defect in the measurement and not the reason the saving does not convert.

An earlier set of throughput numbers was withdrawn, and the fix for it is why the control arm
above is clean. vLLM pads a batch for cudagraph and data-parallel capture and marks the padding
by forcing those tokens' expert ids to `-1`, so the dispatch and the experts drop them
(`VLLM_MOE_SKIP_PADDING`, on by default since 0.31). A `-1` is not a selection, and the first
kernel treated it as one: its `argmax` ran over an all-`-inf` block, fewer than `top_k` output
slots got written, and the row came out holding whatever `torch.empty` had left there. Real
expert ids, for tokens the unmodified engine never dispatches - so the baseline arms skipped
that padding and the rule's arms paid full dispatch and expert GEMM for it, which at decode
batch sizes is most of a measurement. It also explains the one failure that was not a quiet
bias: a `-1` reaching Gemma's index scatter asserted out of bounds mid-run
(`ScatterGatherKernel.cu:203`).

The kernel now declines any row that is not `top_k` distinct in-range experts and returns it
byte-for-byte, sentinel included; `src/par/kernel.py` asserts that on GPU before any arm runs.
That moved the no-op control from about -9% to +0.22%. The quality numbers were never affected
- vLLM's fused MoE imposes no expert capacity, so a padded token routed by mistake costs time
and takes nothing from a real token's output.

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
against the measured fan-out in the self-check:

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
(`experiments/serve_run.py`, the `dp` tags). Four GPUs, `dp=2 tp=2`, so `ep_size=4`;
`deepep_high_throughput`; 256 requests at 1024 prompt / 128 generated, except `gmdec`, which is
a decode-dominant 16 at 128/512, and `q30pf`, which is 32 at 4096/8. The first four rows ran at
vLLM's batched-DP default of **256 tokens per step**, which is the dispatch payload per
all-to-all and therefore the ceiling on what this rule can move - see the scope note above.
`q30pf` is the same hardware and backend with the budget set to 8192 instead, so it is the row
to read; the capped four are the launch-bound regime, kept because they are what the earlier
claims were made on.

Out-tok/s against that same launch's own base arms, median of the warm repeats
(`experiments/paired.py`):

| | launches | `keep=8` (no-op) | `keep=6` | `keep=4` | `keep=2` | base spread | decay |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-4-26B-A4B | 2 | **+0.22%** | -0.62% | **-4.41%** | **-5.00%** | 2.3-3.5% | ~0% |
| Qwen3-30B-A3B | 2 | -3.60% | -2.85% | -3.56% | -0.74% | 1.8-3.8% | +7-8% |
| Qwen3.5-35B | 2 | -4.15% | -1.66% | -0.70% | -1.60% | 0.6-3.0% | ~0% |
| Gemma, decode-heavy | 1 | +1.95% | - | +1.65% | +0.62% | 2.7% | +0.7% |
| Qwen3-30B-A3B, prefill-first, 8192-token steps | 2 (5 bases) | **+0.33%** | -0.49% | **-3.16%** | **-5.43%** | 0.22-0.64% | -3 to -4% |

Read the no-op column first, because it is the instrument. On Gemma it is +0.22%: the plugin
costs nothing measurable to run, and the keep arms there are the rule and nothing else. On both
Qwen rows it is *more negative than the keep arms*, which is incoherent - a control that asks
for no substitution cannot cost more than substituting - so those two rows resolve nothing.
`q30dp` also decays +7-8% from first warm repeat to last **on its own unmodified base**, which
puts its resolution floor above every effect in the table. And the decode-heavy row's positive
arms are a single launch whose largest number is the no-op control: noise with a sign.

That leaves two usable rows, and both are negative. Gemma: `keep=6` at -0.62% is inside the
base spread, `keep=4` at -4.41% and `keep=2` at -5.00% are outside it. `q30pf`, at a real token
budget and with five base launches agreeing to under 0.64%, is the cleanest measurement in this
repo and says the same thing with a tenth of the noise. Its no-op is +0.33% and its `keep=6` is
-0.49%, both inside the spread; `keep=4` at -3.16% and `keep=2` at -5.43% are an order of
magnitude outside it. So the rule is free as long as it only moves the eighth slot, and it
costs throughput monotonically in how many slots it moves. There is no arm anywhere in this
table where it pays.

A launch is the unit of analysis, not a repeat. An unmodified engine loaded twice measured
+5.4% and +5.9% against itself across launches, so only within-launch ratios against that
launch's own base arms are used, repeat 0 is discarded as cold, and the within-launch decay is
reported next to every row as what that row can resolve.

### Why a smaller payload is not a faster step

The rule looks like it does two things: moves fewer bytes, and makes a token wait on fewer
GPUs. The first is true. The second is not, and `experiments/fanout.py` is what separates them.
Three quantities decide a step, and the rule moves two of them:

| per step, `keep=4` vs `keep=8` | `ep=4`, B=256 | `ep=8`, B=256 |
| --- | --- | --- |
| volume on the wire | -24.1% | -37.4% |
| ranks taking part in the collective | 4.00 -> 4.00 | 8.00 -> 8.00 |
| busiest rank's share | -22.4% | -34.8% |

The middle row is the answer. The all-to-all is **one collective per layer per step over the
whole in-flight batch**, not one per token. Even 16 tokens at top-8 reach every rank with
near-certainty, so the union of ranks is the full world size at every `keep` and no rank is
ever dropped from the collective. The payload shrinks; the participation does not, and the
fixed per-collective cost - launch, handshake, sync - is paid in full either way. There is no
"waiting for fewer GPUs" to collect, at any `keep`.

This also refutes the opposite worry, which is worth stating because it was ours: substitutes
are drawn from ranks the head already visits, so one would expect traffic to pile onto hot
ranks and eat the saving. It does not. The busiest rank's load falls -11% to -35%, under a
gamma-skewed router as well as iid. Both the volume and the critical-path proxy improve, and
still nothing is faster.

Three things finish the explanation:

* **The bytes that shrink are on the cheap tier.** DeepEP counts ranks and nodes separately
  (`num_tokens_per_rank` vs `num_tokens_per_rdma_rank`): NVLink is charged per distinct rank,
  RDMA per distinct node. These runs are 2 GPUs per node, so `ep=4` is 2 nodes, where `keep=4`
  cuts rank fan-out -24.3% and node fan-out only **-5.9%** - top-8 already reaches both nodes,
  so almost nothing on the expensive tier is left to cut. What governs the node saving is the
  node count, not EP width: at N nodes it is exactly the rank saving at `ep=N`
  (`experiments/fanout.py` prints -5.9% at 2 nodes, -24.3% at 4, -37.5% at 8). Widening EP
  inside the same two hosts puts everything the rule gains on NVLink.
* **The step was not bandwidth-bound, and part of that was the harness.** `T_comm` is roughly
  `L + V/B_eff`: the rule moves `V`, never `L`. At the 256-token budget most of these runs
  inherited, `V` per dispatch is 256 rows of hidden state - order a megabyte at these hidden
  sizes, which NVLink clears in single-digit microseconds against a per-collective floor of
  tens. `L` dominated by construction, so no change in `V` could register. The engine's own
  hint says the same - `DeepEP: Disabling CUDA Graphs since DeepEP high-throughput kernels are
  optimized for prefill`.
* **Lifting the budget does not fix that, because it skips the regime entirely.** `q30pf` runs
  the same ep=4 host at 8192 tokens per step on 4096-token prompts, and `keep=4` still loses
  3.16%. A 32x4096-token prefill step is GEMM-bound, not link-bound: small steps are
  latency-bound and large ones are compute-bound, and on a single host there is no budget in
  between where NVLink is the constraint. The payload saving can only convert where the link is
  slow enough to matter, which means across nodes over RDMA - the shape the tier note above
  describes, and the one we cannot currently schedule.
* **Compute is untouched by construction.** The rule changes *which* rank holds a token's
  expert, never how many experts it has, so the `sum_t k` expert-token GEMM pairs are identical
  in every arm. Nothing is removed from the only part of the step that scales with the model.

The regime that would test the rule properly is more nodes. What the rule caps is *ranks* - a
token reaches at most `keep` of them - and nodes fall only as a consequence, so there is no
threshold, only saturation. Mean distinct nodes per token, top-8 against `keep=4`:

| nodes | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- |
| base top-8 | 1.99 | 3.62 | 5.32 | 6.58 |
| `keep=4` | 1.88 | 2.74 | 3.33 | 3.67 |
| | -5.9% | -24.3% | -37.5% | -44.2% |

At 2 nodes top-8 already reaches both, so there is nothing to confine. `keep=4` pins the count
near 3.3-3.7 however wide the deployment goes, so the saving grows with node count without
bound. And unlike the NVLink case the critical path moves with it: the busiest node drops
-34.6% iid and -21.1% skewed at 8 nodes, -40.2% and -25.5% at 16.

Participation is invariant at the RDMA tier too, though (8.00 -> 8.00, 16.00 -> 16.00), so the
per-collective floor is unchanged and a win needs the step to be RDMA-bandwidth-bound. Both
DeepEP decode paths work against that: high-throughput disables CUDA graphs, and low-latency
dispatches into a fixed padded buffer whose volume does not depend on fan-out at all.

**That shape was built and never measured** - `NODES=` in `launch.sh` renders one pod per node
with `rdma/roce_gdr`, and the job was killed before it wrote an artefact. It is an open
question, not a refuted one, and it is the one experiment that could still turn this around.

Everything from `experiments/fanout.py` is an iid (optionally gamma-skewed) Gaussian logit
model. It tests the **mechanism, not the magnitude**: a trained router's logits correlate
across tokens and layers in ways no synthetic draw reproduces.

## The measurements

Three, because they fail in different ways.

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

**Against a real server - `experiments/serve_run.py`, `experiments/paired.py`.** The offline
API all-gathers, whose volume is independent of per-token fan-out, so it cannot show this
saving at all; vLLM 0.31 gates the DeepEP all-to-all on data parallelism, which needs one
process per rank and therefore a real `vllm serve`. Those runs carry the headline throughput
table. A launch, not a repeat, is the unit of analysis: an unmodified engine loaded twice
measured +5.4% and +5.9% against itself, so only paired ratios inside one launch are usable
and `paired.py` reports nothing else.

**Offline, isolating the all-to-all - `experiments/{accuracy,bench,latency}.py`.** The vLLM
number is end-to-end, so a real saving in the dispatch shows up diluted by everything else in
a forward pass. The offline sweep replays recorded routing through one hand-built
expert-parallel layer over NCCL `all_to_all_single` and times the collective itself across a
batch sweep, which is the only way to see the mechanism. It is a model of serving, not serving;
the pair is the point.

## Where this sits in the literature

The rule is not new. Stated plainly:

* **The method has prior art.** *Mixture of Cache-Conditional Experts* (arXiv 2412.00099)
  keeps the top `J` of a top-k selection by score and fills the rest from a restricted allowed
  set, training-free and at inference time, including hard-reroute variants. Its `J` is
  literally our `keep`. The only thing that differs is what the allowed set is conditioned on:
  cache residency there, expert-to-GPU placement here. Anyone reading this as a new routing
  rule is reading it wrong.
* **The communication objective has prior art too, but it is not training-free.** C2R
  (arXiv 2504.01337) and Occult (arXiv 2505.13345) both bias routing toward fewer destination
  devices and both get there by changing the model - SFT in the first, modified fine-tuning in
  the second. The training-free version of that objective is what is left over.
* **What we claim is left.** No prior work we found measures this exact bias's quality cost at
  inference with the weights frozen, across families, against a random-substitution control;
  and none of it is a kernel inside a served engine with the throughput conversion measured
  rather than counted. Both of those are in this repo, and the throughput half currently says
  the saving does not convert - see above.

The premise the rule rests on - that the top few experts carry the result and the tail is
substitutable - is also 2412.00099's ablation, not ours. We re-measure it here because it is
load-bearing, not because it was in doubt.

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
MODEL=/models/qwen3-30b-a3b TAG=q30rh ARMS=base,base2,6,4,2,base3 experiments/run.sh
python experiments/compare.py

# the DeepEP shape, where the volume this shrinks is the volume the engine moves
MODEL=/models/qwen3-30b-a3b TAG=q30dp SERVE=1 DP=2 A2A=deepep_high_throughput \
  ARMS=base,base2,6,4,2,base3 experiments/run.sh
python experiments/paired.py

# no GPU: which tier the saving lands on, and whether anything on the critical path moves
python experiments/fanout.py

# the offline sweep: needs a cached token stream at $STREAMS/$STREAM.npy
MODEL=/models/qwen3-30b-a3b TAG=q30rh experiments/run.sh
python experiments/plot.py
```

On OpenShift, one Job per model (`ARMS` switches a job to the vLLM path):

```bash
experiments/k8s/launch.sh NAME=q30 TAG=q30rh MODEL=/models/bf16      ARMS=base,base2,6,4,2,base3
experiments/k8s/launch.sh NAME=q35 TAG=q35   MODEL=/models/q35_bf16  ARMS=base,base2,6,4,2,base3
experiments/k8s/launch.sh NAME=gm  TAG=gm26b MODEL=/models/gm_bf16   ARMS=base,base2,6,4,2,base3
# Hunyuan has no vLLM arm - see "The measurements" - so it runs the offline sweep only:
experiments/k8s/launch.sh NAME=hy  TAG=hy13b MODEL=/models/hy_bf16   TF4=1 BATCH=1
```

`SERVE=1 DP=2 A2A=deepep_high_throughput` runs the same arms against a real `vllm serve` on the
DeepEP dispatch instead of the offline API. Give those runs their own `TAG`; the backend is a
measurement condition, not an implementation detail.

`NODES=4 GPUS=2 DP=2` renders one pod per node for an `ep_size = DP * NODES` run spread across
hosts, which is the shape where the saving reaches the RDMA tier. Built, not yet measured.

## Files

| | |
| --- | --- |
| `src/par/kernel.py` | the rule: Triton kernel, torch reference, fan-out metric, self-check |
| `src/par/plugin.py` | the vLLM hook and the guards that decline to apply it |
| `docker/`, `helm/` | the stock serving image with the router swapped, and a chart for it |
| `experiments/vllm_run.py` | one arm in real vLLM: GPQA and decode throughput |
| `experiments/compare.py` | the vLLM figure and the paired statistics |
| `experiments/paired.py` | the serve throughput table, paired within each launch |
| `experiments/accuracy.py` | offline `delta-NLL` per arm, plus the routing traces |
| `experiments/moe.py` | finding and overriding a router across four families that agree on nothing |
| `experiments/placement.py` | the rule in torch, plus the analytic fan-out, shared by the offline scripts |
| `experiments/bench.py` | offline GPQA per arm, McNemar's exact paired test |
| `experiments/latency.py` | offline expert-parallel MoE layer time under those traces |
| `experiments/plot.py` | the offline figure |
| `experiments/serve_run.py` | one arm against a real `vllm serve`, which is the only shape vLLM 0.31 builds the DeepEP all-to-all in |
| `experiments/fanout.py` | which tier a fan-out saving lands on, and whether per-step participation moves with it |
| `experiments/hunyuan_capacity.py` | Hunyuan's gate drops tokens past a capacity equal to the mean load; this raises it to the true maximum |
| `experiments/run.sh` | the in-container driver: self-check, then the arms, then the sweep |
| `experiments/k8s/` | one Job per model |
