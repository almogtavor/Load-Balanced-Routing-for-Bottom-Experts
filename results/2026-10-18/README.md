# Audited follow-up results

Model: Qwen3-30B-A3B-Instruct-2507, checkpoint revision
`0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`; production source
`57c67bbe4228313a684d15815a684209f56e4de0`. The frozen scientific helpers are in
[`reproduction/2026-10-18/`](../../reproduction/2026-10-18/).
The paper contains the full methods and earlier studies. These summaries retain
all planned arms/endpoints, including negative or inconclusive results.

## Prefill and decode

The clean long-prefill replication (four H100 GPUs, EP4, 64 prompts of 4,096
tokens, one output token, six fresh paired sweeps) measured throughput changes
of +6.69% for keep=6 and +8.39% for keep=4 with penalty 8. Their launch-level
95% intervals were [4.45, 8.97]% and [5.63, 11.23]%. These are separate measurements from
the original author's +6.3/+7.9% results, with a pinned checkpoint and protocol.

The completed split-serving test uses one eight-H200 host: four PAR prefill GPUs
and four stock decode GPUs, both EP4, with actual NIXL handoff. Its clean workload
is 256 prompts of 1,024 tokens, concurrency 64, and 128 output tokens with EOS
ignored. The matched baseline also uses stock prefill/stock decode with NIXL.
Three fresh paired sweeps give:

| Prefill router | Throughput change | Nominal 95% interval | GSM8K flexible correct / 1319 | Strict correct / 1319 |
| --- | ---: | ---: | ---: | ---: |
| Stock | Reference | — | 1262 | 1175 |
| keep=6, penalty=8 | +0.32% | [−3.59, +4.40]% | 1263 | 1177 |
| keep=4, penalty=8 | +0.63% | [−4.72, +6.28]% | 1256 | 1167 |

There is **no demonstrated end-to-end throughput gain** in this split-serving
workload. The intervals are descriptive, pointwise paired-log-t intervals with
two degrees of freedom, not an equivalence test or an SLO guarantee. No causal
phase breakdown, cross-node result, or tensor-level KV equivalence is established.
An auxiliary CPU archival command failed before its copying routine during the
first keep=6 launch; its conservative failure/inspection bracket overlaps that
launch's three measured cohorts. All rounds remain included, with no claim of
zero CPU scheduling interference or attribution of the result to that event.

GSM8K uses one instrumented launch per routing setting, greedy generation with
natural EOS and a 2,048-token cap. These wall times are not performance results.
All capped responses and parser failures remain in the denominator. Strict
extraction finds no answer in 108/103/110 stock/keep6/keep4 responses; flexible
parse failures are zero. Small differences do not establish quality equivalence.
All 15 pairs (3 proof, 9 timing, 3 quality) completed and passed raw replay.
Exact numbers and qualification details: [`split-serving.json`](split-serving.json).

## Serial causal HellaSwag

Four H100 GPUs, EP4, penalty 8; six training diagnostic launches followed by
three fresh scoring launches. All arms score the same sorted 256-item public
validation subset selected by `random.Random(0).sample(range(10042),256)`.
Each scored token receives only its preceding prefix as input, serially, with
a full re-prefill. The target ID is requested as an output log-probability entry.
This policy differs from incremental KV reuse and mixed prefill/decode routing.

| Router | Prefix normalized correct / 256 | Change versus stock | Conditional 95% item interval |
| --- | ---: | ---: | ---: |
| Stock | 206 | Reference | — |
| keep=6 | 203 | −1.17 pp | [−2.73, 0.00] pp |
| keep=4 | 202 | −1.56 pp | [−3.91, +0.78] pp |

The report retains all 12 arm/scoring endpoints and all 14 prespecified paired
accuracy and candidate-score contrasts. Its 10,000 common paired-item resamples
(seed 0) give **conditional item sensitivity**, unadjusted and descriptive;
there is one launch per arm, not a launch/host confidence interval. The subset
is public validation, not an unseen test or the complete 10,042-item dataset.

Teacher and prefix scoring have equal aggregate normalized accuracy for each
arm, but five keep6 and six keep4 chosen answers change. Stock changes four
answers under raw scoring. Teacher-minus-prefix differences include packing and
numerical execution as well as candidate conditioning; the stock control must
be retained. Candidate-score means average four endings per item and are not
gold-target NLL/perplexity. Exact counts, intervals, gains/losses and changed
item indices: [`causal-hellaswag.json`](causal-hellaswag.json).

## Evidence fingerprints

Both raw readers and statistics were independently replayed before publication.
All 12 GPUs from these two campaigns were released after verified full backups.
Raw archives and cluster operations are excluded from Git.

| Artifact | SHA-256 |
| --- | --- |
| H200 whole archive | `fbb5a3915c8f0d74fae39a51668e3563e78bf2ee280f4ce228547534b41732e7` |
| H200 accepted raw audit | `65ff53c3fea2d89e437b9ae2e2a3a895c53c3f07fe8320bb8ac42bd829e556ca` |
| Causal whole archive | `d09e6f6bacb2b1c8ed6f7e9315087e02e14ee38bf5953f9b010e376bac9f999e` |
| Causal accepted raw audit | `c966696beb2465f48ca2515c2a1e886819ce627523ebd71ee5fc25b3009aa377` |
| Exact causal report included here | `a32d9c61949ee40a3f45a9cc0a7c0af768fd095987bb1135639f5a44797e8563` |
