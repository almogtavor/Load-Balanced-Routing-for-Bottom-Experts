# Frozen paper follow-up sources — 18 October 2026

These are byte-identical snapshots of the scientific helpers used by the paper
follow-up. `source-manifest.json` records every included file's SHA-256 and size.
Check these copies with `python tools/verify_reproduction.py` from the root.
This verifies file integrity, not raw scientific acceptance.

Keep these snapshots immutable: paths, native runtime identities and helper hashes
are part of their acceptance contracts. New studies should use a new directory,
new manifests and an explicitly reviewed protocol.

## Layout

| Directory | Purpose and active entry points |
| --- | --- |
| `performance/` | Clean six-sweep prefill replication: `prefill_e2e.py`, separate `preflight_probe.py`, fixed synthetic `prompts64.json`, CPU fixtures |
| `hybrid/` | Separate PAR prefill / stock decode with actual NIXL handoff: `hybrid_outer_driver.py`, `hybrid_driver.py`, `hybrid_timing.py`, native request/transfer accounting, raw readers and GSM scorer |
| `causal/` | Serial prefix-only HellaSwag scoring: `causal_outer_v3.py`, `causal_driver_v3.py`, `heldout_causal_v2.py`, worker/probe contracts and predecessor helpers imported by those entry points |
| `stochastic/` | Nine-method stochastic-tail quality study: `campaign.py`, task adapter, opt-in `stochastic_tail` package, training gate and fixtures |
| `analysis/` | Complete raw causal audit, frozen paired-item report protocol/statistics and presentation-only LaTeX formatter |

Earlier helper versions remain where the accepted drivers import them or their
tests compare policy changes. They are dependencies/provenance, not additional
completed experimental observations. The older raw causal reader is included to
preserve its failed `"8.0"` versus `"8"` environment serialization check; the
accepted `audit_causal_raw_v3r2.py` repairs exactly that expected string. It does
not change the numeric penalty, scoring, tolerances or source checks.

## Runtime and source identity

The completed serving studies used production commit
`57c67bbe4228313a684d15815a684209f56e4de0`, model
`Qwen/Qwen3-30B-A3B-Instruct-2507` at revision
`0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`, and the immutable image
`vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`.
The model weights, downloaded datasets and image are not vendored here.

To inspect the exact production source independently of later research changes:

```sh
git show 57c67bbe4228313a684d15815a684209f56e4de0:src/placement_router/kernel.py
python reproduction/2026-10-18/performance/prefill_e2e.py --help
python reproduction/2026-10-18/analysis/causal_report_statistics_v1.py --help
```

GPU execution requires the pinned runtime, verified checkpoint/input files,
worker source manifests, and the harness's ownership/provenance preconditions.
This directory is not a cluster allocator. Authentication, private cluster
connections, pod creation/deletion and collection scripts are intentionally
excluded. No command here should be interpreted as allocating a GPU automatically.
Do not bypass an ownership or hash check to reuse a previous study's plan.

## Tests and replay boundaries

The repository's default CPU suite is `python -m pytest` from the root.
The archived suites use local module names such as `common_v2`; run different
snapshot directories in separate Python processes to avoid module-name collisions.
For example, the prefill fixtures are self-contained:

```sh
(cd reproduction/2026-10-18/performance && python -m unittest test_prefill_e2e)
```

Historical ownership/integration fixtures and complete raw audits additionally
require the corresponding archived evidence files. They are not advertised as
self-contained tests. The raw archives are intentionally not committed: the
verified H200 archive has 84,521 files and the causal archive 275,837 files.
Their content hashes are recorded in the [published summaries](../../results/2026-10-18/README.md).
Summary JSON is an audit output, not a replacement for the raw wire, source,
closure and native-routing evidence required to replay acceptance.

After a complete raw audit, the causal report requires its SHA-256, matching plan
SHA-256, and the unchanged `analysis/final-acceptance-report-protocol-v1.json`.
`render_causal_tables_v1.py` only formats that accepted report; it computes no
additional statistics. Partial older campaigns contribute no comparative rows.

## Measurement scope

Keep clean HTTP timing separate from instrumented preflight/quality/profiling
launches. Stochastic quality does not establish stochastic serving throughput;
the separate 39-launch stochastic timing matrix was not run. Serial causal
scoring recomputes the full prefix for every target; it is neither incremental
KV scoring nor the quality of a PAR-prefill/stock-decode deployment. See the
[result notes](../../results/2026-10-18/README.md) before interpreting an effect.
