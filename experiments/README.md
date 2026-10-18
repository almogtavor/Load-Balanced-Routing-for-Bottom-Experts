# Experiment entry points

Run commands from the repository root. Existing filenames remain stable because
launchers, captured commands and source-hash manifests refer to them. Tests live
in `tests/`; immutable paper-study snapshots live in `reproduction/2026-10-18/`.
Neither is installed as part of the production plugin.

## Setup and tests

Use Python 3.10 or newer:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
python -m pytest
```

For offline model quality evaluation, install `.[quality]` as well. Serving needs
the vLLM/Torch/CUDA/DeepEP environment recorded by the particular experiment;
install the router there with `pip install --no-deps -e .` to preserve its runtime.
CPU tests do not download models, allocate GPUs, or contact clusters. GPU checks
are explicit operations, not part of the default test suite.

## Tools by purpose

| Purpose | Entry points |
| --- | --- |
| Original layer timing and routing statistics | `serve_run.py`, `moetime.py`, `imbalance.py`, `moeplot.py`, `lbgain.py`, `readme_tables.py` |
| Controlled HTTP serving and separate profiling | `serve_profile.py`, `serve_focus.py`, `serve_backend_profiles.py`, `serve_backend_confirm.py`, `serve_sms_confirm.py` |
| Paper-protocol planning, timing and profiles | `paper_reproduce.py`, `paper_probe.py`, `paper_profile.py`, `analyze_paper_profiles.py` |
| Natural-text quality and routing parity | `accuracy.py`, `moe.py`, `placement.py`, `prepare_text.py`, `paper_quality.py`, `analyze_quality.py` |
| GSM8K/HellaSwag task scoring and worker contracts | `task_bench.py`, `task_bench_probe.py` |
| Trace, serving and telemetry analysis | `analyze_profiles.py`, `diagnose_profiles.py`, `audit_router_casts.py`, `analyze_serving.py`, `analyze_telemetry.py` |
| Kernel/DeepEP research prototypes | `kernel_candidates.py`, `kernel_direct.py`, `kernel_sweep.py`, `replay_deepep.py`, `replay_deepep_graph.py`, `deepep_sms_extension.py` |
| Local GPU/cgroup telemetry | `monitor.py` |
| Existing Kubernetes deployment templates | `k8s/`, `run.sh` |

`paper_reproduce.py` is the earlier instrumented protocol, not the six-sweep
clean prefill replication. That completed replication is preserved as
`reproduction/2026-10-18/performance/prefill_e2e.py`.

## Inspect a plan without launching a model

```sh
python experiments/serve_profile.py --help
python experiments/paper_reproduce.py --mode plan --output runs/example-prefill
python experiments/paper_quality.py --mode plan --output runs/example-quality
python experiments/task_bench.py --help
```

Preparation, preflight, execution and analysis are separate where the harness
supports them. Use a fresh output directory for a changed protocol. Do not mix
partial campaigns, pick fast rounds after seeing outcomes, or remove source,
checkpoint, request-count or native-routing checks to make a run pass.

## Prototype compatibility

`src/placement_router/experimental.py` preserves the older per-token selector
prototypes. Their CUDA launch ABI and load-penalty semantics predate the current
16-token tile kernel. CUDA use is explicitly rejected with that kernel until a
candidate is ported and validated. CPU fallbacks still use the current production
reference. Leave `PLACEMENT_EXPERIMENTAL_SELECTOR` unset (or `baseline`) for the
paper implementation. Kernel-only timing is not an HTTP serving speedup.

## Outputs and measurements

Write generated records under `runs/`; traces, models, virtual environments,
local PDFs and temporary downloads are ignored. The tracked plots already under
`runs/` remain historical results. Compact audited summaries belong in `results/`.
Cluster authentication and scheduling are separate from these scripts; no saved
connections, kubeconfigs, Keychain helpers or access tokens belong in the repo.

Distinguish HTTP wall-clock throughput, per-request latency, per-rank kernel
durations, and routing statistics. CUDA durations can overlap; summing them is
not a causal decomposition of end-to-end latency. Teacher-forced scoring with
tile-coupled routing is candidate-conditioned pseudo-likelihood, not causal NLL.
