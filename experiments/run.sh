#!/usr/bin/env bash
# Every measurement for one model. `PY` is the interpreter to use (the pod sets it to a
# transformers-4.x venv for Hunyuan's HF path); `GPUS` must match the Phase A trace.
#
#   MODEL=/models/bf16 TAG=q30rh ./run.sh                   # offline: accuracy, bench, latency
#   MODEL=/models/bf16 TAG=q30rh ARMS=base,base2,4 ./run.sh # real: vLLM, one engine per arm
set -xeuo pipefail
PY="${PY:-python3}"
# Run from the repo root so every script writes to the one `runs/`, while sys.path[0] stays
# the script's own directory and the `import placement` siblings keep resolving.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -n "${ARMS:-}" ]; then
  # Installed from a copy, not from /results: the entry point has to be importable in every
  # worker process, and four jobs share that PVC path.
  cp -r . /tmp/par
  pip -q install --no-deps --no-build-isolation -e /tmp/par
  # The Triton path has no CPU fallback to be checked against on a laptop, so the first thing
  # a GPU job does is prove it agrees with the reference on this hardware.
  "${PY}" src/par/kernel.py
  # SERVE=1 measures through a real `vllm serve` instead of the offline API, which is the
  # only way to reach the DeepEP dispatch - see the header of serve_run.py.
  if [ "${SERVE:-0}" = "1" ]; then script=experiments/serve_run.py
  else script=experiments/vllm_run.py; fi
  # `base` and `base2` are the same configuration launched twice, and on the first clean
  # sweep they disagreed by 5.3% while the warm repeats inside each launch spread 0.8%. So a
  # launch is the unit of analysis, not a repeat: repeats measure one engine's steady state,
  # and the arms differ by which engine was started. REPS sweeps every arm once per pass, so
  # whatever drifts between launches lands on all arms rather than on whichever ran first.
  # REP0 offsets the index so a second job can add sweeps to a tag already measured
  # instead of overwriting its files.
  for rep in $(seq "${REP0:-0}" $(( ${REP0:-0} + ${REPS:-1} - 1 ))); do
    for arm in ${ARMS//,/ }; do
      REP="${rep}" ARM="${arm}" "${PY}" "${script}"
    done
  done
  exit 0
fi

"${PY}" experiments/accuracy.py
"${PY}" experiments/bench.py
"${PY}" -m torch.distributed.run --nproc_per_node="${GPUS:-4}" experiments/latency.py
