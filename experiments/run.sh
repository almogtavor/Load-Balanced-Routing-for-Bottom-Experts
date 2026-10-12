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
  for arm in ${ARMS//,/ }; do
    ARM="${arm}" "${PY}" experiments/vllm_run.py
  done
  exit 0
fi

"${PY}" experiments/accuracy.py
"${PY}" experiments/bench.py
"${PY}" -m torch.distributed.run --nproc_per_node="${GPUS:-4}" experiments/latency.py
