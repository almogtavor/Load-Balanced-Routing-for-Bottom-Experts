#!/usr/bin/env bash
# Every measurement for one model. `PY` is the interpreter to use.
#
#   MODEL=/models/bf16 TAG=q30rh ./run.sh                   # offline: paired NLL per setting
#   MODEL=/models/bf16 TAG=q30rh SETTINGS=base,base2,4 ./run.sh # real: one vllm serve per setting
set -xeuo pipefail
PY="${PY:-python3}"
# Run from the repo root so every script writes to the one `runs/`, while sys.path[0] stays
# the script's own directory and the `import placement` siblings keep resolving.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -n "${SETTINGS:-}" ]; then
  # Installed from a copy, not from /results: the entry point has to be importable in every
  # worker process, and four jobs share that PVC path.
  cp -r . /tmp/placement-router
  pip -q install --no-deps --no-build-isolation -e /tmp/placement-router
  # The Triton path has no CPU fallback to be checked against on a laptop, so the first thing
  # a GPU job does is prove it agrees with the reference on this hardware.
  "${PY}" src/placement_router/kernel.py
  # A real `vllm serve` is the only shape vLLM 0.31 builds the DeepEP all-to-all in - see the
  # header of serve_run.py. SCRIPT names a one-off measurement to run instead of the setting sweep
  # (imbalance.py scores every setting inside one engine load, so it wants SETTINGS=<anything> only to
  # enter this path).
  script="${SCRIPT:-experiments/serve_run.py}"
  # `base` and `base2` are the same configuration launched twice, and on the first clean
  # sweep they disagreed by 5.3% while the warm repeats inside each launch spread 0.8%. So a
  # launch is the unit of analysis, not a repeat: repeats measure one engine's steady state,
  # and the settings differ by which engine was started. REPS sweeps every setting once per pass, so
  # whatever drifts between launches falls on all settings rather than on whichever ran first.
  # REP0 offsets the index so a second job can add sweeps to a tag already measured
  # instead of overwriting its files.
  mkdir -p runs
  for rep in $(seq "${REP0:-0}" $(( ${REP0:-0} + ${REPS:-1} - 1 ))); do
    # SCRIPT may list several, e.g. experiments/evalrun.py,experiments/nllserve.py: every setting finishes one before the
    # next starts, and each setting starts its own server.
    for one in ${script//,/ }; do
      for setting in ${SETTINGS//,/ }; do
        REP="${rep}" SETTING="${setting}" "${PY}" "${one}"
      done
    done
  done
  exit 0
fi

PYTHONPATH=src "${PY}" experiments/accuracy.py
