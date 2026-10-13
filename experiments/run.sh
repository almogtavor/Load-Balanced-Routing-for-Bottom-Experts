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

# One IB device, named - and which one it is depends on the pod, so it is resolved here rather
# than passed in. `nvidia.com/gpu: 1` out of eight hands each rank an arbitrary GPU, and
# GPU-initiated RDMA wants the NIC under the same PCIe root: on the host we probed the granted
# GPU sat at 3b:00.0, which `nvidia-smi topo -m` puts PIX to `ibp1` and NODE to the other
# seven - so the `ibp0` we had been pinning every rank to was the wrong device on that rank.
# `eth0` is beside it because UCX needs `tcp` for the NIXL agent metadata exchange, and `tcp`
# with no netdev of its own is what brought `loadRemoteMD` -3 back.
if [ -z "${UCX_NET_DEVICES:-}" ] && [ -n "$(ls -d /sys/class/infiniband/ibp* 2>/dev/null)" ]; then
  gpu="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | cut -d: -f2- \
         | tr 'A-Z' 'a-z')"
  root="$(readlink -f "/sys/bus/pci/devices/0000:${gpu}" | cut -d/ -f4)"
  for d in /sys/class/infiniband/ibp*; do
    [ "$(readlink -f "${d}/device" | cut -d/ -f4)" = "${root}" ] || continue
    export UCX_NET_DEVICES="$(basename "${d}"):1,eth0"
    break
  done
  echo "gpu ${gpu} under ${root} -> UCX_NET_DEVICES=${UCX_NET_DEVICES:-<none local>}"
fi

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
  # Multi-node expert parallelism is one pod per node (an indexed Job), and only rank 0
  # benchmarks. The peers hold expert shards and route their own tokens, so they need the
  # arm's PAR_KEEP too - the head publishes it to the shared PVC before each arm. A peer's
  # engine dies when the head kills its own between arms, which is the restart signal.
  mkdir -p runs
  sent="runs/mn_${TAG}.arm"
  if [ "${JOB_COMPLETION_INDEX:-0}" != "0" ]; then
    while :; do
      arm="$(cat "${sent}" 2>/dev/null || echo wait)"
      case "${arm}" in
        done) exit 0 ;;
        wait) sleep 5; continue ;;
      esac
      ARM="${arm}" "${PY}" experiments/serve_run.py || true
      sleep 5
    done
  fi
  [ "${NODES:-1}" = "1" ] || echo wait > "${sent}"
  for rep in $(seq "${REP0:-0}" $(( ${REP0:-0} + ${REPS:-1} - 1 ))); do
    for arm in ${ARMS//,/ }; do
      # Published first, then a pause: the peers still running the previous arm have to see
      # their engine go away and come back on this one before the head waits for /health.
      [ "${NODES:-1}" = "1" ] || { echo "${arm}" > "${sent}"; sleep 30; }
      REP="${rep}" ARM="${arm}" "${PY}" "${script}"
    done
  done
  [ "${NODES:-1}" = "1" ] || echo done > "${sent}"
  exit 0
fi

"${PY}" experiments/accuracy.py
"${PY}" experiments/bench.py
"${PY}" -m torch.distributed.run --nproc_per_node="${GPUS:-4}" experiments/latency.py
