#!/usr/bin/env bash
# Render and apply `job.yaml` for one model.
#
#   ./launch.sh NAME=q30 TAG=q30rh MODEL=/models/bf16
#   ./launch.sh NAME=q35 TAG=q35   MODEL=/models/q35_bf16
#   ./launch.sh NAME=hy  TAG=hy13b MODEL=/models/hy_bf16 TF4=1 BATCH=1
#   ./launch.sh NAME=gm  TAG=gm26b MODEL=/models/gm_bf16
#
# `ARMS=base,base2,6,4,2,base3` switches the job to the real vLLM measurement, one engine load per
# arm; the offline three-phase run is what you get without it. Adding `SERVE=1 DP=2` runs those
# arms against a real `vllm serve` on the DeepEP dispatch instead of the offline API, which is
# the only shape where the volume this kernel shrinks is the volume the engine moves - vLLM's
# default all-gathers, and 0.31 gates the all-to-all kernels on data parallelism. Give those
# runs their own TAG (`q30dp`, `q35dp`, `gmdp`); the backend is a measurement condition.
#
# Add `NODE=<hostname>` to pin a run to one node, which is how you get off a node whose GPU
# has fallen off the bus - the packing scheduler is deterministic and will keep choosing it.
#
# Syncing the code is deliberately a separate step - the results PVC is shared, so a launch
# does not overwrite it behind your back:
#
#   oc cp . <pod>:/results/par
set -euo pipefail

: "${NAME:=}" ; : "${TAG:=}" ; : "${MODEL:=}"
: "${GPUS:=4}" ; : "${MEM:=700Gi}" ; : "${TF4:=0}" ; : "${TFVER:=4.57.1}"
: "${BATCH:=4}" ; : "${NWIN:=32}" ; : "${KEEPS:=6,4,2}"
: "${STREAM:=multilingual}" ; : "${TOKREF:=/models/qwen_tok}" ; : "${NODE:=}"
: "${N_REQ:=256}" ; : "${PROMPT_LEN:=1024}" ; : "${GEN_LEN:=128}" ; : "${REPEATS:=6}"
: "${ARMS:=}" ; : "${IMG:=v0.31.0}" ; : "${A2A:=}" ; : "${SERVE:=0}" ; : "${DP:=2}" ; : "${REPS:=1}" ; : "${REP0:=0}"

for kv in "$@"; do export "${kv?}"; done
[ -n "${NAME}" ] && [ -n "${TAG}" ] && [ -n "${MODEL}" ] || {
  echo "need NAME, TAG, MODEL" >&2; exit 2; }

# `nodeSelector: {${NODE}}`, so the pin is a whole `key: value` pair, not just a hostname.
NODE="${NODE:+kubernetes.io/hostname: ${NODE}}"
export NAME TAG MODEL GPUS MEM TF4 TFVER BATCH NWIN KEEPS STREAM TOKREF NODE ARMS IMG A2A SERVE DP REPS REP0
export N_REQ PROMPT_LEN GEN_LEN REPEATS
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
oc delete job "par-${NAME}" --ignore-not-found >/dev/null
envsubst '$NAME $TAG $MODEL $GPUS $MEM $TF4 $TFVER $BATCH $NWIN $KEEPS $STREAM $TOKREF $NODE $ARMS $IMG $A2A $SERVE $DP $REPS $REP0 $N_REQ $PROMPT_LEN $GEN_LEN $REPEATS' \
  < "${here}/job.yaml" | oc apply -f -
echo "launched par-${NAME}: ${TAG} ${MODEL} gpus=${GPUS} tf4=${TF4} arms=${ARMS:-offline} a2a=${A2A:-default} serve=${SERVE} dp=${DP} reps=${REPS} rep0=${REP0} shape=${N_REQ}x${PROMPT_LEN}/${GEN_LEN}"
