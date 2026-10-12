#!/usr/bin/env bash
# Render and apply `job.yaml` for one model.
#
#   ./launch.sh NAME=q30 TAG=q30rh MODEL=/models/bf16
#   ./launch.sh NAME=q35 TAG=q35   MODEL=/models/q35_bf16
#   ./launch.sh NAME=hy  TAG=hy13b MODEL=/models/hy_bf16 TF4=1 BATCH=1
#   ./launch.sh NAME=gm  TAG=gm26b MODEL=/models/gm_bf16
#
# `ARMS=base,base2,6,4,2` switches the job to the real vLLM measurement, one engine load per
# arm; the offline three-phase run is what you get without it.
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
: "${ARMS:=}" ; : "${IMG:=v0.31.0}"

for kv in "$@"; do export "${kv?}"; done
[ -n "${NAME}" ] && [ -n "${TAG}" ] && [ -n "${MODEL}" ] || {
  echo "need NAME, TAG, MODEL" >&2; exit 2; }

# `nodeSelector: {${NODE}}`, so the pin is a whole `key: value` pair, not just a hostname.
NODE="${NODE:+kubernetes.io/hostname: ${NODE}}"
export NAME TAG MODEL GPUS MEM TF4 TFVER BATCH NWIN KEEPS STREAM TOKREF NODE ARMS IMG
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
oc delete job "par-${NAME}" --ignore-not-found >/dev/null
envsubst '$NAME $TAG $MODEL $GPUS $MEM $TF4 $TFVER $BATCH $NWIN $KEEPS $STREAM $TOKREF $NODE $ARMS $IMG' \
  < "${here}/job.yaml" | oc apply -f -
echo "launched par-${NAME}: ${TAG} ${MODEL} gpus=${GPUS} tf4=${TF4} arms=${ARMS:-offline}"
