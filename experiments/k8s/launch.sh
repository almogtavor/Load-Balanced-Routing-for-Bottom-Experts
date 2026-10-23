#!/usr/bin/env bash
# Render and apply `job.yaml` for one model.
#
#   ./launch.sh NAME=q30 TAG=q30rh MODEL=/models/bf16
#   ./launch.sh NAME=q35 TAG=q35   MODEL=/models/q35_bf16
#   ./launch.sh NAME=gm  TAG=gm26b MODEL=/models/gm_bf16
#
# `SETTINGS=base,base2,keep4-penalty8-any-rank` switches the job to the in-engine measurement, one
# `vllm serve` per setting (names as in `serve_run.setting_env`); without it the job runs the offline paired-NLL pass. `DP=4
# A2A=deepep_high_throughput` is the only shape where the volume this kernel shrinks is the
# volume the engine moves - vLLM's default all-gathers, and 0.31 gates the all-to-all kernels
# on data parallelism. Give those runs their own TAG; the backend is a measurement condition.
#
# Add `NODE=<hostname>` to pin a run to one node, which is how you get off a node whose GPU
# has fallen off the bus - the packing scheduler is deterministic and will keep choosing it.
#
# Syncing the code is deliberately a separate step - the results volume is shared, so a launch
# does not overwrite it behind your back: copy this repository to `/results/placement-router`
# through any pod that mounts the claim (`oc cp . <pod>:/results/placement-router`).
set -euo pipefail

: "${NAME:=}" ; : "${TAG:=}" ; : "${MODEL:=}"
: "${GPUS:=4}" ; : "${MEM:=700Gi}"
: "${BATCH:=4}" ; : "${NWIN:=32}" ; : "${KEEPS:=6,4,2}"
: "${STREAM:=multilingual}" ; : "${TOKREF:=/models/qwen_tok}" ; : "${NODE:=}"
: "${N_REQ:=256}" ; : "${PROMPT_LEN:=1024}" ; : "${GEN_LEN:=128}" ; : "${REPEATS:=6}"
: "${GPU_UTIL:=0.90}"
: "${BATCH_TOK:=0}"   # 0 = serve_run picks max(MAX_LEN, 2048); vLLM's own DP default is 256
: "${WARMUP:=0}" ; : "${EPLB_STEPS:=100}" ; : "${EPLB_REDUNDANT:=16}"
: "${EVAL_TASKS:=}" ; : "${EVAL_CHAT:=}" ; : "${EVAL_GEN_KWARGS:=}" ; : "${MAX_LEN:=}" ; : "${CHAT:=}"
: "${PENS:=0,1}" ; : "${SETTINGS:=}" ; : "${SCRIPT:=}" ; : "${IMG:=v0.31.0}" ; : "${A2A:=}" ; : "${DP:=2}" ; : "${REPS:=1}" ; : "${REP0:=0}"

for kv in "$@"; do export "${kv?}"; done
[ -n "${NAME}" ] && [ -n "${TAG}" ] && [ -n "${MODEL}" ] || {
  echo "need NAME, TAG, MODEL" >&2; exit 2; }

# `nodeSelector: {${NODE}}`, so the pin is a whole `key: value` pair, not just a hostname.
NODE="${NODE:+kubernetes.io/hostname: ${NODE}}"
export NAME TAG MODEL GPUS MEM BATCH NWIN KEEPS STREAM TOKREF NODE PENS SETTINGS SCRIPT IMG A2A DP REPS REP0
export N_REQ PROMPT_LEN GEN_LEN REPEATS BATCH_TOK GPU_UTIL WARMUP EPLB_STEPS EPLB_REDUNDANT EVAL_TASKS EVAL_CHAT EVAL_GEN_KWARGS MAX_LEN CHAT
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
oc delete job "placement-${NAME}" --ignore-not-found >/dev/null
envsubst '$NAME $TAG $MODEL $GPUS $MEM $BATCH $NWIN $KEEPS $STREAM $TOKREF $NODE $PENS $SETTINGS $SCRIPT $IMG $A2A $DP $REPS $REP0 $N_REQ $PROMPT_LEN $GEN_LEN $REPEATS $BATCH_TOK $GPU_UTIL $WARMUP $EPLB_STEPS $EPLB_REDUNDANT $EVAL_TASKS $EVAL_CHAT $EVAL_GEN_KWARGS $MAX_LEN $CHAT' \
  < "${here}/job.yaml" | oc apply -f -
echo "launched placement-${NAME}: ${TAG} ${MODEL} gpus=${GPUS} settings=${SETTINGS:-offline} a2a=${A2A:-default} dp=${DP} reps=${REPS} rep0=${REP0} shape=${N_REQ}x${PROMPT_LEN}/${GEN_LEN} btok=${BATCH_TOK}"
