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
# `NODES=4 GPUS=2 DP=2` is one pod per node, ep_size = DP * NODES = 8, spread over 4 hosts.
# That is the shape this kernel needs to pay: DeepEP charges RDMA per distinct *node*, and
# inside one node the top `keep` experts already reach every node, so a fan-out saving there
# lands entirely on NVLink. There is no threshold, only saturation: the tail can be confined
# once a tier has more destinations than the top `keep` already reach, which at 2 nodes it
# does not and at 8 it does.
#
# Syncing the code is deliberately a separate step - the results PVC is shared, so a launch
# does not overwrite it behind your back:
#
#   oc cp . <pod>:/results/par
#
# On CoreWeave the same step is `kubectl cp`, and the claim carries the weights too:
#
#   KCTL='kubectl --context=waldorf_US-EAST-04A -n almogtavor-dev' TPL=job-cw.yaml \
#     NS=almogtavor-dev CLAIM=par-cw BAD=gd91fda ./launch.sh NAME=q30mn TAG=q30mn \
#     MODEL=/results/models/q30 NODES=4 GPUS=1 DP=1 MEM=200Gi \
#     SERVE=1 A2A=nixl_ep ARMS=base,base2,4,base3
#
# `nixl_ep`, not `deepep_high_throughput`, at one GPU per node: DeepEP computes `nvl_rank =
# rank % 8` and `rdma_rank = rank / 8` unconditionally, so its only legal shapes are one node
# with ep <= 8 or several nodes with exactly 8 GPUs each. One GPU per node asserts in HT and
# dies in `cudaIpcOpenMemHandle` in LL. `nixl_ep` has the same dispatch contract and no such
# constraint, which makes it the only backend that runs this shape.
#
# `BAD` is per-cluster, so pass it: the default names the two unusable OpenShift hosts. Four
# arms rather than six because only `keep=4` has a predicted RDMA saving at ep=4, and three
# base launches measure the launch-to-launch spread it has to clear.
set -euo pipefail

: "${NAME:=}" ; : "${TAG:=}" ; : "${MODEL:=}"
: "${GPUS:=4}" ; : "${MEM:=700Gi}" ; : "${TF4:=0}" ; : "${TFVER:=4.57.1}"
: "${BATCH:=4}" ; : "${NWIN:=32}" ; : "${KEEPS:=6,4,2}"
: "${STREAM:=multilingual}" ; : "${TOKREF:=/models/qwen_tok}" ; : "${NODE:=}"
: "${N_REQ:=256}" ; : "${PROMPT_LEN:=1024}" ; : "${GEN_LEN:=128}" ; : "${REPEATS:=6}"
: "${BATCH_TOK:=0}"   # 0 = serve_run picks max(MAX_LEN, 2048); vLLM's own DP default is 256
: "${NODES:=1}" ; : "${BAD:=pokprod-b93r43s1, pokprod-b93r38s0}"
: "${ARMS:=}" ; : "${IMG:=v0.31.0}" ; : "${A2A:=}" ; : "${SERVE:=0}" ; : "${DP:=2}" ; : "${REPS:=1}" ; : "${REP0:=0}"
# Which cluster. The defaults are the OpenShift one, whose template carries its namespace and
# model PVCs as literals; `TPL=job-cw.yaml` is the CoreWeave multi-node shape, which takes
# both as variables because neither pre-exists there.
: "${KCTL:=oc}" ; : "${TPL:=job.yaml}"
: "${NS:=llm-d-pic}" ; : "${CLAIM:=kvcache-bench-results-almogtavor}"

for kv in "$@"; do export "${kv?}"; done
[ -n "${NAME}" ] && [ -n "${TAG}" ] && [ -n "${MODEL}" ] || {
  echo "need NAME, TAG, MODEL" >&2; exit 2; }

# `nodeSelector: {${NODE}}`, so the pin is a whole `key: value` pair, not just a hostname.
NODE="${NODE:+kubernetes.io/hostname: ${NODE}}"
export NAME TAG MODEL GPUS MEM TF4 TFVER BATCH NWIN KEEPS STREAM TOKREF NODE ARMS IMG A2A SERVE DP REPS REP0
export NODES BAD NS CLAIM
export N_REQ PROMPT_LEN GEN_LEN REPEATS BATCH_TOK
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
${KCTL} delete job "par-${NAME}" --ignore-not-found >/dev/null
envsubst '$NAME $TAG $MODEL $GPUS $MEM $TF4 $TFVER $BATCH $NWIN $KEEPS $STREAM $TOKREF $NODE $ARMS $IMG $A2A $SERVE $DP $REPS $REP0 $N_REQ $PROMPT_LEN $GEN_LEN $REPEATS $BATCH_TOK $NODES $BAD $NS $CLAIM' \
  < "${here}/${TPL}" | ${KCTL} apply -f -
echo "launched par-${NAME}: ${TAG} ${MODEL} gpus=${GPUS} tf4=${TF4} arms=${ARMS:-offline} a2a=${A2A:-default} serve=${SERVE} dp=${DP} reps=${REPS} rep0=${REP0} shape=${N_REQ}x${PROMPT_LEN}/${GEN_LEN} btok=${BATCH_TOK} nodes=${NODES} ep=$(( DP * NODES ))"
