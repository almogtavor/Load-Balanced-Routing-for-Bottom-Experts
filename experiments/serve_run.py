"""One arm of the DeepEP measurement: a real `vllm serve`, loaded over HTTP.

The offline `LLM` API cannot reach the backend this kernel exists for. vLLM 0.31 builds
all-to-all kernels only when expert parallelism is paired with data or prefill-context parallelism:

    use_all2all_kernels = use_ep and (dp_size > 1 or pcp_size > 1 or is_sequence_parallel)
      -- vllm/model_executor/layers/fused_moe/config.py

With plain `tensor_parallel_size=4` the default `allgather_reducescatter` gathers every token
onto every rank, so dispatch volume is independent of per-token rank fan-out and there is
nothing for the rule to save. Of the three gates, `pcp_size` does engage DeepEP and then dies
in attention selection (`No valid attention backend found ... PCP not supported`), and
`is_sequence_parallel` itself requires data parallelism - so `dp_size > 1` is the only way in,
and the offline `LLM` refuses it ("not supported for single-process usage and may hang").

Hence a server. `vllm serve --data-parallel-size 2 --tensor-parallel-size 2
--enable-expert-parallel --all2all-backend deepep_high_throughput` is one process per rank,
expert-parallel over all four GPUs, dispatching through DeepEP's token-sending path - which is
both the configuration a real deployment uses and the one whose volume is `sum_t d(t)`, the
quantity the rule shrinks.

Throughput only. Accuracy is a property of the routing, which `accuracy.py` measures item by
item on the same models; the all-to-all backend moves the same selections.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import pathlib
import signal
import subprocess
import time
import urllib.error
import urllib.request

import numpy as np

MODEL = os.environ["MODEL"]
TAG = os.environ["TAG"]
ARM = os.environ.get("ARM", "base")                   # base* | `keep` | `keep`p`load_penalty`
REP = os.environ.get("REP", "0")                      # which sweep; a launch is the unit
N_GPUS = int(os.environ.get("GPUS", 4))
DP = int(os.environ.get("DP", 2))
# Expert parallelism spread over more than one node. DeepEP charges NVLink per distinct rank
# but RDMA per distinct *node*, and inside a single node the top `keep` experts alone already
# reach every node there is - so the fan-out the rule removes is all intra-node, and the
# expensive tier barely moves. Only once the node count exceeds `keep` can the tail be
# confined, which needs one pod per node: an indexed Job, rank 0 driving, the rest headless.
NODES = int(os.environ.get("NODES", 1))
RANK = int(os.environ.get("JOB_COMPLETION_INDEX", 0))
HEAD = os.environ.get("HEAD", "127.0.0.1")
RPC_PORT = int(os.environ.get("RPC_PORT", 5555))
# `or`, not a `get` default: the job template exports every knob it knows about, so an
# unset `A2A` arrives as the empty string and would reach vLLM as an invalid choice.
A2A = os.environ.get("A2A") or "deepep_high_throughput"
N_REQ = int(os.environ.get("N_REQ", 256))
PROMPT_LEN = int(os.environ.get("PROMPT_LEN", 1024))
GEN_LEN = int(os.environ.get("GEN_LEN", 128))
# Derived, so a prefill-heavy shape does not also need its own context knob passed to the
# job template: a prompt longer than the context window is rejected, not truncated.
MAX_LEN = int(os.environ.get("MAX_LEN") or 0) or PROMPT_LEN + GEN_LEN + 8
# The scheduler's token budget per step, which *is* the dispatch payload per all-to-all. Under
# data parallelism vLLM defaults it to 256, not the usual 2048
# (`DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP`, applied in arg_utils.py when the flag is
# unset), so every earlier run here chunked a long prompt into 256-token steps: `V` per
# collective was pinned small however many requests were in flight, which is exactly the
# `L`-dominated regime where a rule that only shrinks `V` cannot show. It is a measurement
# condition, so it is set explicitly rather than inherited, and recorded in the artefact.
BATCH_TOK = int(os.environ.get("BATCH_TOK") or 0) or max(MAX_LEN, 2048)
REPEATS = int(os.environ.get("REPEATS", 6))
PORT = int(os.environ.get("PORT", 8077))
# Engine load with DeepEP buffers and no CUDA graphs is slow, and a job that silently waits
# forever on a server that died is worse than one that fails.
BOOT_S = int(os.environ.get("BOOT_S", 2400))
URL = f"http://127.0.0.1:{PORT}"
OUT = pathlib.Path("runs")
assert N_GPUS % DP == 0, f"GPUS={N_GPUS} not divisible by DP={DP}"


def cmdline() -> list[str]:
    """The engine command. `DP` is per pod, so the engine's dp size is `DP * NODES` and the
    expert-parallel world is the whole Job - every rank of it, head and peers alike."""
    cmd = ["vllm", "serve", MODEL, "--port", str(PORT),
           "--data-parallel-size", str(DP * NODES),
           "--data-parallel-size-local", str(DP),
           "--tensor-parallel-size", str(N_GPUS // DP),
           "--enable-expert-parallel", "--all2all-backend", A2A,
           "--trust-remote-code", "--max-model-len", str(MAX_LEN), "--seed", "0",
           "--max-num-batched-tokens", str(BATCH_TOK),
           "--no-enable-prefix-caching", "--disable-log-stats",
           "--gpu-memory-utilization", os.environ.get("GPU_UTIL", "0.90")]
    if NODES > 1:
        cmd += ["--data-parallel-address", HEAD, "--data-parallel-rpc-port", str(RPC_PORT)]
        if RANK:
            cmd += ["--headless", "--data-parallel-start-rank", str(RANK * DP)]
    return cmd


def peer() -> None:
    """A non-head pod. Holds its expert shards and routes its own tokens - so it needs the
    same `PLACEMENT_KEEP` - but measures nothing. Exits when the head kills its engine between
    arms, and `run.sh` restarts it for the next one."""
    cmd = cmdline()
    print("[peer] " + " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


def serve() -> subprocess.Popen:
    """Start the server and block until it answers `/health`, or raise with its last output."""
    cmd = cmdline()
    print("[serve] " + " ".join(cmd), flush=True)
    # Its own process group, so one kill reaches the data-parallel children too.
    p = subprocess.Popen(cmd, start_new_session=True)
    for _ in range(BOOT_S // 5):
        if p.poll() is not None:
            raise SystemExit(f"[serve] exited {p.returncode} before serving")
        try:
            with urllib.request.urlopen(URL + "/health", timeout=5) as r:
                if r.status == 200:
                    print("[serve] up", flush=True)
                    return p
        except (urllib.error.URLError, OSError):
            time.sleep(5)
    kill(p)
    raise SystemExit(f"[serve] not healthy after {BOOT_S}s")


def kill(p: subprocess.Popen) -> None:
    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    try:
        p.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)


def post(prompt: list[int]) -> int:
    """One completion. Returns the number of tokens generated, which is checked, not trusted."""
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": GEN_LEN,
                       "temperature": 0, "ignore_eos": True}).encode()
    req = urllib.request.Request(URL + "/v1/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["usage"]["completion_tokens"]


# Every other number in this file is a time, and a time also moves when a launch is slow -
# which is exactly the confound the keep==top_k control arm exists to absorb. These counters
# are the one direct read of the quantity the routing rule actually changes. `port_xmit_data`
# counts octets/4 per the IB spec, and is per *host*, so it includes the engine's own NCCL
# traffic - the comparison that means anything is arm against arm within one launch.
IB = sorted(pathlib.Path("/sys/class/infiniband").glob("*/ports/1/counters/port_xmit_data"))


def ib_bytes() -> int:
    """Bytes transmitted across every IB port this pod can see, or 0 where there are none."""
    tot = 0
    for c in IB:
        try:
            tot += int(c.read_text()) * 4
        except (OSError, ValueError):
            pass
    return tot


def throughput(prompts: list[list[int]]) -> list[dict]:
    """Wall-clock and bytes for a fixed amount of work, all requests in flight at once."""
    out = []
    with cf.ThreadPoolExecutor(max_workers=N_REQ) as ex:
        for r in range(REPEATS):
            b0, t0 = ib_bytes(), time.perf_counter()
            n_out = sum(ex.map(post, prompts))
            dt, db = time.perf_counter() - t0, ib_bytes() - b0
            assert n_out == N_REQ * GEN_LEN, f"decoded {n_out}, expected {N_REQ * GEN_LEN}"
            n_tot = n_out + N_REQ * PROMPT_LEN
            out.append({"repeat": r, "seconds": dt, "out_tok_per_s": n_out / dt,
                        "tot_tok_per_s": n_tot / dt,
                        # The 32-bit counters wrap; a negative delta is a wrap, not a saving.
                        "ib_gb": db / 1e9 if db >= 0 else None,
                        "ib_b_per_tok": db / n_tot if db >= 0 else None})
            print(f"  [{ARM}] repeat {r}: {dt:.2f}s  {n_out / dt:.1f} out-tok/s  "
                  f"{n_tot / dt:.0f} tot-tok/s  {db / 1e9:.2f} GB IB", flush=True)
    return out


def main() -> None:
    # `4` is keep=4; `4p0.5` is keep=4 with the load penalty. Both in the arm name so a sweep
    # needs no extra variable to carry them, and one filename still identifies the run.
    k, _, a = ("", "", "") if ARM.startswith("base") else ARM.partition("p")
    keep = None if k == "" else int(k)
    load_penalty = float(a) if a else 0.0
    if keep is None:
        os.environ.pop("PLACEMENT_KEEP", None)
        os.environ.pop("PLACEMENT_LOAD_PENALTY", None)
    else:
        os.environ["PLACEMENT_KEEP"] = str(keep)
        os.environ["PLACEMENT_LOAD_PENALTY"] = str(load_penalty)
    print(f"[{TAG}/{ARM}] PLACEMENT_KEEP={os.environ.get('PLACEMENT_KEEP', '<unset>')} "
          f"PLACEMENT_LOAD_PENALTY={os.environ.get('PLACEMENT_LOAD_PENALTY', '<unset>')} gpus={N_GPUS} "
          f"dp={DP * NODES} (local {DP}) tp={N_GPUS // DP} nodes={NODES} rank={RANK} "
          f"a2a={A2A}", flush=True)
    # Rides along on every arm of every launch: two CUDA events per MoE forward, the same on
    # the base arms, and the only engine-level clock on the layer this rule actually moves.
    os.environ.setdefault("PLACEMENT_PROBE",
                          str(pathlib.Path(__file__).with_name("moetime.py")))
    if RANK:
        peer()

    # Fixed token ids rather than text, seeded: the batch has to
    # be identical across arms, and a prompt that tokenizes differently would not be.
    rng = np.random.default_rng(0)
    prompts = rng.integers(1, 100000, size=(N_REQ, PROMPT_LEN)).tolist()

    p = serve()
    try:
        tp = throughput(prompts)
    finally:
        kill(p)
    # Early repeats pay for CUDA graph capture and the first all-to-all handshake, which is a
    # launch cost, not a routing one. How many is not fixed: at a 4096-token prompt the second
    # repeat was still half rate, so the cut is "slower than 80% of the fastest repeat" rather
    # than a hard `tp[1:]`. The headline is a median and survived either way; the decay
    # diagnostic beside it did not, and read -40% when a cold repeat stayed in the set.
    top = max(r["out_tok_per_s"] for r in tp)
    warm = [r for r in tp if r["out_tok_per_s"] >= 0.8 * top] or tp[-1:]
    med = lambda k: float(np.median(v)) if (v := [r[k] for r in warm if r[k] is not None]) else None

    OUT.mkdir(exist_ok=True)
    (OUT / f"serve_{TAG}_{ARM}_r{REP}.json").write_text(json.dumps({
        # A result carries no clue which kernel wrote it, and `compare.py` globs arms by
        # tag - so without this a file left over from a superseded run aggregates
        # silently beside a fresh one. A missing field fails loudly; a stale number does not.
        "rep": int(REP),
        "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tag": TAG, "model": MODEL, "arm": ARM, "keep": keep, "load_penalty": load_penalty,
        "n_gpus": N_GPUS,
        "vllm": __import__("vllm").__version__, "a2a": A2A, "dp": DP * NODES, "tp": N_GPUS // DP,
        "nodes": NODES, "ep_size": N_GPUS * NODES,
        "n_req": N_REQ, "prompt_len": PROMPT_LEN, "gen_len": GEN_LEN,
        "batch_tok": BATCH_TOK, "max_len": MAX_LEN,
        "throughput": tp, "warm_tok_per_s": [r["out_tok_per_s"] for r in warm],
        "out_tok_per_s": med("out_tok_per_s"),
        # Prefill is `PROMPT_LEN / GEN_LEN` times the token volume of decode but only enters
        # `out_tok_per_s` through the denominator, and prefill is where the dispatch payload
        # is large enough for `V / B_eff` to outweigh the fixed `L` - so at a prefill-heavy
        # shape this, not the decode rate, is the number the rule is predicted to move.
        "tot_tok_per_s": med("tot_tok_per_s"),
        "ib_b_per_tok": med("ib_b_per_tok"),
    }, indent=1) + "\n")
    print(f"WROTE {OUT}/serve_{TAG}_{ARM}_r{REP}.json", flush=True)


if __name__ == "__main__":
    main()
