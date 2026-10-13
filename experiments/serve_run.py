"""One arm of the DeepEP measurement: a real `vllm serve`, loaded over HTTP.

`vllm_run.py` cannot reach the backend this kernel exists for. vLLM 0.31 builds all-to-all
kernels only when expert parallelism is paired with data or prefill-context parallelism:

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

Throughput only. Accuracy is a property of the routing, which is what `vllm_run.py` measures
item by item on the same models; the all-to-all backend moves the same selections.
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
ARM = os.environ.get("ARM", "base")                   # base | base2 | an integer `keep`
N_GPUS = int(os.environ.get("GPUS", 4))
DP = int(os.environ.get("DP", 2))
# `or`, not a `get` default: the job template exports every knob it knows about, so an
# unset `A2A` arrives as the empty string and would reach vLLM as an invalid choice.
A2A = os.environ.get("A2A") or "deepep_high_throughput"
MAX_LEN = int(os.environ.get("MAX_LEN", 2048))
N_REQ = int(os.environ.get("N_REQ", 256))
PROMPT_LEN = int(os.environ.get("PROMPT_LEN", 1024))
GEN_LEN = int(os.environ.get("GEN_LEN", 128))
REPEATS = int(os.environ.get("REPEATS", 6))
PORT = int(os.environ.get("PORT", 8077))
# Engine load with DeepEP buffers and no CUDA graphs is slow, and a job that silently waits
# forever on a server that died is worse than one that fails.
BOOT_S = int(os.environ.get("BOOT_S", 2400))
URL = f"http://127.0.0.1:{PORT}"
OUT = pathlib.Path("runs")
assert N_GPUS % DP == 0, f"GPUS={N_GPUS} not divisible by DP={DP}"


def serve() -> subprocess.Popen:
    """Start the server and block until it answers `/health`, or raise with its last output."""
    cmd = ["vllm", "serve", MODEL, "--port", str(PORT),
           "--data-parallel-size", str(DP), "--tensor-parallel-size", str(N_GPUS // DP),
           "--enable-expert-parallel", "--all2all-backend", A2A,
           "--trust-remote-code", "--max-model-len", str(MAX_LEN), "--seed", "0",
           "--no-enable-prefix-caching", "--disable-log-stats",
           "--gpu-memory-utilization", os.environ.get("GPU_UTIL", "0.90")]
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


def throughput(prompts: list[list[int]]) -> list[dict]:
    """Wall-clock for a fixed amount of decode work, all requests in flight at once."""
    out = []
    with cf.ThreadPoolExecutor(max_workers=N_REQ) as ex:
        for r in range(REPEATS):
            t0 = time.perf_counter()
            n_out = sum(ex.map(post, prompts))
            dt = time.perf_counter() - t0
            assert n_out == N_REQ * GEN_LEN, f"decoded {n_out}, expected {N_REQ * GEN_LEN}"
            out.append({"repeat": r, "seconds": dt, "out_tok_per_s": n_out / dt,
                        "tot_tok_per_s": (n_out + N_REQ * PROMPT_LEN) / dt})
            print(f"  [{ARM}] repeat {r}: {dt:.2f}s  {n_out / dt:.1f} out-tok/s", flush=True)
    return out


def main() -> None:
    keep = None if ARM in ("base", "base2") else int(ARM)
    if keep is None:
        os.environ.pop("PAR_KEEP", None)
    else:
        os.environ["PAR_KEEP"] = str(keep)
    print(f"[{TAG}/{ARM}] PAR_KEEP={os.environ.get('PAR_KEEP', '<unset>')} gpus={N_GPUS} "
          f"dp={DP} tp={N_GPUS // DP} a2a={A2A}", flush=True)

    # Fixed token ids rather than text, with the same seed as `vllm_run.py`: the batch has to
    # be identical across arms, and a prompt that tokenizes differently would not be.
    rng = np.random.default_rng(0)
    prompts = rng.integers(1, 100000, size=(N_REQ, PROMPT_LEN)).tolist()

    p = serve()
    try:
        tp = throughput(prompts)
    finally:
        kill(p)
    warm = [r["out_tok_per_s"] for r in tp[1:]] or [tp[0]["out_tok_per_s"]]

    OUT.mkdir(exist_ok=True)
    (OUT / f"serve_{TAG}_{ARM}.json").write_text(json.dumps({
        # A result carries no clue which kernel wrote it, and `compare.py` globs arms by
        # tag - so without this a file left over from a superseded run aggregates
        # silently beside a fresh one. A missing field fails loudly; a stale number does not.
        "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tag": TAG, "model": MODEL, "arm": ARM, "keep": keep, "n_gpus": N_GPUS,
        "vllm": __import__("vllm").__version__, "a2a": A2A, "dp": DP, "tp": N_GPUS // DP,
        "n_req": N_REQ, "prompt_len": PROMPT_LEN, "gen_len": GEN_LEN,
        "throughput": tp, "warm_tok_per_s": warm,
        "out_tok_per_s": float(np.median(warm)),
    }, indent=1) + "\n")
    print(f"WROTE {OUT}/serve_{TAG}_{ARM}.json", flush=True)


if __name__ == "__main__":
    main()
