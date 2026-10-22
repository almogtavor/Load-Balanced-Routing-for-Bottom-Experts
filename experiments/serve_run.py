"""One setting of the DeepEP measurement: a real `vllm serve`, loaded over HTTP.

The offline `LLM` API cannot reach the backend this kernel exists for. vLLM 0.31 builds
all-to-all kernels only when expert parallelism is paired with data or prefill-context parallelism:

    use_all2all_kernels = use_ep and (dp_size > 1 or pcp_size > 1 or is_sequence_parallel)
      (vllm/model_executor/layers/fused_moe/config.py)

With plain `tensor_parallel_size=4` the default `allgather_reducescatter` gathers every token
onto every rank, so dispatch volume is independent of per-token rank fan-out and there is
nothing for the rule to save. Of the three gates, `pcp_size` does engage DeepEP and then dies
in attention selection (`No valid attention backend found ... PCP not supported`), and
`is_sequence_parallel` itself requires data parallelism - so `dp_size > 1` is the only way in,
and the offline `LLM` refuses it ("not supported for single-process usage and may hang").

Hence a server. `vllm serve --data-parallel-size 4 --enable-expert-parallel --all2all-backend
deepep_high_throughput` is one process per rank, expert-parallel over all four GPUs, dispatching through DeepEP's token-sending path - which is
both the configuration a real deployment uses and the one whose volume is `sum_t d(t)`, the
quantity the rule shrinks.

The load it serves is what `moetime.py`, riding along via `PLACEMENT_PROBE`, times.
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
SETTING = os.environ.get("SETTING", "base")           # base*, eplb, or a name `setting_env` parses
REP = os.environ.get("REP", "0")                      # which sweep; a launch is the unit
N_GPUS = int(os.environ.get("GPUS", 4))
DP = int(os.environ.get("DP", 2))
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
WARMUP = int(os.environ.get("WARMUP") or 0)          # untimed repeats before the timed ones
EPLB_STEPS = int(os.environ.get("EPLB_STEPS") or 100)
PORT = int(os.environ.get("PORT", 8077))
# Engine load with DeepEP buffers and no CUDA graphs is slow, and a job that silently waits
# forever on a server that died is worse than one that fails.
BOOT_S = int(os.environ.get("BOOT_S", 2400))
URL = f"http://127.0.0.1:{PORT}"
OUT = pathlib.Path("runs")
assert N_GPUS % DP == 0, f"GPUS={N_GPUS} not divisible by DP={DP}"


def cmdline() -> list[str]:
    cmd = ["vllm", "serve", MODEL, "--port", str(PORT),
           "--data-parallel-size", str(DP),
           "--tensor-parallel-size", str(N_GPUS // DP),
           "--enable-expert-parallel", "--all2all-backend", A2A,
           "--trust-remote-code", "--max-model-len", str(MAX_LEN), "--seed", "0",
           "--max-num-batched-tokens", str(BATCH_TOK),
           "--no-enable-prefix-caching", "--disable-log-stats",
           "--gpu-memory-utilization", os.environ.get("GPU_UTIL", "0.90")]
    # `eplb`: default routing with vLLM's EPLB, the baseline that balances by replicating and
    # moving experts instead of rerouting tokens. A short window and interval, so it rebalances
    # during the untimed warm-up repeats of a run this short.
    # `-with-eplb`: a routing setting stacked on the same EPLB.
    if SETTING.startswith("eplb") or SETTING.endswith("-with-eplb"):
        cmd += ["--enable-eplb", "--eplb-config", json.dumps({
            "window_size": EPLB_STEPS, "step_interval": EPLB_STEPS, "log_balancedness": True,
            "num_redundant_experts": int(os.environ.get("EPLB_REDUNDANT", 16))})]
        os.environ["PLACEMENT_EPLB"] = "1"  # the plugin wires Gemma-4's MoE layers into EPLB
    return cmd


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
    pg = os.getpgid(p.pid)
    os.killpg(pg, signal.SIGTERM)
    try:
        p.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(pg, signal.SIGKILL)
    # The data-parallel engines outlive the API process and hold their GPU memory until they
    # exit, so the next setting's server, started seconds later, refused to boot ("Free memory on
    # device ... less than desired GPU memory utilization"). Wait for the whole group.
    for _ in range(120):
        try:
            os.killpg(pg, 0)
        except ProcessLookupError:
            return
        time.sleep(1)
    os.killpg(pg, signal.SIGKILL)


def post(prompt: list[int]) -> int:
    """One completion. Returns the number of tokens generated, which is checked, not trusted."""
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": GEN_LEN,
                       "temperature": 0, "ignore_eos": True}).encode()
    req = urllib.request.Request(URL + "/v1/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["usage"]["completion_tokens"]


def throughput(prompts: list[list[int]], repeats: int = REPEATS) -> list[dict]:
    """Wall-clock for a fixed amount of work, all requests in flight at once."""
    out = []
    with cf.ThreadPoolExecutor(max_workers=N_REQ) as ex:
        for r in range(repeats):
            t0 = time.perf_counter()
            n_out = sum(ex.map(post, prompts))
            dt = time.perf_counter() - t0
            assert n_out == N_REQ * GEN_LEN, f"decoded {n_out}, expected {N_REQ * GEN_LEN}"
            n_tot = n_out + N_REQ * PROMPT_LEN
            out.append({"repeat": r, "seconds": dt, "out_tok_per_s": n_out / dt,
                        "tot_tok_per_s": n_tot / dt})
            print(f"  [{SETTING}] repeat {r}: {dt:.2f}s  {n_out / dt:.1f} out-tok/s  "
                  f"{n_tot / dt:.0f} tot-tok/s", flush=True)
    return out


def setting_env() -> tuple[int | None, float, float]:
    """Set the plugin's env for `SETTING`; returns `(keep, load_penalty, mass)`, keep None for base.

    `base*` and `eplb` are default routing. Otherwise the name is
    `keep<k>[-fullmass<m>|-mass<m>][-penalty<p>][-sampled[-tau<t>]][-from-top<r>][-prefill-only]`
    `[-any-rank][-coact[-loaded]-slack<x>][-with-eplb]`, e.g. `keep4-penalty8` or
    `keep1-mass0.6-penalty8-any-rank`, so one filename still identifies the run.
    """
    import re

    if SETTING.startswith(("base", "eplb")):
        for v in ("PLACEMENT_KEEP", "PLACEMENT_LOAD_PENALTY", "PLACEMENT_MASS", "PLACEMENT_MASS_NORM",
                  "PLACEMENT_TAU", "PLACEMENT_RANGE", "PLACEMENT_PREFILL_ONLY", "PLACEMENT_ANY_RANK",
                  "PLACEMENT_COACT", "PLACEMENT_COACT_LOAD"):
            os.environ.pop(v, None)
        return None, 0.0, 0.0
    s = SETTING.removesuffix("-with-eplb")  # EPLB itself is set on the command line
    # `-coact-slack<x>`: the tail's ranks from the co-activation placement built at that slack
    # (`coact_place.py`). `-coact-loaded-slack<x>`: the experts are also loaded in that placement.
    s, loaded, slack = s.partition("-coact-loaded-slack")
    if not loaded:
        s, _, slack = s.partition("-coact-slack")
    os.environ["PLACEMENT_COACT_LOAD"] = "1" if loaded else "0"
    os.environ["PLACEMENT_COACT"] = (f"runs/coact_{os.environ.get('COACT_TAG', 'mix')}-slack{slack}.npy"
                                     if slack else "")
    # `-any-rank`: the tail from every rank, picked by the load penalty alone.
    os.environ["PLACEMENT_ANY_RANK"] = "1" if "-any-rank" in s else "0"
    s = s.replace("-any-rank", "")
    # `-fullmass`: mass of the full softmax; `-mass`: of default top-k's renormalized weights.
    # `-sampled[-tau<t>]-from-top<r>`: Expert-Sample's sampled tail, the control.
    # `-prefill-only`: decode tokens keep default routing, as on a P/D decode node.
    k, c, m, a, g, r, f = re.fullmatch(
        r"keep(\d+)(?:-(fullmass|mass)([\d.]+))?(?:-penalty([\d.]+))?(?:-sampled(?:-tau([\d.]+))?)?"
        r"(?:-from-top(\d+))?(-prefill-only)?", s).groups()
    g = g or ("1" if "-sampled" in s else None)
    os.environ["PLACEMENT_KEEP"] = k
    os.environ["PLACEMENT_LOAD_PENALTY"] = a or "0"
    os.environ["PLACEMENT_MASS"] = m or "0"
    os.environ["PLACEMENT_MASS_NORM"] = "1" if c == "mass" else "0"
    os.environ["PLACEMENT_TAU"] = g or "0"
    os.environ["PLACEMENT_RANGE"] = r or "0"
    os.environ["PLACEMENT_PREFILL_ONLY"] = "1" if f else "0"
    return int(k), float(a or 0), float(m or 0)


def main() -> None:
    keep, load_penalty, mass = setting_env()
    print(f"[{TAG}/{SETTING}] PLACEMENT_KEEP={os.environ.get('PLACEMENT_KEEP', '<unset>')} "
          f"PLACEMENT_LOAD_PENALTY={os.environ.get('PLACEMENT_LOAD_PENALTY', '<unset>')} "
          f"PLACEMENT_MASS={os.environ.get('PLACEMENT_MASS', '<unset>')} gpus={N_GPUS} "
          f"dp={DP} tp={N_GPUS // DP} a2a={A2A}", flush=True)
    # Rides along on every setting of every launch: two CUDA events per MoE forward, the same on
    # the base settings, and the only engine-level clock on the layer this rule actually moves.
    os.environ.setdefault("PLACEMENT_PROBE",
                          str(pathlib.Path(__file__).with_name("moetime.py")))

    # Fixed token ids rather than text, seeded: the batch has to
    # be identical across settings, and a prompt that tokenizes differently would not be.
    rng = np.random.default_rng(0)
    prompts = rng.integers(1, 100000, size=(N_REQ, PROMPT_LEN)).tolist()

    p = serve()
    try:
        if WARMUP:
            throughput(prompts, WARMUP)
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
    med = lambda k: float(np.median([r[k] for r in warm]))

    OUT.mkdir(exist_ok=True)
    (OUT / f"serve_{TAG}_{SETTING}_r{REP}.json").write_text(json.dumps({
        # A result carries no clue which kernel wrote it, and `compare.py` globs settings by
        # tag - so without this a file left over from a superseded run aggregates
        # silently beside a fresh one. A missing field fails loudly; a stale number does not.
        "rep": int(REP),
        "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tag": TAG, "model": MODEL, "setting": SETTING, "keep": keep, "load_penalty": load_penalty, "mass": mass,
        "n_gpus": N_GPUS,
        "vllm": __import__("vllm").__version__, "a2a": A2A, "dp": DP, "tp": N_GPUS // DP,
        "ep_size": N_GPUS,
        "n_req": N_REQ, "prompt_len": PROMPT_LEN, "gen_len": GEN_LEN,
        "batch_tok": BATCH_TOK, "max_len": MAX_LEN,
        "throughput": tp, "warm_tok_per_s": [r["out_tok_per_s"] for r in warm],
        "out_tok_per_s": med("out_tok_per_s"),
        # Prefill is `PROMPT_LEN / GEN_LEN` times the token volume of decode but only enters
        # `out_tok_per_s` through the denominator, and prefill is where the dispatch payload
        # is large enough for `V / B_eff` to outweigh the fixed `L` - so at a prefill-heavy
        # shape this, not the decode rate, is the number the rule is predicted to move.
        "tot_tok_per_s": med("tot_tok_per_s"),
    }, indent=1) + "\n")
    print(f"WROTE {OUT}/serve_{TAG}_{SETTING}_r{REP}.json", flush=True)


def _check() -> None:
    """`MODEL=x TAG=x python experiments/serve_run.py --check`: the setting names parse as documented."""
    def env(name):
        global SETTING
        SETTING = name
        r = setting_env()
        return r, {k: os.environ.get("PLACEMENT_" + k) for k in
                   ("KEEP", "MASS_NORM", "TAU", "RANGE", "PREFILL_ONLY", "ANY_RANK", "COACT", "COACT_LOAD")}

    assert env("base")[0] == (None, 0.0, 0.0)
    r, e = env("keep1-mass0.6-penalty8-any-rank")
    assert r == (1, 8.0, 0.6) and e["MASS_NORM"] == "1" and e["ANY_RANK"] == "1", e
    assert env("keep1-fullmass0.7")[1]["MASS_NORM"] == "0"
    r, e = env("keep6-sampled-from-top32")
    assert (e["TAU"], e["RANGE"]) == ("1", "32"), e
    assert env("keep6-sampled-tau0.5-from-top32")[1]["TAU"] == "0.5"
    r, e = env("keep4-penalty8-prefill-only-with-eplb")
    assert r == (4, 8.0, 0.0) and e["PREFILL_ONLY"] == "1" and e["TAU"] == "0", e
    r, e = env("keep1-mass0.6-penalty8-coact-loaded-slack0.02")
    assert e["COACT"] == "runs/coact_mix-slack0.02.npy" and e["COACT_LOAD"] == "1", e
    print("serve_run: setting names ok")


if __name__ == "__main__":
    _check() if "--check" in __import__("sys").argv else main()
