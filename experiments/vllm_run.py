"""One arm of the real measurement: vLLM, expert-parallel over 4 GPUs, with the kernel on.

The offline sweep in `latency.py` replays recorded routing through a hand-built
expert-parallel layer, which is the only way to isolate the all-to-all - and it is still a
model of serving, not serving. This script is the other half: a stock vLLM engine, the stock
all-to-all, the stock expert GEMMs, with nothing swapped except `BaseRouter._select_experts`.
What it reports is therefore the benefit an actual deployment gets, paid for at an actual
accuracy, with no accounting left to argue about.

`PAR_KEEP` is read once, in each worker, when the engine loads, so one arm is one process -
hence one script per arm and `compare.py` to pair them up. Two things make the pairing fair:

* **identical work per arm.** The throughput phase decodes a fixed token count with
  `ignore_eos`, so no arm can look fast by stopping early, and the prompts are a fixed token
  sequence rather than text, so tokenizer differences cannot change the batch.
* **a repeat baseline.** `ARM=base2` is the stock arm run a second time, in its own process,
  on its own freshly loaded engine. Every number here is a wall-clock measurement on a shared
  cluster, and the within-arm spread across repeats bounds the noise of one engine while
  `base2` bounds the part that a whole engine load can move - which is the larger of the two
  and therefore the one a speedup has to clear.

The accuracy phase scores GPQA by option likelihood through `prompt_logprobs`, item for item
and normalized by option length exactly as `bench.py` does it, so the two are comparable and
`compare.py` can run the same exact McNemar test on the result.

The language-modelling phase scores held-out wikitext-2 through that same `prompt_logprobs`
and reports mean NLL per window - a corpus fetched over the Hub API rather than the offline
sweep's locally cached stream, so this runs wherever the image runs. It is the engine's own
answer to the question the offline sweep answers with an HF forward pass: does the rewritten
routing change what the model predicts. Paired by window index across arms, so the difference carries
a standard error rather than two means to eyeball.
"""

from __future__ import annotations

import json
import os
import pathlib
import time

import numpy as np

MODEL = os.environ["MODEL"]
TAG = os.environ["TAG"]
ARM = os.environ.get("ARM", "base")                   # base* | an integer `keep`
REP = os.environ.get("REP", "0")                      # which sweep; a launch is the unit
N_GPUS = int(os.environ.get("GPUS", 4))
MAX_LEN = int(os.environ.get("MAX_LEN", 2048))
# The throughput phase. The benefit is a bandwidth saving, so it only shows up once the
# dispatch is bandwidth-bound: that needs many tokens resident at once, which is what a long
# prompt and a wide batch buy.
N_REQ = int(os.environ.get("N_REQ", 256))
PROMPT_LEN = int(os.environ.get("PROMPT_LEN", 1024))
GEN_LEN = int(os.environ.get("GEN_LEN", 128))
# The first repeat pays for CUDA graph capture and the Triton autotune and is discarded; the
# rest are the sample. Engine load dominates the job, so a repeat is nearly free - and one
# warm number per arm would leave no within-arm spread to compare an arm gap against.
REPEATS = int(os.environ.get("REPEATS", 6))
# Which all-to-all the engine dispatches through. vLLM 0.31 defaults to
# `allgather_reducescatter`, which gathers every token onto every rank: its volume does not
# depend on per-token rank fan-out, so there is nothing for this kernel to save here and its
# extra launches are pure cost. That is the stock serving shape and the number this script
# reports. The DeepEP path, whose volume *is* `sum_t d(t)`, cannot be reached from the offline
# API at all - it is gated on data or prefill-context parallelism, offline data parallelism
# refuses to run in one process, and prefill-context parallelism has no attention backend in
# 0.31 - so it is measured against a real server in `serve_run.py`.
A2A = os.environ.get("A2A", "")
# The language-modelling phase: `N_WIN` windows of held-out natural text, each one token short
# of `MAX_LEN` because `prompt_logprobs` needs room for the one token `max_tokens=1` samples.
N_WIN = int(os.environ.get("NWIN", 32))
LM_SOURCE = os.environ.get("LM_SOURCE", "Salesforce/wikitext")
LM_CONFIG = os.environ.get("LM_CONFIG", "wikitext-2-raw-v1")
LM_SPLIT = os.environ.get("LM_SPLIT", "test")
OUT = pathlib.Path("runs")


def engine():
    from vllm import LLM

    # Prefix caching off on purpose: the accuracy phase scores four options that share a
    # context, and a cache hit changes which tokens get prompt log-probs computed at all.
    return LLM(model=MODEL, tensor_parallel_size=N_GPUS, enable_expert_parallel=True,
               trust_remote_code=True, max_model_len=MAX_LEN, seed=0,
               enable_prefix_caching=False,
               gpu_memory_utilization=float(os.environ.get("GPU_UTIL", 0.90)),
               **({"all2all_backend": A2A} if A2A else {}))


def gpqa(llm) -> np.ndarray:
    """`[n]` bool: did this arm rank the gold option first, item by item.

    All four options of all items go in one `generate` call - the engine schedules them - and
    each option's score is the mean log-prob of its own tokens, so an option is not preferred
    merely for being short. `max_tokens=1` because nothing is generated here; the prompt
    log-probs are the measurement.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    import bench

    items = bench.load_items(np.random.default_rng(0))
    tok = llm.get_tokenizer()
    prompts, spans = [], []
    for ctx, choices, _ in items:
        cid = tok(ctx).input_ids
        for c in choices:
            ch = tok(c, add_special_tokens=False).input_ids
            prompts.append(TokensPrompt(prompt_token_ids=cid + ch))
            spans.append(len(ch))
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0))
    score = []
    for o, k in zip(outs, spans):
        # `prompt_logprobs[i]` is the log-prob of prompt token `i` given everything before it,
        # and `[0]` is None because the first token is not predicted by anything.
        lp = [next(iter(d.values())).logprob for d in o.prompt_logprobs[-k:]]
        score.append(sum(lp) / k)
    s = np.array(score).reshape(len(items), 4)
    return s.argmax(1) == np.array([g for _, _, g in items])


def nll(llm) -> list[float]:
    """Mean NLL per window of held-out text, over `N_WIN` windows.

    `prompt_logprobs=0` returns, for each prompt token, the log-prob the model gave it in
    context, which is exactly the negative of the cross-entropy `accuracy.py` computes through
    an HF forward pass - so this is that same quantity, read out of the serving engine with
    the kernel in the loop. Position 0 has no log-prob because nothing predicts it.

    Wikitext-2 rather than the offline sweep's cached token stream: the stream is a local
    artefact and this has to run wherever the image runs, and a corpus that is not also the
    accuracy phase's corpus keeps the two measurements from sharing a sample. Windows are
    paired by index across arms, so the arm difference is a paired difference.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    import bench

    tok = llm.get_tokenizer()
    # No BOS prepended, although a window sliced out of the middle of a corpus has no document
    # start. That looks like the obvious fix for Gemma's high absolute NLL here (5.66 nats
    # against 1.9 for the Qwens) and was measured as one: prepending `<bos>` takes Gemma to
    # 9.73 nats and makes `base` and `base2` disagree by 0.009+-0.016 where they had agreed
    # exactly. 5.66 is also what `accuracy.py` gets for this checkpoint through HF on a
    # different corpus (5.85), so the high value is the model, not a missing BOS.
    seq = MAX_LEN - 1
    # A wikitext row is ~300 characters, so ~75 tokens; ask for twice the windows' worth and
    # stop there rather than pulling all 4358 rows in 44 requests an arm. The assert below is
    # what catches a corpus that turns out shorter than this arithmetic assumes.
    rows = bench.fetch_rows(LM_SOURCE, LM_CONFIG, LM_SPLIT, cap=N_WIN * seq // 50)
    text = "".join(r["text"] for r in rows)
    ids = tok(text, add_special_tokens=False).input_ids
    assert len(ids) >= N_WIN * seq, f"{LM_SOURCE} gave {len(ids)} tokens, need {N_WIN * seq}"
    win = np.asarray(ids[: N_WIN * seq]).reshape(N_WIN, seq)
    outs = llm.generate([TokensPrompt(prompt_token_ids=w.tolist()) for w in win],
                        SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0))
    # From 1, because position 0 is predicted by nothing.
    return [-float(np.mean([next(iter(d.values())).logprob for d in o.prompt_logprobs[1:]]))
            for o in outs]


def throughput(llm) -> list[dict]:
    """Wall-clock for a fixed amount of decode work, repeated. Identical batch in every arm."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    rng = np.random.default_rng(0)
    vocab = llm.get_tokenizer().vocab_size
    # Fixed token ids rather than text: the batch has to be byte-identical across arms, and a
    # prompt that tokenizes differently on two models would not be.
    ids = rng.integers(1, vocab - 1, size=(N_REQ, PROMPT_LEN)).tolist()
    prompts = [TokensPrompt(prompt_token_ids=r) for r in ids]
    sp = SamplingParams(temperature=0, max_tokens=GEN_LEN, ignore_eos=True)
    out = []
    for r in range(REPEATS):
        t0 = time.perf_counter()
        res = llm.generate(prompts, sp)
        dt = time.perf_counter() - t0
        n_out = sum(len(x.outputs[0].token_ids) for x in res)
        assert n_out == N_REQ * GEN_LEN, f"decoded {n_out}, expected {N_REQ * GEN_LEN}"
        out.append({"repeat": r, "seconds": dt, "out_tok_per_s": n_out / dt,
                    "tot_tok_per_s": (n_out + N_REQ * PROMPT_LEN) / dt})
        print(f"  [{ARM}] repeat {r}: {dt:.2f}s  {n_out / dt:.1f} out-tok/s", flush=True)
    return out


def main() -> None:
    keep = None if ARM.startswith("base") else int(ARM)
    # The plugin reads this in every worker at load time. Setting it here rather than outside
    # keeps the arm name and the engine's actual configuration from drifting apart.
    if keep is None:
        os.environ.pop("PAR_KEEP", None)
    else:
        os.environ["PAR_KEEP"] = str(keep)
    print(f"[{TAG}/{ARM}] PAR_KEEP={os.environ.get('PAR_KEEP', '<unset>')} gpus={N_GPUS} "
          f"a2a={A2A or '<default>'}", flush=True)

    llm = engine()
    hit = gpqa(llm)
    print(f"[{TAG}/{ARM}] gpqa acc={hit.mean():.4f} n={len(hit)}", flush=True)
    pw = nll(llm)
    print(f"[{TAG}/{ARM}] nll={np.mean(pw):.5f} over {len(pw)} windows", flush=True)
    tp = throughput(llm)
    warm = [r["out_tok_per_s"] for r in tp[1:]] or [tp[0]["out_tok_per_s"]]

    OUT.mkdir(exist_ok=True)
    (OUT / f"vllm_{TAG}_{ARM}_r{REP}.json").write_text(json.dumps({
        # A result carries no clue which kernel wrote it, and `compare.py` globs arms by
        # tag - so without this a file left over from a superseded run aggregates
        # silently beside a fresh one. A missing field fails loudly; a stale number does not.
        "rep": int(REP),
        "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tag": TAG, "model": MODEL, "arm": ARM, "keep": keep, "n_gpus": N_GPUS,
        "vllm": __import__("vllm").__version__, "a2a": A2A or "<default>",
        "n_req": N_REQ, "prompt_len": PROMPT_LEN, "gen_len": GEN_LEN,
        "gpqa_acc": float(hit.mean()), "gpqa_hit": hit.astype(int).tolist(),
        # Per window, not just the mean: the arm gap is a paired difference and needs the
        # pairing kept. `compare.py` reads this list.
        "nll": float(np.mean(pw)), "nll_per_window": pw, "n_win": len(pw),
        "lm_source": f"{LM_SOURCE}/{LM_CONFIG}/{LM_SPLIT}",
        "throughput": tp,
        # Warm repeats only, median: the first pays for CUDA graph capture and the Triton
        # autotune, which is a real cost but not the one being compared. `compare.py` reads
        # the list, not the median, so the arm gap is tested against this spread.
        "warm_tok_per_s": warm,
        "out_tok_per_s": float(np.median(warm)),
    }, indent=1) + "\n")
    print(f"WROTE {OUT}/vllm_{TAG}_{ARM}_r{REP}.json", flush=True)


if __name__ == "__main__":
    main()
