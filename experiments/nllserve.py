"""One setting of the served NLL check: teacher-forced NLL read from `vllm serve` prompt
logprobs, on the windows `accuracy.py` scores, through the same server `serve_run.py` times.

The HF evaluation code gave Gemma-4-26B a baseline of 5.7-9.8 nats on wikitext, and it never ran the
tile kernel the throughput numbers come from. Here the plugin, set by `SETTING`, is the deployed
kernel, and every setting is the same engine launched again - so `base2` against `base` is the
floor any difference has to clear. Each setting merges into `runs/nll_<tag>.json`.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import time
import urllib.request

import numpy as np
from transformers import AutoTokenizer

import accuracy as A

# The window plus the one token `max_tokens=1` asks for; `or`, since the job exports "".
os.environ["MAX_LEN"] = os.environ.get("MAX_LEN") or str(A.SEQ_LEN + 8)
import serve_run as S  # noqa: E402

# Qwen3 scores ~1.9 on these streams and a broken setup (no BOS, wrong tokenizer) 5+.
MAX_BASE_NLL = 4.0
# CHAT=1 scores each window as the model's own reply to a chat request, for checkpoints that
# are instruction-tuned only: Gemma-4-26B-it puts -28 nats on " capital" after "<bos>The".
CHAT = os.environ.get("CHAT") == "1"
START = 1  # first scored position; the chat prefix moves it


def windows() -> np.ndarray:
    """`N_WIN x SEQ_LEN` ids, byte-identical on every setting, BOS first where the model has one."""
    win = A.token_ids((A.N_WIN + 1) * A.SEQ_LEN)[: A.N_WIN * A.SEQ_LEN].reshape(A.N_WIN, A.SEQ_LEN)
    # Read from the attribute: Gemma 4's tokenizer does not add it on encode. Qwen has none.
    tok = AutoTokenizer.from_pretrained(S.MODEL, trust_remote_code=True)
    if CHAT:
        global START
        pre = tok.apply_chat_template([{"role": "user", "content": "Write an encyclopedia article."}],
                                      add_generation_prompt=True, tokenize=True)
        pre = list(getattr(pre, "input_ids", pre))
        START = len(pre)
        return np.concatenate([np.tile(pre, (len(win), 1)), win[:, : A.SEQ_LEN - START]], 1)
    if (bos := tok.bos_token_id) is not None:
        win[:, 0] = bos
    return win


def window_nll(win: list[int]) -> float:
    """Mean NLL of positions `START..L-1`, from the server's logprob of each actual token."""
    body = json.dumps({"model": S.MODEL, "prompt": win, "max_tokens": 1, "temperature": 0,
                       "prompt_logprobs": 0}).encode()
    req = urllib.request.Request(S.URL + "/v1/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        lp = json.load(r)["choices"][0]["prompt_logprobs"]
    assert len(lp) == len(win) and lp[0] is None, f"{len(lp)} logprobs for {len(win)} tokens"
    # Keyed by token id, so the lookup also checks the server scored this stream and not a
    # re-tokenized one. vLLM clamps -inf to -9999, which then shows as an absurd mean.
    return -float(np.mean([lp[i][str(t)]["logprob"] for i, t in enumerate(win) if i >= START]))


def main() -> None:
    keep, load_penalty, mass = S.setting_env()
    win = windows()
    sha = hashlib.sha256(win.tobytes()).hexdigest()
    print(f"[{S.TAG}/{S.SETTING}] nll keep={keep} load_penalty={load_penalty} mass={mass} "
          f"dp={S.DP} a2a={S.A2A} windows={A.N_WIN}x{A.SEQ_LEN} stream={A.STREAM} "
          f"chat={CHAT} start={START} first ids {win[0, :START + 4].tolist()} sha={sha[:12]}", flush=True)
    p = S.serve()
    try:
        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            pw = list(ex.map(window_nll, win.tolist()))
    finally:
        S.kill(p)

    path = S.OUT / f"nll_{S.TAG}.json"
    # The first setting of a job starts the file, so a leftover from an earlier job under this
    # tag cannot pair with this one.
    first = S.SETTING == (os.environ.get("SETTINGS") or S.SETTING).split(",")[0]
    res = (json.loads(path.read_text()) if path.exists() and not first else
           {"tag": S.TAG, "model": S.MODEL, "stream": A.STREAM, "seq_len": A.SEQ_LEN,
            "n_win": A.N_WIN, "windows_sha256": sha, "vllm": __import__("vllm").__version__,
            "a2a": S.A2A, "dp": S.DP, "tp": S.N_GPUS // S.DP, "settings": {}})
    assert res["windows_sha256"] == sha, "windows differ from the settings already in the file"
    assert S.SETTING == "base" or "base" in res["settings"], "run base first: every dnll pairs with it"
    d = np.array(pw) - np.array(res["settings"]["base"]["per_window"] if S.SETTING != "base" else pw)
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    r = res["settings"][S.SETTING] = {
        "nll": float(np.mean(pw)), "per_window": pw, "dnll": float(d.mean()), "dnll_se": se,
        "t": float(d.mean() / se) if se > 0 else 0.0, "keep": keep, "load_penalty": load_penalty,
        "mass": mass, "rep": int(S.REP),
        "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    S.OUT.mkdir(exist_ok=True)
    path.write_text(json.dumps(res, indent=1) + "\n")
    print(f"[{S.SETTING}] nll={r['nll']:.5f} dnll={r['dnll']:+.5f}+-{r['dnll_se']:.5f} "
          f"(t={r['t']:+.1f})", flush=True)
    print(f"WROTE {path}", flush=True)
    # Written first, so the broken number is on file; then the job stops before the other settings.
    if S.SETTING.startswith("base"):
        print(f"[{S.SETTING}] baseline nll {r['nll']:.4f} nats", flush=True)
        assert r["nll"] < MAX_BASE_NLL, f"baseline nll {r['nll']:.3f} >= {MAX_BASE_NLL}: setup is broken"


if __name__ == "__main__":
    main()
