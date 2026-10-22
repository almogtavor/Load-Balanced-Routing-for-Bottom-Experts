"""Expert-to-rank placement by co-activation: experts a token tends to pick together share a rank.

Default routing is unchanged by a placement (it only renames which GPU holds which expert), so
the model is the same model; what changes is how many ranks a token's top-k spans, and with it
how often the mass rule's tail finds a good expert on a rank the head already visits.

Per layer, from default top-k on a calibration stream, `C[i, j]` counts tokens routing to both
i and j. Starting from id blocks (expert e on rank e*G//E), greedy swaps first bring every
rank's routed load within `SLACK` of the mean, then raise the within-rank co-activation without
leaving that bound.
With `CALIB=mix` the calibration tokens come in equal parts from wikitext, GSM8K train (chat
format, gold solution) and HellaSwag train, and the load bound holds for each of the three on its
own: a placement balanced on wikitext alone came out 17-26% busier than id blocks on the
evaluation tasks.
Fan-out is then measured on held-out tokens of the same stream. Writes `runs/coact_<tag>.npy`,
`[layers, E]` expert -> rank, and the calibration top-k to `runs/coact_ids_<tag>.npy`
(`[layers, tokens, k]`) so the placement can be redone without a GPU (`--from-ids`).
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import torch

import accuracy as A
import moe

MODEL = os.environ.get("MODEL", "/models/bf16")
TAG = os.environ.get("TAG", "coact")
G = int(os.environ.get("RANKS", 4))
SLACK = float(os.environ.get("SLACK", 0.02))
WINDOWS = int(os.environ.get("COACT_WINDOWS", 64))
CALIB = os.environ.get("CALIB", "wikitext")


def place(C: np.ndarray, f: np.ndarray, G: int, slack: float, iters: int = 4000) -> np.ndarray:
    """Expert -> rank, balanced to `slack` of the mean load, maximizing within-rank `C`.

    `f` is `[E]`, or `[D, E]` for D calibration domains, each of which must meet the bound.
    """
    f = np.atleast_2d(f).astype(np.float64)
    f = f / f.sum(1, keepdims=True)                              # each domain's load sums to 1
    E = f.shape[1]
    r = np.arange(E) * G // E
    C = C.astype(np.float64).copy()
    np.fill_diagonal(C, 0)
    M = np.stack([C[:, r == g].sum(1) for g in range(G)], 1)     # [E, G] co-activation with rank g
    load = np.stack([np.bincount(r, weights=fd, minlength=G) for fd in f])   # [D, G]
    cap = 1 / G * (1 + slack)
    idx = np.arange(E)

    def swap(a, b):
        A_, B_ = r[a], r[b]
        M[:, A_] += C[:, b] - C[:, a]
        M[:, B_] += C[:, a] - C[:, b]
        load[:, A_] += f[:, b] - f[:, a]
        load[:, B_] += f[:, a] - f[:, b]
        r[a], r[b] = B_, A_

    # Balance first: id blocks can start 40% over the mean, and that is compute on the busiest rank.
    for _ in range(iters):
        g = int(np.argmax(load.max(0)))
        worst = load[:, g].max()
        if worst <= cap:
            break
        a = np.flatnonzero(r == g)[:, None]
        b = np.flatnonzero(r != g)[None, :]
        new = np.maximum((load[:, g, None, None] - f[:, a] + f[:, b]).max(0),
                         (load[:, r[b]] + f[:, a] - f[:, b]).max(0))
        i, j = np.unravel_index(np.argmin(new), new.shape)
        if new[i, j] >= worst:
            break
        swap(a[i, 0], b[0, j])
    cap = max(cap, load.max())
    for _ in range(iters):
        ra = r[:, None]; rb = r[None, :]
        gain = (M[idx[:, None], rb] - M[idx[:, None], ra]
                + M[idx[None, :], ra] - M[idx[None, :], rb] - 2 * C)
        la = (load[:, ra] - f[:, :, None] + f[:, None, :]).max(0)    # rank of a after the swap
        lb = (load[:, rb] - f[:, None, :] + f[:, :, None]).max(0)
        gain[(ra == rb) | (la > cap) | (lb > cap)] = -np.inf
        a, b = np.unravel_index(np.argmax(gain), gain.shape)
        if gain[a, b] <= 1e-9:
            break
        swap(a, b)
    assert np.bincount(r, minlength=G).tolist() == [E // G] * G
    return r


def fanout(ids: np.ndarray, r: np.ndarray, G: int) -> float:
    hot = np.zeros((len(ids), G), bool)
    np.put_along_axis(hot, r[ids], True, 1)
    return float(hot.sum(1).mean())


@torch.no_grad()
def main() -> None:
    cfg, E, k = moe.routing_cfg(MODEL, G)
    model = moe.load(MODEL)
    gates = moe.routers(model, E, int(cfg.hidden_size))
    order = {id(g): i for i, g in enumerate(gates)}
    seen: dict[int, list] = {}

    def hook(mod, args, out):
        z = (out[0] if isinstance(out, tuple) else out).reshape(-1, E)
        seen.setdefault(order[id(mod)], []).append(z.float().topk(k, -1).indices.short().cpu())

    for g in gates:
        g.register_forward_hook(hook)
    pathlib.Path("runs").mkdir(exist_ok=True)
    doms = calib_windows()
    dev = model.get_input_embeddings().weight.device
    dom = []
    for d, win in enumerate(doms):
        for i in range(0, len(win), 4):
            model(input_ids=torch.as_tensor(win[i : i + 4], device=dev))
            dom.append(np.full(win[i : i + 4].size, d, np.int8))
    all_ids = np.stack([torch.cat(seen[l]).numpy() for l in range(len(gates))])
    dom = np.concatenate(dom)
    np.save(pathlib.Path("runs") / f"coact_ids_{TAG}.npy", all_ids)
    np.save(pathlib.Path("runs") / f"coact_dom_{TAG}.npy", dom)
    build(all_ids, E, dom)


def calib_windows() -> list[np.ndarray]:
    """Token windows per calibration domain; none of them overlaps an evaluation set."""
    L = A.SEQ_LEN
    if CALIB == "wikitext":
        return [A.token_ids((WINDOWS + 1) * L)[: WINDOWS * L].reshape(WINDOWS, L)]
    import json
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    n = WINDOWS // 3
    eos = [tok.eos_token_id or 0]

    def stream(texts):
        ids = []
        for t in texts:
            ids += tok(t, add_special_tokens=False).input_ids + eos
            if len(ids) >= n * L:
                return np.asarray(ids[: n * L], np.int64).reshape(n, L)
        raise SystemExit(f"calibration set ran out at {len(ids)} ids")

    def rows(path):
        return [json.loads(x) for x in pathlib.Path(path).read_text().splitlines()]

    gsm = (tok.apply_chat_template([{"role": "user", "content": r["question"]}],
                                   add_generation_prompt=True, tokenize=False, enable_thinking=False)
           + r["answer"] for r in rows("/results/gsm8k_train.jsonl"))
    hs = (r["ctx"] + " " + r["endings"][r["label"]] for r in rows("/results/hellaswag_train.jsonl"))
    wiki = A.token_ids((n + 1) * L)[: n * L].reshape(n, L)
    return [wiki, stream(gsm), stream(hs)]


def build(all_ids: np.ndarray, E: int, dom: np.ndarray | None = None) -> None:
    out = np.zeros((len(all_ids), E), np.int64)
    if dom is None:
        dom = np.zeros(all_ids.shape[1], np.int8)
    # Fit on the first half of every domain, hold out the second half.
    fit_m = np.zeros(len(dom), bool)
    for d in np.unique(dom):
        w = np.flatnonzero(dom == d)
        fit_m[w[: len(w) // 2]] = True
    for l in range(len(all_ids)):
        ids = all_ids[l].astype(np.int64)
        fit, held = ids[fit_m], ids[~fit_m]
        oh = np.zeros((len(fit), E), np.float32)
        np.put_along_axis(oh, fit, 1.0, 1)
        C = oh.T @ oh
        f = np.stack([oh[dom[fit_m] == d].sum(0) for d in np.unique(dom)])
        out[l] = place(C, f, G, SLACK)
        block = np.arange(E) * G // E
        hd = dom[~fit_m]

        def busiest(rk):
            return max(np.bincount(rk[held[hd == d]].ravel(), minlength=G).max()
                       / (held[hd == d].size / G) for d in np.unique(dom))
        print(f"layer {l}: fanout {fanout(held, block, G):.3f} -> {fanout(held, out[l], G):.3f}  "
              f"worst-domain busiest/mean {busiest(block):.3f} -> {busiest(out[l]):.3f}", flush=True)
    path = pathlib.Path("runs") / f"coact_{TAG}.npy"
    print(f"mean fanout {np.mean([fanout(all_ids[l][len(all_ids[l]) // 2:].astype(np.int64), np.arange(E) * G // E, G) for l in range(len(all_ids))]):.3f} -> "
          f"{np.mean([fanout(all_ids[l][len(all_ids[l]) // 2:].astype(np.int64), out[l], G) for l in range(len(all_ids))]):.3f}", flush=True)
    np.save(path, out)
    print(f"WROTE {path}", flush=True)


def _check() -> None:
    # Two planted cliques of co-routed experts end up on one rank each.
    rng = np.random.default_rng(0)
    E, G, k = 16, 4, 4
    groups = rng.permutation(E).reshape(G, E // G)
    ids = np.stack([groups[rng.integers(G)] for _ in range(2000)])
    oh = np.zeros((len(ids), E)); np.put_along_axis(oh, ids, 1.0, 1)
    r = place(oh.T @ oh, oh.sum(0), G, 0.5)
    assert fanout(ids, r, G) == 1.0, fanout(ids, r, G)
    # Two domains with opposite hot experts: each must end up within the bound on its own.
    f = np.stack([np.linspace(1, 3, E), np.linspace(3, 1, E)])
    r = place(np.ones((E, E)), f, G, 0.05)
    for fd in f:
        ld = np.bincount(r, weights=fd / fd.sum(), minlength=G)
        assert ld.max() <= 1 / G * 1.05 + 1e-9, ld
    print("coact_place: ok")


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        _check()
    elif "--from-ids" in sys.argv:
        path = sys.argv[sys.argv.index("--from-ids") + 1]
        ids = np.load(path)
        dp = pathlib.Path(path.replace("coact_ids_", "coact_dom_"))
        build(ids, int(ids.max()) + 1, np.load(dp) if dp.exists() else None)
    else:
        main()
