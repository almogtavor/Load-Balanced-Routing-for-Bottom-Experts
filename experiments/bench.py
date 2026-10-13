"""Phase A2: can the model still do the task? GPQA-Diamond, every arm, paired per item.

Perplexity is the wrong instrument on its own. A delta-NLL of +0.01 nat is easy to report and
impossible to interpret - it does not say whether anything the model is actually used for got
worse. So the same arms are scored on a downstream benchmark, and the headline claim is a
*paired* test on per-item correctness rather than a comparison of two accuracy numbers.

198 graduate-level multiple-choice items sampled from GPQA, scored by ranking the mean token
log-likelihood of each option. Three deliberate choices:

* **Likelihood ranking, not generation.** Generation would make every arm a chat-template and
  answer-parsing problem on four families that format differently, and a parse failure would
  read as an accuracy drop. Ranking is one teacher-forced forward per item and measures the
  thing the intervention could plausibly damage - which expert the router picks - without a
  decoding layer in between. It also makes all seven arms affordable, where generation would
  not be.
* **198 items sampled from GPQA-main, not GPQA-Diamond.** Diamond is the 198-item subset and
  would have been the obvious choice, but the only copy carrying the four answer options is
  `Idavidrein/gpqa`, which is gated; the ungated Diamond mirrors have been reformatted to
  free-response and have no distractors to rank. So the items are drawn from the ungated
  `ankner/gpqa` mirror of GPQA-main with a fixed seed, which keeps the sample size the same and
  is honest about being a sample of main rather than Diamond. Nothing in the design depends on
  *which* items these are, only that every arm sees the same ones in the same choice order.

* **McNemar, on the items where the arms disagree.** 198 items is a weak instrument for an
  *absolute* accuracy: the standard error at ~35% is 3.4 points, so an unpaired comparison of
  two arms cannot resolve anything smaller than ~7 points. The paired test is far sharper,
  because the items both arms get right or both get wrong carry no information about the
  difference and are correctly discarded; only the discordant pairs count. The exact binomial
  p-value on those pairs is reported, and so is the discordant count, because when that count
  is small the honest reading is "not resolved" rather than "no difference".

The choice order is shuffled with a fixed seed, because GPQA stores the correct answer first,
and the same shuffle is used for every arm so the pairing is exact.

The rows come over the Hub's dataset-server API with `urllib`, not through `datasets`: the
container image has neither `datasets` nor `pandas` nor `pyarrow`, and installing them pulls a
dependency tree next to a torch the image already pins. Two HTTP requests and `json` are enough
for 448 rows.

Writes `runs/bench_<tag>.json`.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import urllib.parse
import urllib.request

import numpy as np

# torch, transformers and `moe` are imported where they are used, not here: `compare.py` wants
# `mcnemar` and nothing else, and it draws the figure from a laptop that has none of them.
MODEL = os.environ.get("MODEL", "")
TAG = os.environ.get("TAG", "")
N_GPUS = int(os.environ.get("GPUS", 4))
KEEPS = tuple(int(x) for x in os.environ.get("KEEPS", "6,4,2").split(","))
# The sample size, matching GPQA-Diamond's. `LIMIT` is for smoke tests, not the reported run.
N_ITEMS = int(os.environ.get("NITEMS", 198))
LIMIT = int(os.environ.get("LIMIT", 0))
OUT = pathlib.Path("runs")
# Ungated mirror of GPQA-main carrying the original `Correct Answer` / `Incorrect Answer i`
# columns, which the reformatted Diamond mirrors have dropped.
SOURCE = os.environ.get("GPQA", "ankner/gpqa")
SPLIT = os.environ.get("GPQA_SPLIT", "train")


def fetch_rows() -> list[dict]:
    """Every row of `SOURCE`, over the dataset-server API. 100 rows is its per-request cap."""
    tok = os.environ.get("HF_TOKEN")
    head = {"Authorization": f"Bearer {tok}"} if tok else {}
    rows: list[dict] = []
    while True:
        u = (f"https://datasets-server.huggingface.co/rows"
             f"?dataset={urllib.parse.quote(SOURCE, safe='')}&config=default"
             f"&split={SPLIT}&offset={len(rows)}&length=100")
        d = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=head), timeout=120))
        rows += [r["row"] for r in d["rows"]]
        if len(rows) >= d["num_rows_total"]:
            return rows


def load_items(rng) -> list[tuple[str, list[str], int]]:
    """`(context, four options, gold index)` per item, in a fixed shuffled choice order.

    The choice shuffle and the item subsample both come off the same seeded generator, so the
    item set is a property of the seed alone and is identical on all four models.
    """
    rows = fetch_rows()
    items = []
    for r in rows:
        opts = [r["Correct Answer"]] + [r[f"Incorrect Answer {i}"] for i in (1, 2, 3)]
        if not all(isinstance(o, str) and o.strip() for o in opts) or not r.get("Question"):
            continue                       # a handful of mirror rows have an empty distractor
        order = rng.permutation(4)
        items.append((f"{str(r['Question']).strip()}\nAnswer:",
                      [f" {opts[i].strip()}" for i in order], int(np.where(order == 0)[0][0])))
    keep = sorted(rng.choice(len(items), size=min(N_ITEMS, len(items)), replace=False))
    items = [items[i] for i in keep][: LIMIT or None]
    print(f"[data] {SOURCE}: {len(items)} of {len(rows)} items", flush=True)
    return items


def correctness(model, tok, items, dev, pad_id: int) -> np.ndarray:
    """`[n]` bool: did this arm rank the gold option first, item by item.

    All four options of an item go in one right-padded batch - scoring them one at a time is
    four forwards of a 40-token sequence and leaves the GPU idle. The per-option log-prob is
    gathered from its own unpadded span, so padding never enters the sum, and the sum is
    divided by the option's length because the options differ in length and an unnormalized
    sum would simply prefer the shortest one.
    """
    import torch

    with torch.no_grad():
        hit = np.zeros(len(items), dtype=bool)
        for i, (ctx, choices, gold) in enumerate(items):
            cid = tok(ctx, return_tensors="pt").input_ids[0]
            seqs, lens = [], []
            for c in choices:
                ch = tok(c, add_special_tokens=False, return_tensors="pt").input_ids[0]
                seqs.append(torch.cat([cid, ch]))
                lens.append(len(ch))
            w = max(len(x) for x in seqs)
            inp = torch.full((len(seqs), w), pad_id, dtype=torch.long)
            att = torch.zeros((len(seqs), w), dtype=torch.long)
            for j, sq in enumerate(seqs):
                inp[j, : len(sq)] = sq
                att[j, : len(sq)] = 1
            lg = model(input_ids=inp.to(dev), attention_mask=att.to(dev)).logits.float()
            lp = torch.log_softmax(lg[:, :-1], -1)
            tot = []
            for j, sq in enumerate(seqs):
                n, k = len(sq), lens[j]
                tgt = sq[n - k : n].to(dev)
                tot.append(lp[j, n - k - 1 : n - 1].gather(-1, tgt[:, None]).sum().item() / k)
            hit[i] = int(np.argmax(tot)) == gold
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(items)} items", flush=True)
        return hit


def mcnemar(base: np.ndarray, arm: np.ndarray) -> dict:
    """Exact two-sided McNemar on per-item correctness. Concordant pairs carry no information.

    `b` is items the baseline got right and the arm got wrong, `c` the reverse. Under the null
    the split of the `b + c` discordant pairs is Binomial(n, 1/2), so the exact p-value is the
    two-sided tail - computed from `math.comb` rather than a chi-square approximation, which is
    unreliable at exactly the small discordant counts this benchmark size produces, and rather
    than scipy, which the transformers-4 venv does not have.
    """
    b = int((base & ~arm).sum())
    c = int((~base & arm).sum())
    n, k = b + c, min(b, c)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"acc": float(arm.mean()), "dacc": float(arm.mean() - base.mean()),
            "base_right_arm_wrong": b, "base_wrong_arm_right": c, "n_discordant": n,
            "p_mcnemar": p, "resolved": bool(p < 0.05)}


def main() -> None:
    import torch
    from transformers import AutoTokenizer

    import moe

    assert MODEL and TAG, "MODEL and TAG are required for the measurement"
    cfg, n_experts, top_k = moe.routing_cfg(MODEL, N_GPUS)
    print(f"{TAG}: E={n_experts} top_k={top_k} G={N_GPUS}", flush=True)
    items = load_items(np.random.default_rng(0))
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    pad = tok.pad_token_id or tok.eos_token_id or 0
    model = moe.load(MODEL)
    state, handles = moe.install(model, n_experts, top_k, moe.moe_shapes(cfg)["hidden"], N_GPUS)
    dev = next(model.parameters()).device

    hits: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {}
    for name, arm, keep in moe.arms_of(top_k, KEEPS):
        torch.manual_seed(0)        # same reason as in `accuracy.py`: a reproducible control
        state.update(arm=arm, keep=keep, changed=0, rows=0)
        print(f"[{name}] arm={arm} keep={keep}", flush=True)
        hits[name] = correctness(model, tok, items, dev, pad)
        changed = state["changed"] / max(state["rows"], 1)
        if name == "baseline":
            results[name] = {"acc": float(hits[name].mean()), "changed_frac": changed}
        else:
            results[name] = {**mcnemar(hits["baseline"], hits[name]), "changed_frac": changed}
        r = results[name]
        print(f"[{name}] acc={100 * r['acc']:5.2f} "
              + (f"dacc={100 * r['dacc']:+5.2f} pts  discordant={r['n_discordant']} "
                 f"p={r['p_mcnemar']:.3f} {'resolved' if r['resolved'] else 'NOT resolved'} "
                 if name != "baseline" else "")
              + f"changed={changed:.3f}", flush=True)

    for h in handles:
        h.remove()

    # The same hard stop as Phase A: the no-op arm's selection must equal the baseline's token
    # for token, or the hook is perturbing the routing by itself and no other arm on this model
    # means anything. Its *answers* are held to McNemar instead of to zero flips, because the
    # mask still changes the gate softmax's denominator and the top-k renormalization cancels
    # that only in exact arithmetic - in bf16 a handful of items can land on the other side of a
    # close call. What must not happen is a flip count McNemar can separate from chance.
    # `baseline2` is the same arm run twice, so its discordant count is how many items this
    # benchmark flips on its own. The no-op arm is allowed that many and no more: a flip count
    # McNemar can separate from chance, or one that clears the repeat baseline's, is the hook
    # perturbing the routing rather than bf16 rounding.
    noop, floor = results["noop"], results["baseline2"]
    assert noop["changed_frac"] == 0.0, f"no-op arm changed {noop['changed_frac']:.4f} of rows"
    assert not noop["resolved"], (f"no-op arm flipped {noop['n_discordant']} items "
                                  f"({noop['dacc']:+.4f}, p={noop['p_mcnemar']:.3f})")
    assert noop["n_discordant"] <= max(floor["n_discordant"], 2), (
        f"no-op arm flipped {noop['n_discordant']} items where the repeat baseline flipped "
        f"{floor['n_discordant']}")

    OUT.mkdir(exist_ok=True)
    (OUT / f"bench_{TAG}.json").write_text(json.dumps({
        "tag": TAG, "model": MODEL, "bench": SOURCE, "n_items": len(items),
        "n_experts": n_experts, "top_k": top_k, "n_gpus": N_GPUS,
        "noise_floor": {"n_discordant": floor["n_discordant"], "dacc": floor["dacc"],
                        "p_mcnemar": floor["p_mcnemar"]},
        # Per-item correctness is kept so a reader can redo the pairing, or pool items across
        # models, without rerunning anything.
        "correct": {k: v.astype(np.uint8).tolist() for k, v in hits.items()},
        "arms": results,
    }, indent=1) + "\n")
    print(f"WROTE {OUT}/bench_{TAG}.json", flush=True)


def _check() -> None:
    """McNemar against hand-computable cases, since the whole claim rests on this p-value."""
    r = np.ones(10, bool)
    assert mcnemar(r, r.copy())["p_mcnemar"] == 1.0            # no discordant pairs at all
    a = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], bool)
    b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], bool)
    m = mcnemar(a, b)
    # Five pairs, all one way: 2 * (1/2)^5 = 0.0625, the smallest p five pairs can give - which
    # is the point of reporting the discordant count beside it.
    assert m == {**m, "n_discordant": 5, "base_right_arm_wrong": 5, "base_wrong_arm_right": 0}
    assert abs(m["p_mcnemar"] - 0.0625) < 1e-12 and not m["resolved"]
    c = np.zeros(10, bool); c[:8] = True
    m = mcnemar(np.zeros(10, bool), c)
    assert m["n_discordant"] == 8 and abs(m["p_mcnemar"] - 2 / 256) < 1e-12 and m["resolved"]
    print("bench.py: ok")


if __name__ == "__main__":
    _check() if os.environ.get("CHECK") else main()
