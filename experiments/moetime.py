#!/usr/bin/env python3
"""MoE layer time inside a running engine, per setting, bucketed by the forward's token count.

End-to-end tokens/s is the wrong instrument for this rule. The rule moves dispatch bytes, and
dispatch is a fraction of the MoE layer, which is a fraction of the step: at ep=4 on one host a
24% byte cut arrives end-to-end as ~2.7%, under the launch-to-launch floor. So time the thing
the rule acts on. Expert GEMM work is conserved by construction (`imbalance.py` asserts it), so
a drop in MoE layer time at a fixed token count is the collective and nothing else.

Two CUDA events per MoE forward, recorded on the compute stream and reduced a chunk behind the
front, so nothing here synchronises on work that has not already finished. Installed in
every worker by `plugin.register()` via `PLACEMENT_PROBE`, on base settings too - it reads.

`_split` adds a second and third span inside each forward, around DeepEP's dispatch and
combine, because the whole-layer clock nets the byte saving against a grouped expert GEMM that
skews when the allowed expert set narrows. Dispatch is the half the rule acts on.

Buckets are powers of two of the forward's token count, because a prefill forward and a decode
forward are different measurements and the mix between them drifts between launches. Comparing
settings means comparing the same bucket.

Writes `runs/moetime_{TAG}_{SETTING}_r{REP}_{rank}.json` every chunk, so a worker killed by a
signal - which is how vLLM tears them down - still leaves the measurement behind.
"""

from __future__ import annotations

import json
import os
import pathlib
import statistics as st

CHUNK = 64                                     # event pairs reduced per flush
WARMUP = int(os.environ.get("MOE_WARMUP", 96))  # forwards discarded before measuring

_pend: list = []                  # (start, end, phase, bucket) awaiting reduction
_ms: dict = {}                    # (phase, bucket) -> every timing, so the statistic is a median
_seen = 0
_nmoe = 0                         # measured forwards since the last flush
_cur = 0                          # bucket of the forward in flight, for the inner spans
_live = False                     # inside a measured forward: the inner spans key off this
_out: pathlib.Path | None = None

PHASES = ("moe", "disp", "comb")   # whole layer, DeepEP dispatch, DeepEP combine


def _flush() -> None:
    global _pend
    for start, end, ph, b in _pend:
        end.synchronize()
        _ms.setdefault((ph, b), []).append(start.elapsed_time(end))
    _pend = []
    _out.parent.mkdir(exist_ok=True)
    _out.write_text(json.dumps({
        "setting": os.environ.get("SETTING", "base"),
        "keep": os.environ.get("PLACEMENT_KEEP"),
        "load_penalty": os.environ.get("PLACEMENT_LOAD_PENALTY"),
        "seen": _seen,
        # bucket -> [median ms per MoE forward, forwards]. Keys are the low edge of a power-of-
        # two token-count bucket; a bucket present in one setting and not another is not comparable.
        # Median, like `serve_run.py`'s repeat statistic: an setting's first repeat is cold and a
        # mean would carry it into the comparison.
        # "buckets" is the whole layer, kept under its original name so the analysis that
        # already reads it needs no change; "disp"/"comb" are the two halves of the collective.
        **{("buckets" if ph == "moe" else ph):
           {str(b): [st.median(v), len(v)] for (p, b), v in sorted(_ms.items()) if p == ph}
           for ph in PHASES},
    }, indent=1))


def install() -> None:
    """Wrap the MoE layer's forward. Any failure leaves the engine exactly as it was: this is a
    measurement riding along on a throughput run, and it must not be able to end one."""
    try:
        import torch

        # `MoERunner._forward_impl` - router selection, dispatch, expert GEMM, combine - is the
        # only one of the three candidates a timer can see. `RoutedExperts.forward` raises in 0.31
        # (callers reach `forward_modular`), and `MoERunner.forward` only ever runs under Dynamo
        # tracing: it hands off to the `torch.ops.vllm.moe_forward` custom op, so the compiled
        # graph calls the op directly at run time and the Python body is dead. The op's callee is
        # `_forward_impl`, which is outside the graph and therefore safe to record events in.
        from vllm.model_executor.layers.fused_moe.layer import MoERunner as MoE

        global _out
        # Absolute, off this file: a worker process runs with cwd=/vllm-workspace, so a relative
        # `runs/` would write four copies into the image and none onto the shared volume the
        # analysis reads.
        _out = pathlib.Path(__file__).resolve().parent.parent / "runs" / (
            f"moetime_{os.environ.get('TAG', 'x')}_{os.environ.get('SETTING', 'base')}"
            # The pid, not a rank env var: every worker process of a DP launch reported rank 0,
            # so four engine cores raced on one file and three measurements were lost.
            f"_r{os.environ.get('REP', '0')}_{os.getpid()}.json")
        if "_forward_impl" not in MoE.__dict__:
            # A rename upstream would otherwise leave the probe wrapping an inherited
            # attribute, or nothing at all, and report a silent zero.
            raise AttributeError(f"{MoE.__name__} defines no _forward_impl")
        inner = MoE._forward_impl

        def forward(self, hidden_states, *a, **kw):
            global _seen, _cur, _live, _nmoe
            _seen += 1
            # A graph capture records kernels rather than running them, so an event recorded
            # inside one times the replay of whatever ran last. DeepEP's high-throughput
            # kernels disable graphs anyway, which is the backend this measures.
            if _seen <= WARMUP or torch.cuda.is_current_stream_capturing():
                return inner(self, hidden_states, *a, **kw)
            # One row per token, so the bucket separates a prefill forward from a decode one.
            # Reading `.shape` is metadata, not a sync.
            _cur = 1 << max(0, hidden_states.shape[0].bit_length() - 1)
            _live = True
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            try:
                out = inner(self, hidden_states, *a, **kw)
            finally:
                _live = False
            end.record()
            _pend.append((start, end, "moe", _cur))
            # Counted in whole forwards, not in spans: the split adds two spans per forward and
            # flushing three times as often would change the cadence of the one synchronise
            # this probe performs, which is the thing the measured floor was established with.
            _nmoe += 1
            if _nmoe >= CHUNK:
                _nmoe = 0
                _flush()
            return out

        MoE._forward_impl = forward
        print(f"[moet] timing {MoE.__name__}._forward_impl -> {_out}", flush=True)
        _split(torch)
    except Exception as e:  # noqa: BLE001 - see the docstring
        print(f"[moet] not installed: {type(e).__name__}: {e}", flush=True)


def _split(torch) -> None:
    """Time dispatch and combine separately, so the byte saving is read off the term it acts on.

    The whole-layer clock nets two opposite effects: fewer dispatch bytes, and a grouped expert
    GEMM whose group sizes skew once the allowed expert set is restricted - in the engine,
    unlike the offline simulator, nothing caps an expert's token count. Splitting the clock
    separates them, and dispatch is the half the rule acts on.

    Both public entry points of each half fan into one method: `prepare` calls
    `self.prepare_async`, and `finalize`/`finalize_async` both call `self._finalize`. So two
    wrappers cover four paths. Each returns a receiver that performs the wait on the collective's
    completion event, so the span closes inside the receiver, not when the call returns.
    """
    try:
        from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht import (
            DeepEPHTPrepareAndFinalize as P)
    except Exception as e:  # noqa: BLE001 - the other backends have no dispatch worth splitting
        print(f"[moet] no split: {type(e).__name__}: {e}", flush=True)
        return

    def span(phase: str, fn):
        def wrapped(*a, **kw):
            if not _live:
                return fn(*a, **kw)
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            b = _cur
            start.record()
            out = fn(*a, **kw)
            if not callable(out):          # the synchronous path: it already waited
                end.record()
                _pend.append((start, end, phase, b))
                return out

            def receiver():
                v = out()
                end.record()
                _pend.append((start, end, phase, b))
                return v
            return receiver
        return wrapped

    for name in ("prepare_async", "_finalize"):
        if name not in P.__dict__:
            # Same reasoning as the `_forward_impl` guard: a rename upstream must fail loudly
            # rather than leave a timer on nothing.
            print(f"[moet] no split: {P.__name__} defines no {name}", flush=True)
            return
    P.prepare_async = span("disp", P.prepare_async)
    P._finalize = span("comb", P._finalize)
    print("[moet] timing dispatch and combine separately", flush=True)


if __name__ == "__main__":
    # The span has to close inside the receiver on the async path and immediately on the sync
    # one; getting that backwards times the launch instead of the collective, which still looks
    # like a plausible number. A fake event checks it without a GPU.
    import sys
    import types

    class _Event:
        def __init__(self, enable_timing=False): pass

        def record(self): pass
        def synchronize(self): pass
        def elapsed_time(self, other): return 1.0

    class _Torch:
        class cuda:
            Event = _Event

    class P:
        def prepare_async(self):
            return lambda: "tokens"

        def _finalize(self, do_async):
            return (lambda: None) if do_async else None

    mod = types.ModuleType("deepep_ht")
    mod.DeepEPHTPrepareAndFinalize = P
    for n in ("vllm", "vllm.model_executor", "vllm.model_executor.layers",
              "vllm.model_executor.layers.fused_moe",
              "vllm.model_executor.layers.fused_moe.prepare_finalize"):
        sys.modules.setdefault(n, types.ModuleType(n))
    sys.modules["vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht"] = mod

    _live, _cur = True, 16384
    _split(_Torch)
    p = P()
    receiver = p.prepare_async()
    assert not _pend, "dispatch span closed at launch, before the wait on the collective"
    assert receiver() == "tokens", "the wrapper swallowed the dispatch result"
    assert [e[2] for e in _pend] == ["disp"], _pend
    assert p._finalize(do_async=False) is None
    assert [e[2] for e in _pend] == ["disp", "comb"], "sync combine was not timed"
    receiver = p._finalize(do_async=True)
    assert [e[2] for e in _pend] == ["disp", "comb"], "async combine closed before the wait"
    receiver()
    assert [e[2] for e in _pend] == ["disp", "comb", "comb"], _pend
    assert all(e[3] == 16384 for e in _pend), "spans lost the token bucket"
    print("ok")
