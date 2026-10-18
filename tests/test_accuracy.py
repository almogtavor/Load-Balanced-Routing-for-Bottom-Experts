"""CPU checks for the quality experiment's production-routing and statistical contracts."""
import pathlib
import sys

import numpy as np
import pytest
import torch

import accuracy
import moe
import placement as P
from placement_router.kernel import gate_weights, rewrite_torch, select


def logits_fixture():
    z = torch.full((2, 16), -8.0)
    z[:, [0, 8, 1, 2]] = torch.tensor([6.0, 5.8, 5.6, 5.5])
    z[0, [9, 10]] = torch.tensor([5.0, 4.9])
    z[1, [9, 10]] = torch.tensor([3.0, 2.9])
    return z


class QwenRouter(torch.nn.Module):
    norm_topk_prob = True

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(16))

    def forward(self, x):
        z = x @ self.weight.T
        vals, ids = z.float().softmax(-1).topk(4, -1)
        vals = vals / vals.sum(-1, keepdim=True)
        return z, vals.to(x.dtype), ids


class GemmaRouter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(16, 16, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(16))
        self.register_buffer("per_expert_scale", torch.linspace(0.8, 1.2, 16))

    def forward(self, x):
        probs = self.proj(x).float().softmax(-1)
        vals, ids = probs.topk(4, -1)
        vals = vals / vals.sum(-1, keepdim=True)
        return probs, (vals * self.per_expert_scale[ids]).to(x.dtype), ids


class TinyModel(torch.nn.Module):
    def __init__(self, gate):
        super().__init__()
        self.router = gate

    def forward(self, x):
        return self.router(x)


def sorted_weights(weights, ids):
    return weights.gather(1, ids.argsort(-1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mask_matches_production_and_penalty_changes_selection(dtype):
    z = logits_fixture().to(dtype)
    rank = P.gpu_of(16, 2)
    plain = P.allowed_mask(z, 2, rank, 2, 4)
    balanced = P.allowed_mask(z, 2, rank, 2, 4, 1.0)
    assert not torch.equal(plain[0], balanced[0])
    assert torch.equal(plain[1], balanced[1])
    for penalty in (0.0, 0.5, 1.0):
        mask = P.allowed_mask(z, 2, rank, 2, 4, penalty)
        ids = P.mask_logits(z, mask).topk(4, -1).indices
        _, production = select(z, 4, 2, 2, load_penalty=penalty)
        assert torch.equal(ids.sort(-1).values, production.long().sort(-1).values)
        assert torch.all(mask.sum(1) == 4)
        assert torch.all(P.fanout(ids, rank, 2) == 2)
        assert (mask[:, [0, 8]]).all()  # protected head survives


@pytest.mark.parametrize("router_cls", [QwenRouter, GemmaRouter])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_baseline_noop_and_balanced_weights(router_cls, dtype):
    z = logits_fixture().to(dtype)
    model = TinyModel(router_cls()).to(dtype)
    native = model(z)
    state, handles = moe.install(model, 16, 4, 16, 2, trace_tokens=10, parity_tokens=1)
    try:
        baseline = model(z)
        assert all(torch.equal(a, b) for a, b in zip(native, baseline))
        state.update(arm="placement", keep=4, load_penalty=0.0)
        noop = model(z)
        assert torch.equal(noop[2].sort(-1).values, native[2].sort(-1).values)
        assert torch.equal(sorted_weights(noop[1], noop[2]), sorted_weights(native[1], native[2]))
        assert state["changed"] == 0
        state.update(keep=2, load_penalty=1.0, parity={})
        got = model(z)
        expected_w, expected_ids = select(
            z, 4, 2, 2, scale=getattr(model.router, "per_expert_scale", None), load_penalty=1.0)
        assert torch.equal(got[0], native[0])
        assert torch.equal(got[2].sort(-1).values, expected_ids.long().sort(-1).values)
        torch.testing.assert_close(sorted_weights(got[1], got[2]),
                                   sorted_weights(expected_w.to(got[1].dtype), expected_ids.long()), rtol=1e-6, atol=1e-7)
        assert state["parity"][0]["rows"] == 1
        assert state["parity"][0]["selection_mismatch_rows"] == 0
        assert state["parity"][0]["weight_max_abs_on_matching_selection"] < 1e-6
        if router_cls is GemmaRouter:
            # This fixture distinguishes penalty-on-probabilities from penalty-on-logits.
            wrong = rewrite_torch(native[0], native[2], 2, 2, 1.0)
            assert not torch.equal(wrong.sort(-1).values, got[2].sort(-1).values)
    finally:
        for handle in handles:
            handle.remove()


def test_penalty_does_not_change_gate_probabilities():
    z = logits_fixture()
    _, ids = select(z, 4, 2, 2, load_penalty=1.0)
    weights = gate_weights(z, ids, renormalize=True)
    probs = z.softmax(-1).gather(1, ids.long())
    torch.testing.assert_close(weights, probs / probs.sum(-1, keepdim=True))


def test_arm_grid_and_legacy_interface():
    names = [a[0] for a in moe.arms_of(8, (6,), (0, 1))]
    assert names == ["baseline", "baseline2", "noop", "placement_6", "placement_6p1", "random_6"]
    assert all(len(a) == 3 for a in moe.arms_of(8, (6, 4, 2)))
    with pytest.raises(ValueError):
        moe.arms_of(8, (6,), (-1,))


def test_constant_nonzero_delta_cannot_falsely_pass_noop():
    base = [1.0] * 32
    residual = accuracy.paired_summary([1.25] * 32, base)
    assert residual["t"] is None
    assert residual["zero_se_nonzero_delta"]
    assert residual["dnll_ci95"] == [0.25, 0.25]
    common = {"arm": "baseline", "changed_frac": 0.0, "per_window": base}
    results = {"baseline2": common,
               "noop": {**common, "arm": "placement", "per_window": [1.25] * 32}}
    checks = accuracy.sanity_checks(results)
    assert checks["noop_selection_exact"]
    assert not checks["noop_nll_within_resolution"]
    assert not checks["passed"]


def test_paired_ci_uses_window_differences():
    baseline = np.linspace(1, 10, 32)
    values = baseline + np.linspace(0.01, 0.03, 32)
    summary = accuracy.paired_summary(values, baseline)
    assert summary["dnll"] == pytest.approx(0.02)
    assert 0 < summary["dnll_ci95"][0] < 0.02 < summary["dnll_ci95"][1] < 0.04


def test_target_token_file_provenance(tmp_path, monkeypatch):
    source = tmp_path / "tokens.npy"
    np.save(source, np.arange(20, dtype=np.int32))
    monkeypatch.setattr(accuracy, "TOKEN_IDS_FILE", str(source))
    monkeypatch.setattr(accuracy, "TEXT_FILE", None)
    ids, metadata = accuracy.token_ids(12)
    assert ids.tolist() == list(range(12))
    assert metadata["source_sha256"] == accuracy.file_sha256(source)
    assert metadata["used_input_tokens"] == 12
    assert metadata["available_model_tokens"] == 20
    assert metadata["token_ids_require_model_vocabulary"]
    with pytest.raises(ValueError, match="need 21"):
        accuracy.token_ids(21)


def test_plain_logit_router_forces_production_selection():
    gate = torch.nn.Linear(16, 16, bias=False)
    with torch.no_grad():
        gate.weight.copy_(torch.eye(16))
    model = TinyModel(gate)
    z = logits_fixture()
    state, handles = moe.install(model, 16, 4, 16, 2)
    try:
        assert torch.equal(model(z), z)
        state.update(arm="placement", keep=2, load_penalty=1.0)
        masked = model(z)
        got_ids = masked.topk(4, -1).indices
        _, expected_ids = select(z, 4, 2, 2, load_penalty=1.0)
        assert torch.equal(got_ids.sort(-1).values, expected_ids.long().sort(-1).values)
        expected_w = gate_weights(z, got_ids, renormalize=True)
        torch.testing.assert_close(masked.softmax(-1).gather(1, got_ids), expected_w)
    finally:
        for handle in handles:
            handle.remove()


def test_text_file_uses_target_tokenizer(tmp_path, monkeypatch):
    from types import SimpleNamespace
    source = tmp_path / "text.txt"
    text = "Natural text corpus with multiple windows."
    source.write_text(text)
    class Tokenizer:
        def __call__(self, value, add_special_tokens):
            assert value == text
            assert not add_special_tokens
            return SimpleNamespace(input_ids=[ord(c) for c in value])
    monkeypatch.setattr(accuracy, "TEXT_FILE", str(source))
    monkeypatch.setattr(accuracy, "TOKEN_IDS_FILE", None)
    monkeypatch.setattr(accuracy.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    ids, provenance = accuracy.token_ids(12)
    assert ids.tolist() == [ord(c) for c in text[:12]]
    assert provenance["mode"] == "utf8_text"
    assert provenance["tokenizer_class"] == "Tokenizer"
    assert provenance["source_sha256"] == accuracy.file_sha256(source)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA GPUs and Triton")
def test_hook_launches_on_layer_device_and_restores_current_device():
    """HF auto-sharding can run a CUDA 1 router while CUDA 0 remains current."""
    pytest.importorskip("triton")
    layer_device = torch.device("cuda:1")
    model = TinyModel(QwenRouter()).to(layer_device)
    z = logits_fixture().to(layer_device)
    state, handles = moe.install(model, 16, 4, 16, 2, parity_tokens=2)
    try:
        with torch.cuda.device(0), torch.no_grad():
            # No-op reaches the sampled parity selector, not the treatment selector.
            state.update(arm="placement", keep=4, load_penalty=0.0, parity={})
            noop = model(z)
            torch.cuda.synchronize(layer_device)
            assert noop[2].device == layer_device
            assert state["parity"][0]["rows"] == 2
            assert torch.cuda.current_device() == 0
            # Balanced routing exercises the full treatment launch on the other GPU.
            state.update(keep=2, load_penalty=1.0, parity={})
            got = model(z)
            torch.cuda.synchronize(layer_device)
            ref_ids = rewrite_torch(z, z.topk(4, -1).indices, 2, 2, 1.0)
            assert torch.equal(got[2].sort(-1).values, ref_ids.sort(-1).values)
            expected_w = gate_weights(z, got[2], renormalize=True)
            torch.testing.assert_close(got[1], expected_w, rtol=1e-6, atol=1e-7)
            assert got[1].device == layer_device
            assert state["parity"][0]["rows"] == 2
            assert torch.cuda.current_device() == 0
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("router_cls", [QwenRouter, GemmaRouter])
def test_actual_production_noop_matches_native(router_cls):
    z = logits_fixture()
    model = TinyModel(router_cls())
    native = model(z)
    state, handles = moe.install(model, 16, 4, 16, 2, parity_tokens=2)
    try:
        state.update(arm="production_noop", keep=4, load_penalty=0.0)
        got = model(z)
        assert torch.equal(got[2].sort(-1).values, native[2].sort(-1).values)
        torch.testing.assert_close(sorted_weights(got[1], got[2]),
                                   sorted_weights(native[1], native[2]),rtol=1e-6,atol=1e-7)
        assert state["parity"][0]["selection_mismatch_rows"] == 0
        assert state["raw_sources"][0] == ("native_projection_logits" if router_cls is GemmaRouter
                                            else "native_router_logits")
    finally:
        for h in handles:h.remove()


def test_bos_prepends_context_preserving_exact_raw_targets():
    from types import SimpleNamespace
    raw = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.int64)
    class Model:
        device = torch.device("cpu")
        def __init__(self):
            self.inputs = []
        def __call__(self, input_ids):
            self.inputs.append(input_ids.clone())
            # Each raw-token logit row depends only on that token, so a prepended
            # BOS changes context length but cannot change which targets we score.
            logits = torch.zeros((*input_ids.shape, 6))
            logits.scatter_(-1, ((input_ids + 1) % 6).unsqueeze(-1), 3.0)
            return SimpleNamespace(logits=logits)
    native = Model(); with_bos = Model()
    a = accuracy.per_window_nll(native, raw)
    b = accuracy.per_window_nll(with_bos, raw, bos_token_id=5)
    assert a == b
    assert torch.equal(native.inputs[0], torch.from_numpy(raw))
    assert torch.equal(with_bos.inputs[0][:, 1:], torch.from_numpy(raw))
    assert with_bos.inputs[0][:, 0].tolist() == [5, 5]
