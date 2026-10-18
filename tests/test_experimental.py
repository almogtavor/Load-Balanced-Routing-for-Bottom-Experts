"""CPU tests for explicit selector configuration and baseline fallbacks.

GPU semantics, ties and performance are checked by kernel_candidates.py against
unchanged direct production launches, rather than mocked CUDA operations.
"""
from __future__ import annotations

import importlib
import logging

import pytest
import torch

from placement_router import experimental
from placement_router import kernel


def test_legacy_cuda_launch_rejects_tile_kernel_before_touching_buffers():
    with pytest.raises(NotImplementedError, match="older per-token kernel"):
        experimental.launch_preallocated(None, None, None, 8, 6, 4)


@pytest.mark.parametrize("name", ["warp1", "rows4", "rows8w8", "rows16_loop", "rows4_tl",
                                  "warp1_direct", "rows4_direct"])
def test_cpu_fallback_keeps_padding_scale_and_id_contract(name):
    torch.manual_seed(42)
    logits = torch.randn(64, 19).t()  # noncontiguous [19,64]
    padding = torch.arange(19).remainder(3) == 0
    scale = torch.linspace(0.0, 2.0, 64)
    args = dict(k=8, keep=6, n_ranks=4, is_padding=padding, scale=scale,
                indices_dtype=torch.int64, load_penalty=1.0)
    expected = kernel.select(logits, **args)
    actual = experimental.configured_selector(name)(logits, **args)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert actual[1].dtype == torch.int64
    assert torch.equal(actual[1][padding], torch.full((int(padding.sum()), 8), -1))


def test_configuration_is_read_once_and_logs_identity(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger=kernel.__name__):
        monkeypatch.setenv("PLACEMENT_EXPERIMENTAL_SELECTOR", "warp1")
        importlib.reload(kernel)
        try:
            assert kernel._EXPERIMENTAL_IMPL is experimental.select_onewarp
            assert "experimental selector=warp1" in caplog.text
            messages = len(caplog.records)
            monkeypatch.setenv("PLACEMENT_EXPERIMENTAL_SELECTOR", "rows4")
            logits = torch.randn(2, 64)
            kernel.select(logits, 8, 6, 4)
            kernel.select(logits, 8, 6, 4)
            assert kernel._EXPERIMENTAL_IMPL is experimental.select_onewarp
            assert len(caplog.records) == messages
        finally:
            monkeypatch.delenv("PLACEMENT_EXPERIMENTAL_SELECTOR", raising=False)
            importlib.reload(kernel)
    assert kernel._EXPERIMENTAL_IMPL is None


@pytest.mark.parametrize("name", ["warp2", "rows2", "rows4w2", "unknown"])
def test_unknown_configuration_fails_explicitly(name):
    with pytest.raises(ValueError, match="unknown experimental selector"):
        experimental.configured_selector(name)


def test_baseline_override_preserves_original_dispatch(monkeypatch):
    monkeypatch.setenv("PLACEMENT_EXPERIMENTAL_SELECTOR", "baseline")
    try:
        importlib.reload(kernel)
        assert kernel._EXPERIMENTAL_IMPL is None
    finally:
        monkeypatch.delenv("PLACEMENT_EXPERIMENTAL_SELECTOR", raising=False)
        importlib.reload(kernel)


def test_configuration_validation_is_cached_across_token_counts():
    experimental._validate_config.cache_clear()
    selector = experimental.configured_selector("rows4")
    selector(torch.randn(2, 64), 8, 6, 4, load_penalty=1.0)
    selector(torch.randn(9, 64), 8, 6, 4, load_penalty=1.0)
    info = experimental._validate_config.cache_info()
    assert info.misses == 1
    assert info.hits == 1


@pytest.mark.parametrize("name", ["warp1_direct", "rows4_direct"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.uint32])
def test_direct_fallback_ids_and_unsigned_sentinel(name, dtype):
    z = torch.zeros(3, 64)
    padding = torch.tensor([True, False, True])
    scale = torch.linspace(0.5, 1.5, 64)
    expected = kernel.select(z, 8, 6, 4, padding, scale, dtype, 1.0)
    actual = experimental.configured_selector(name)(z, 8, 6, 4, padding, scale, dtype, 1.0)
    assert actual[1].dtype == dtype
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_direct_rejects_unsupported_dtype_explicitly():
    with pytest.raises(ValueError, match="direct ID output supports"):
        experimental.configured_selector("rows4_direct")(
            torch.randn(2, 64), 8, 6, 4, indices_dtype=torch.float32)


@pytest.mark.parametrize("args", [dict(keep=9), dict(n_ranks=3), dict(load_penalty=-1.0)])
def test_invalid_candidate_parameters_are_clear(args):
    kwargs = dict(k=8, keep=6, n_ranks=4, load_penalty=1.0)
    kwargs.update(args)
    with pytest.raises(ValueError):
        experimental.select_multirow(torch.randn(2, 64), **kwargs)
