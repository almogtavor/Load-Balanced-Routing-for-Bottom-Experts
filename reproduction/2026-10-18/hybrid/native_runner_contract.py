"""Exact installed v0.31 V2 runner contract, verified before any phase claims.

Hashes are from the retained CPU-only image readback native-runner-diagnosis-v1.
No runtime flag changes the default runner; the actual selected self must be V2.
"""
from __future__ import annotations
import hashlib
import inspect
from pathlib import Path

NATIVE = {
    "config/vllm.py": "057b0a7f80db28522ee46b41f9ae5bd7a0c1998a08e111c5a6ddcb8c08f07f8d",
    "v1/worker/gpu_worker.py": "43bd01925cd84f7d8a1d4a2ec30e289014a485fe4492a5343cbc893e8ed2f518",
    "v1/worker/gpu/model_runner.py": "725c47114e148ade3a26b4cc9c5f7295588fdd387afc3cb5fdb2d16513a1bb5c",
    "v1/core/sched/output.py": "972e0bc8e7af1f4262ac73f660f6b8f798ba780f86c031a9c8a7f10cbfaad3a8",
}
RUNNER = "vllm.v1.worker.gpu.model_runner.GPUModelRunner"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def installed_sources(root):
    root = Path(root).resolve()
    paths = {name: str(root / name) for name in NATIVE}
    require(all(digest(path) == NATIVE[name] for name, path in paths.items()),
            "Actual installed V2 runner/config/worker source differs from image readback")
    return {"root": str(root), "files": paths, "sha256": dict(NATIVE)}


def callable_evidence(fn):
    base = inspect.unwrap(fn)
    signature = inspect.signature(fn)
    return {"module": fn.__module__, "qualname": fn.__qualname__,
            "actual_code_filename": str(Path(fn.__code__.co_filename).resolve()),
            "unwrapped_code_filename": str(Path(base.__code__.co_filename).resolve()),
            "unwrapped_source_sha256": digest(base.__code__.co_filename),
            "signature": str(signature), "parameter_names": list(signature.parameters),
            "dummy_run_default": signature.parameters["dummy_run"].default,
            "is_profile_default": signature.parameters["is_profile"].default}


def actual_runner(self, runner_class, execute, sources):
    config = self.vllm_config
    name = type(self).__module__ + "." + type(self).__qualname__
    require(type(self) is runner_class and name == RUNNER and config.use_v2_model_runner is True,
            "Actual selected runner is not exact native V2")
    value = callable_evidence(execute)
    expected = sources["files"]["v1/worker/gpu/model_runner.py"]
    require(value["unwrapped_code_filename"] == expected
            and value["unwrapped_source_sha256"] == NATIVE["v1/worker/gpu/model_runner.py"],
            "Original actual V2 execute callable/source differs")
    return {"class": name, "use_v2_model_runner": True,
            "class_source": expected, "native_sources": sources,
            "original_execute": value, "runner_selection": "observed native default; no V2 environment override"}


def execute_flags(signature, self, scheduler_output, args, kwargs):
    bound = signature.bind(self, scheduler_output, *args, **kwargs)
    bound.apply_defaults()
    dummy, profile = bound.arguments.get("dummy_run"), bound.arguments.get("is_profile")
    require(type(dummy) is bool and type(profile) is bool, "Native V2 dummy/profile argument contract differs")
    return {"dummy_run": dummy, "is_profile": profile}
