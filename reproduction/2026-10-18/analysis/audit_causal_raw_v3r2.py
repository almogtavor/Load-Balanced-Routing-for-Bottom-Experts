#!/usr/bin/env python3
"""Offline fresh-owner full-matrix adapter for unchanged causal-v2 analyzers.

No server, process-control or network operation is called. Six training and
three held-out launches must all be closed before any held-out score is read.
Functions from SHA-pinned modules are cloned into private namespaces; only
read/hash/write I/O is rebound. The nested native proof gate uses that same
namespace. Scientific functions and archived remote path strings are unchanged.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile
import traceback
import types

STUDY = Path(__file__).resolve().parents[1]
OLD_STUDY = STUDY.parent / "paper-followup-20261018"
OLD_AUDIT = OLD_STUDY / "audit"
# Primary old raw utilities remain unchanged and explicitly source pinned.
for _name, _digest in {
    "audit_prefill_raw.py": "53c7645f8729bca04b7d5902dcd0f2841325f19e24aeef281add04c7da216cbb",
    "paired_prefill_statistics.py": "8aeac395be882d8f4611fa89d16b2ae57d077e8ba7e548f9b4f9da2b6a6c8f62",
    "repeated_quality_statistics.py": "e1b49098347c25f50bddc6c6841e944e6cf3c613889d99990f0993bd1458b6f7",
}.items():
    if hashlib.sha256((OLD_AUDIT / _name).read_bytes()).hexdigest() != _digest:
        raise ValueError("Frozen independent audit utility source changed: " + _name)
sys.path.insert(0, str(OLD_AUDIT))
from audit_prefill_raw import SOURCE, COMMIT, REVISION, local_file, require, sha
HELPERS = STUDY / "causal/package"
REMOTE = "/work/results/causal-v3"
TOOLS = "/work/hybrid-tools/causal-v3"
ARMS = ("stock", "keep6p8", "keep4p8")
TRAINING = [(arm, budget) for budget in (16384, 1024) for arm in ARMS]
UID = "891d6a54-3584-48bd-9712-fa7095ef09b3"
STOP = 1792349467
PLAN_SHA = "e65158616bdbba6b83c0ae5539ac4b8bcec8844cd12af77745071884e93c8fb0"
OUTER_REQUEST_SHA = "23a329386d08bb60573e89e330548517b5d77e70e2d0d0f23045e101d3bb8068"
LAUNCH_SHA = "6d01fe5dfbd508165b576041acd074c66be5f6d6b5601e45223b7fa892b95e59"
FINALIZER_SHA = "e65584df654bf9a956322ae3e13f65d657b26775ab4a79156bce9db968fba4d2"
FULL_SHA = "6c3cdfb53702f5b1e07f8d38a9770b5da45b5d419b5ad81d5b9e81908c0cf63b"
PREFLIGHT_SHA = "183888c4a3d0402ab97806d453449dbf740e445901b80ee3d5e156ffc7dd7017"
QUALITY_AUDITOR_SHA = "07213665e813e0c14d9d57956863b42e50d1ba148425e9ec21729542d741ad1f"
FROZEN = {
    "common_v2.py": "9e64f7f704346115f20387240a2a20faba0914b8bc33283f688239679aef8700",
    "common_v3.py": "77fa869f37613b61288e6289c42e6cff8fe3a2d62ac25275217264de679ae981",
    "causal_probe.py": "2d67763c5fffb8154bd95dd10056528943398370313153566f064c13db1791de",
    "causal_contract.py": "66e387eaa33a54e2c2cf780dc092a02af22e6cbe1683065440dd8883acabbee2",
    "causal_worker_probe.py": "5689631a81bc21a00f3baefe235f9db06fa530d06a563c54088c0198820615dc",
    "heldout_causal_v2.py": "229aab2e4cb30acbc1e7c03060625a56da6777bafbe7ef763d74d3b5f737b43c",
    "causal_driver_v1.py": "ff27eca971679a8bf24bbee01505ab1f96d74a9855a9fed07858eb266bbec1df",
    "frozen_causal_outer_v1.py": "bf480d27b7d06eecbcf0f1c90bbe9463db081e49e5dbe58de3b2e213ecf15146",
    "causal_outer_v3.py": "1be4f6d14b695365df195d8a1e2294bc878ed62269534c951b9769bbf4de4934",
    "causal_driver_v3.py": "a46016c3090aaf6a570b5051d9b74026683e0d575ee6eff84f108d2936546aec",
    "fresh_owner.py": "897a5c34b8c5eb41dd4f84e55fb61a5e8fa5ac114c51abb89e377def900523cd",
}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


class BoundIO:
    """Explicit remote bindings, archive-local reads and a new temporary output."""
    def __init__(self, files, bindings):
        self.files = files.resolve()
        self.binding_path = bindings.resolve()
        self.bindings = json.loads(bindings.read_bytes())
        require(isinstance(self.bindings, dict), "Bindings must be an object")
        self.checked, self.work = {}, None
        self.helper_files = {str((HELPERS / n).resolve()): TOOLS + "/" + n for n in FROZEN}
        for remote, relative in self.bindings.items():
            p = PurePosixPath(remote)
            require(p.is_absolute() and str(p) == remote and ".." not in p.parts,
                    "Unsafe external remote binding")
            local_file(self.binding_path.parent, relative)

    def resolve(self, value):
        raw, p = str(value), Path(value)
        require(p.is_absolute() and ".." not in p.parts and str(p) == raw,
                "Noncanonical absolute input path")
        if raw.startswith("/work/results/"):
            relative = raw.removeprefix("/work/results/")
            candidate = self.files / relative
            if candidate.is_file() or raw.startswith(REMOTE + "/"):
                return local_file(self.files, relative), raw
        if raw in self.bindings:
            return local_file(self.binding_path.parent, self.bindings[raw]), raw
        if p.is_relative_to(self.files):
            relative = p.relative_to(self.files).as_posix()
            return local_file(self.files, relative), "/work/results/" + relative
        if raw in self.helper_files:
            require(sha(p) == FROZEN[p.name], "Frozen local helper changed")
            return p, self.helper_files[raw]
        if self.work is not None and p.is_relative_to(self.work):
            return local_file(self.work, p.relative_to(self.work).as_posix()), None
        raise ValueError("Unbound offline input: " + raw)

    def file_sha(self, value):
        p, remote = self.resolve(value)
        digest = sha(p)
        if remote is not None:
            self.checked[remote] = {"sha256": digest, "bytes": p.stat().st_size}
        return digest

    def read(self, value):
        self.file_sha(value)
        return json.loads(self.resolve(value)[0].read_bytes())

    def write_new(self, value, data):
        p = Path(value)
        require(self.work is not None and p.is_absolute() and p.is_relative_to(self.work),
                "Frozen replay attempted a write outside the new temporary audit directory")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("xb") as stream:
            stream.write(encoded(data) + b"\n")


def private_namespace(module, io):
    """Clone all module-native functions together, including nested proof gates."""
    namespace = dict(vars(module))
    for name, value in vars(module).items():
        if isinstance(value, types.FunctionType) and value.__globals__ is vars(module):
            clone = types.FunctionType(value.__code__, namespace, value.__name__,
                                       value.__defaults__, value.__closure__)
            clone.__kwdefaults__ = value.__kwdefaults__
            namespace[name] = clone
    namespace.update(read=io.read, file_sha=io.file_sha, write_new=io.write_new)
    def network_forbidden(*args, **kwargs):
        raise ValueError("Network/model call is forbidden during offline raw audit")
    namespace["post"] = network_forbidden
    return types.SimpleNamespace(**namespace)


@contextlib.contextmanager
def frozen_modules(io):
    names = ["common_v2", "common_v3", "causal_probe", "causal_contract", "heldout_causal_v2", "fresh_owner"]
    for name, digest in FROZEN.items():
        require(sha(HELPERS / name) == digest, "Frozen scientific source differs: " + name)
    before = {n: sys.modules.get(n) for n in names}
    old_path, old_bytecode = list(sys.path), sys.dont_write_bytecode
    try:
        for name in names:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(HELPERS)); sys.dont_write_bytecode = True
        modules = {n: importlib.import_module(n) for n in names}
        for name, value in modules.items():
            require(Path(value.__file__).resolve() == (HELPERS / (name + ".py")).resolve(),
                    "Frozen scientific import resolved elsewhere")
        yield {n: private_namespace(value, io) for n, value in modules.items()}
    finally:
        sys.path[:] = old_path; sys.dont_write_bytecode = old_bytecode
        for name in names:
            sys.modules.pop(name, None)
            if before[name] is not None:
                sys.modules[name] = before[name]


def idle(value):
    require(value.get("returncode") == 0 and not value.get("stdout", "x").strip(),
            "Closed whole-host GPU-idle receipt absent")


def mode_gate(io, plan_sha, mode):
    cells = TRAINING if mode == "training" else [(arm, 16384) for arm in ARMS]
    root = REMOTE + "/" + mode
    receipt = io.read(root + "/receipt.json")
    require(receipt == io.read(root + "/status.json") and receipt.get("complete") is True
            and receipt.get("cleanup_complete") is True and not receipt.get("error")
            and not receipt.get("cleanup_error") and receipt.get("active_servers", {}) == {}
            and receipt["mode"] == mode and receipt["plan_sha256"] == plan_sha
            and len(receipt["launches"]) == len(cells), "Full causal mode is not closed: " + mode)
    idle(receipt["emergency_closure"]["gpu_idle"])
    result = []
    for index, ((arm, budget), row) in enumerate(zip(cells, receipt["launches"])):
        directory = root + f"/cell{index:02d}-{arm}-budget{budget}"
        require(row.get("complete") is True and row["arm"] == arm and row["budget"] == budget
                and row["receipt"] == directory + "/launch.json", "Causal matrix cell order/path differs")
        require(io.file_sha(row["receipt"]) == row["sha256"], "Immutable causal launch bytes differ")
        launch = io.read(row["receipt"])
        require(launch.get("complete") is True and launch.get("workload_complete") is True
                and launch["finalization_errors"] == [] and not ({"error", "workload_error", "analysis_error"} & set(launch))
                and launch["directory"] == directory and launch["arm"] == arm and launch["budget"] == budget
                and launch["mode"] == mode and launch["plan_sha256"] == plan_sha,
                "Causal launch incomplete or changed")
        require(launch["closure"]["causal"]["remaining_members"] == [], "Owned causal server remains")
        idle(launch["closure"]["gpu_idle"])
        require(launch["analysis"]["path"] == directory + "/analysis.json"
                and io.file_sha(launch["analysis"]["path"]) == launch["analysis"]["sha256"],
                "Closed remote analysis bytes differ")
        result.append(launch)
    actual = set(io.resolve(root + "/receipt.json")[0].parent.glob("cell*/launch.json"))
    expected = {io.resolve(v["directory"] + "/launch.json")[0] for v in result}
    require(actual == expected, "Unexpected or missing causal launch directory")
    return receipt, result


def transport_gate(io, folder, *, monotonic=False):
    local = io.resolve(folder + "/result.json")[0].parent
    require(not list(local.rglob("*.error.json")) and not (local / "failure.json").exists()
            and not (local / "partial-scores.json").exists(), "Failed/partial causal HTTP artifacts")
    requests = sorted((local / "raw").glob("*.request.json")) if (local / "raw").exists() else sorted(local.glob("rank*.request.json"))
    require(requests, "No raw causal HTTP requests")
    if (local / "raw").exists():
        require(all(p.is_file() and p.suffix == ".json" for p in (local / "raw").iterdir())
                and len(list((local / "raw").iterdir())) == len(requests) * 3,
                "Unexpected raw causal HTTP inventory")
    previous_unix = previous_mono = 0
    first_unix = None
    for path in requests:
        stem = str(path).removesuffix(".request.json")
        request, response = io.read(path), io.read(stem + ".transport.json")
        response_path = stem + ".response.json"
        require(response.get("http_status") == 200 and response.get("response_complete") is True
                and response["response_sha256"] == io.file_sha(response_path)
                and response["response_bytes"] == io.resolve(response_path)[0].stat().st_size,
                "Incomplete or changed raw HTTP transport")
        body, value = request["body"], io.read(response_path)
        prompts = body["prompt"] if isinstance(body["prompt"][0], list) else [body["prompt"]]
        choices = value.get("choices", [])
        require("error" not in value and len(choices) == len(prompts)
                and all(type(c.get("index")) is int for c in choices)
                and {c["index"] for c in choices} == set(range(len(prompts))), "Raw response choice/error schema differs")
        choices = sorted(choices, key=lambda c: c["index"])
        for prompt, choice in zip(prompts, choices):
            require(choice.get("prompt_token_ids") == prompt and choice.get("finish_reason") in ("length", "stop")
                    and isinstance(choice.get("token_ids"), list)
                    and all(type(t) is int and t >= 0 for t in choice["token_ids"])
                    and len(choice["token_ids"]) <= body["max_tokens"], "Raw consumed/output token IDs differ")
            if body.get("ignore_eos") is True and body["max_tokens"] == 1:
                require(len(choice["token_ids"]) == 1, "Target-position prefix request returned no actual generated position")
        usage = value.get("usage", {})
        expected_usage = {"prompt_tokens": sum(map(len, prompts)),
                          "completion_tokens": sum(len(c["token_ids"]) for c in choices)}
        expected_usage["total_tokens"] = sum(expected_usage.values())
        require(all(type(usage.get(k)) is int and usage[k] == v for k, v in expected_usage.items()),
                "Raw response usage counts differ")
        submitted, returned = response["submitted_unix_ns"], response["returned_unix_ns"]
        require(type(submitted) is int and type(returned) is int and submitted >= previous_unix
                and returned >= submitted and request["submitted_unix_ns"] == submitted,
                "Raw HTTP Unix intervals overlap or differ")
        first_unix = submitted if first_unix is None else first_unix
        previous_unix = returned
        if monotonic:
            submitted, returned = response["submitted_monotonic_ns"], response["returned_monotonic_ns"]
            require(type(submitted) is int and type(returned) is int and submitted >= previous_mono
                    and returned >= submitted and request["submitted_monotonic_ns"] == submitted,
                    "Held-out primary HTTP intervals are not serial")
            previous_mono = returned
    return {"requests": len(requests), "first_submitted_unix_ns": first_unix,
            "last_returned_unix_ns": previous_unix, "monotonic_serial_checked": monotonic}


def server_command(plan, budget):
    return [plan["vllm"], "serve", plan["model"], "--served-model-name", "task-model", "--host", "127.0.0.1",
            "--port", "8040", "--data-parallel-rpc-port", "29690", "--data-parallel-size", "4",
            "--tensor-parallel-size", "1", "--pipeline-parallel-size", "1", "--enable-expert-parallel",
            "--all2all-backend", "deepep_high_throughput", "--dtype", "bfloat16", "--expert-placement-strategy", "linear",
            '--compilation-config={"cudagraph_mode":"NONE"}', "--max-model-len", "8192", "--seed", "0",
            "--max-num-batched-tokens", str(budget), "--max-num-seqs", "256", "--enable-chunked-prefill",
            "--no-enable-prefix-caching", "--logprobs-mode", "raw_logprobs", "--gpu-memory-utilization", ".88"]


def native_gate(io, modules, plan, launch, work):
    directory, arm, budget = launch["directory"], launch["arm"], launch["budget"]
    require(io.file_sha(directory + "/contract.json") == launch["native_contract_sha256"], "Native contract changed")
    output = work / (launch["mode"] + directory.rsplit("/", 1)[1] + "-contract.json")
    modules["causal_contract"].build(types.SimpleNamespace(checkpoint_proof=Path(plan["checkpoint_proof"]),
        source_proof=Path(plan["source_proof"]), probe_dir=io.resolve(directory + "/contract.json")[0].parent / "probe",
        arm=arm, budget=budget, output=output))
    rebuilt = json.loads(output.read_bytes())
    # Only physical probe path keys move during archive extraction.
    rebuilt["probe_files_sha256"] = {directory + "/probe/" + Path(p).name: h for p, h in rebuilt["probe_files_sha256"].items()}
    contract = io.read(directory + "/contract.json")
    require(rebuilt == contract and launch["probe_files_sha256"] == contract["probe_files_sha256"],
            "Native contract regeneration differs")
    workers = [io.read(p) for p in contract["probe_files_sha256"]]
    keep = {"stock": None, "keep6p8": 6, "keep4p8": 4}[arm]
    for v in workers:
        require(set(v["registered_layer_names"]) == {f"model.layers.{i}.mlp.experts" for i in range(48)}
                and len(v["registered_layer_names"]) == 48 and v["hidden_size"] == 2048
                and v["placement_keep"] == keep and v["load_penalty"] == (None if keep is None else 8.)
                and v["load_tile"] == 16 and v["env_cuda_visible_devices"] == "0,1,2,3", "Actual native router metadata differs")
        for q in v["router_contracts"]:
            require(q["expected"] is True and q["top_k"] == 8 and q["experts"] == 128
                    and q["renormalize"] is True and q["scoring_func"] == "softmax" and q["eplb"] is False
                    and q["actual_keep"] == keep and q["routing_module"] == (
                        "vllm.model_executor.layers.fused_moe.router.fused_topk_router" if keep is None
                        else "placement_router.vendor.fused_topk_router"), "Actual native router contract differs")
        require(v["device_index"] == v["ep_rank"] and "H100" in v["gpu_name"], "Actual native rank/device differs")
        for name in v["source_paths"]:
            require(v["source_paths"][name] == plan["site_root"] + "/" + name.removeprefix("src/"),
                    "Native imported source path differs")
    return contract


def warmup_gate(io, modules, plan, launch):
    directory = launch["directory"]
    result = io.read(directory + "/warmup/result.json")
    require(result == {"complete": True, "training_only": True, "requests": 4}, "Identity warmup differs")
    rows = [r for r in io.read(plan["preflight_inputs"]) if r["task"] == "gsm8k" and r["split"] == "train"][:4]
    require(len(rows) == 4, "Four canonical warmup requests missing")
    for rank, row in enumerate(rows):
        stem = directory + f"/warmup/rank{rank}"
        request = io.read(stem + ".request.json")
        expected = modules["common_v2"].base_body("task-model", row["token_ids"],
            directory.rsplit("/", 1)[1] + f"-identity-{rank}")
        require(request["body"] == expected and request["sha256"] == hashlib.sha256(encoded(expected)).hexdigest()
                and request["headers"] == {"X-data-parallel-rank": str(rank)}, "Actual rank warmup request differs")
        modules["common_v2"].choices(io.read(stem + ".response.json"), [row["token_ids"]])
    inventory = {p.name for p in io.resolve(directory + "/warmup/result.json")[0].parent.iterdir()}
    require(inventory == {"result.json"} | {f"rank{r}.{s}.json" for r in range(4) for s in ("request", "response", "transport")},
            "Unexpected warmup raw inventory")
    return transport_gate(io, directory + "/warmup")


def scientific_replay(module, io, prepared, result, output):
    module.analyze(types.SimpleNamespace(prepared=prepared, result=result, output=output))
    return json.loads(output.read_bytes())


def identity(value):
    return int(value["pid"]), int(value["pgrp"]), str(value["start_ticks"])


def archive_gate(args, io):
    marker = json.loads(args.marker.read_bytes())
    source = marker["source"]
    require(marker.get("final_verified") is True and marker.get("writers_stopped_asserted") is True
            and marker["remote_verification"]["ok"] is True
            and source["root"] == "/work/results" and source["pod_uid"] == UID
            and source["pod"] == "par-gap-causal-h100-20261018" and source["namespace"] == "itay"
            and source["container"] == "experiment"
            and source["oc"] == str(Path.home() / ".local/bin/oc-llmd-roce"), "Actual fresh whole archive required")
    require(Path(marker["extracted_directory"]).resolve() / "files" == io.files
            and Path(marker["extracted_directory"]).resolve().parent == args.marker.resolve().parent,
            "Raw files are outside the exact whole-ACK extraction")
    archive = args.marker.parent / "results.tar.gz"
    require(not archive.is_symlink() and archive.stat().st_size == marker["archive_bytes"]
            and sha(archive) == marker["archive_sha256"], "Whole compressed archive changed")
    freeze = io.read("/work/results/provenance/final-inputs-freeze-v1.json")
    require(freeze.get("complete") is True and freeze["pod_uid"] == UID
            and freeze["writers_stopped"] is True and freeze["gpu_idle"] is True
            and freeze["allocation"]["gpu_count"] == 4
            and freeze["finalizer_sha256"] == FINALIZER_SHA
            and len(freeze["gpu_idle_checks"]) == 2
            and all(r["returncode"] == 0 and r["compute_pids"] == [] for r in freeze["gpu_idle_checks"]),
            "Final native writer/allocated-GPU closure proof differs")
    rows = {r["source"]: r for r in freeze["files"]}
    require(len(rows) == len(freeze["files"]), "Duplicate frozen actual input path")
    for row in freeze["files"]:
        path = local_file(io.files, row["copy"])
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Actual frozen input copy changed")
    return {"archive_marker_sha256": sha(args.marker), "archive_sha256": marker["archive_sha256"],
            "final_freeze_sha256": io.file_sha("/work/results/provenance/final-inputs-freeze-v1.json"),
            "final_freeze": freeze, "whole_verified": True}


def outer_gate(io, plan_sha, quality_sha, training_receipt, heldout_receipt):
    root = "/work/results/causal-outer-v3"
    value = io.read(root + "/receipt.json")
    request_path = "/work/results/provenance/causal-outer-request-v3.json"
    require(io.file_sha(request_path) == OUTER_REQUEST_SHA, "Frozen fresh outer request bytes differ")
    request = io.read(request_path)
    require(request["schema"] == "fresh-four-h100-causal-outer-v3" and request["complete"] is True
            and request["old_partial_results_reused"] is False and request["stop_unix"] == STOP
            and request["pod_active_deadline_seconds"] == 21600 and request["child_cleanup_grace_seconds"] == 720
            and request["files"]["/work/results/provenance/causal-prior/prior-quality9.json"] == quality_sha,
            "Fresh outer owner/protocol/prior provenance differs")
    for path, expected in request["files"].items():
        require(io.file_sha(path) == expected, "Fresh outer immutable input differs: " + path)
    require(value == io.read(root + "/status.json") and value.get("complete") is True
            and value.get("cleanup_complete") is True and value["cleanup_errors"] == []
            and "active_child" not in value and "error" not in value and value["request_sha256"] == OUTER_REQUEST_SHA
            and value["plan_sha256"] == plan_sha, "Fresh causal outer controller is not closed")
    idle(value["final_gpu_idle"])
    path = "/work/results/provenance/causal-outer-launch-v3.json"
    require(io.file_sha(path) == LAUNCH_SHA, "Actual fresh original launch receipt changed")
    launch = io.read(path)
    require(launch["complete"] is True and launch["started"] is True and launch["pod_uid"] == UID
            and launch["outer_request_sha256"] == OUTER_REQUEST_SHA and launch["outer_sha256"] == FROZEN["causal_outer_v3.py"]
            and identity(launch["process"]) == identity(value["controller_process"]) == (137, 137, "154691236"),
            "Original fresh outer controller identity differs")
    require(len(value["stages"]) == len(request["commands"]) == 4, "Fresh outer finite stage inventory differs")
    previous, children = value["started_unix_ns"], set()
    for entry, (name, command) in zip(value["stages"], request["commands"]):
        ident = identity(entry["process"])
        require(entry["stage"] == name and entry["command"] == command and entry.get("complete") is True
                and entry["returncode"] == 0 and entry["cleanup"]["remaining_members"] == []
                and identity(entry["cleanup"]["identity"]) == ident and ident not in children
                and ident[0] == ident[1] and previous <= entry["started_unix_ns"] <= entry["finished_unix_ns"],
                "Fresh outer child closure/order differs")
        children.add(ident); previous = entry["finished_unix_ns"]
        reference = entry["receipt"]
        require(io.file_sha(reference["path"]) == reference["sha256"], "Outer child receipt bytes differ")
        if name in ("training", "heldout"):
            expected = training_receipt if name == "training" else heldout_receipt
            require(reference["path"] == REMOTE + "/" + name + "/receipt.json"
                    and ident == identity(expected["controller_process"])
                    and reference["launches"] == (6 if name == "training" else 3),
                    "Original fresh mode controller/outer binding differs")
        elif name == "plan":
            require(reference["path"] == REMOTE + "/plan.json" and reference["sha256"] == plan_sha,
                    "Outer operational plan binding differs")
        elif name == "freeze-heldout":
            require(reference["path"] == REMOTE + "/heldout-freeze-receipt.json", "Held-out lock binding differs")
        else:
            raise ValueError("Unexpected fresh outer stage")
    admin = io.read(REMOTE + "/freeze-heldout/receipt.json")
    require(admin.get("complete") is True and admin.get("cleanup_complete") is True
            and admin["plan_sha256"] == plan_sha and admin["mode"] == "freeze-heldout"
            and admin["launches"] == [] and "error" not in admin, "Administrative held-out lock stage incomplete")
    final = io.read("/work/results/provenance/final-inputs-freeze-v1.json")
    require(previous <= value["finished_unix_ns"] <= int(final["finished_unix"] * 1_000_000_000),
            "Final native no-live-process freeze precedes original controller closure")
    return {"outer_receipt_sha256": io.file_sha(root + "/receipt.json"),
            "outer_process": value["controller_process"], "finished_unix": value["finished_unix_ns"] / 1e9,
            "original_process_exit_proof": "Reviewed finalizer excludes all non-zombie processes except PID1 and itself in this exact fresh allocation"}


def audit(args):
    io = BoundIO(args.files, args.bindings)
    archived = archive_gate(args, io)
    require(args.plan_sha256 == PLAN_SHA and sha(args.plan) == PLAN_SHA, "Exact operational causal plan SHA differs")
    require(io.resolve(REMOTE + "/plan.json")[0] == args.plan.resolve(), "Plan is not the captured causal root plan")
    plan = io.read(args.plan.resolve())
    require(plan["schema"] == "fresh-four-h100-causal-v3" and plan["output"] == REMOTE and plan["helper_dir"] == TOOLS
            and plan["heldout_arms"] == list(ARMS) and plan["training_cells"] == [list(v) for v in TRAINING]
            and plan["stop_unix"] == STOP and plan["heldout_helper"] == TOOLS + "/heldout_causal_v2.py"
            and plan["training_prepared"] == REMOTE + "/training-prepared"
            and plan["heldout_prepared"] == REMOTE + "/heldout-prepared"
            and plan["training_requests_per_cell"] == 493 and plan["quality_controller"] == {"owner_proof": "/work/results/provenance/causal-fresh-owner-v1.json"}
            and plan["boot_timeout"] == 2400 and plan["training_timeout"] == 3600
            and plan["heldout_timeout"] == 18000 and plan["url"] == "http://127.0.0.1:8040",
            "Causal protocol/configuration differs")
    for path, digest in plan["files"].items():
        require(io.file_sha(path) == digest, "Frozen causal input differs: " + path)
    for name, digest in FROZEN.items():
        if name not in ("frozen_causal_outer_v1.py", "causal_outer_v3.py", "causal_driver_v1.py"):
            require(plan["files"][TOOLS + "/" + name] == digest, "Remote causal helper identity differs")
    require(io.file_sha(plan["full_inputs"]) == FULL_SHA and io.file_sha(plan["preflight_inputs"]) == PREFLIGHT_SHA,
            "Canonical full/training inputs differ")
    quality = json.loads(args.quality_audit.read_bytes())
    require(sha(args.quality_audit) == args.quality_audit_sha256 and quality.get("complete") is True
            and quality.get("all_checks_pass") is True and quality["fresh_launches"] == 9
            and quality["repeat_auditor_sha256"] == QUALITY_AUDITOR_SHA
            and quality["driver_status_sha256"] == io.file_sha(plan["quality_receipt"]),
            "Complete independently raw-audited public9 prerequisite differs")
    proof = io.read(plan["source_proof"]); checkpoint = io.read(plan["checkpoint_proof"])
    require(plan["allocation"]["verified_live"] is True and plan["allocation"]["uid"] == proof["pod_identity"]["uid"] == UID
            and plan["allocation"]["gpu_count"] == 4 and "H100" in plan["allocation"]["gpu_name"]
            and all(plan["allocation"][key] == proof["pod_identity"][key] for key in ("namespace", "pod", "node", "image"))
            and proof["source_commit"] == COMMIT and plan["source_identity"]["commit"] == COMMIT
            and proof["source_root"] == plan["source_root"] and proof["site_root"] == plan["site_root"]
            and plan["source_identity"]["proof_sha256"] == io.file_sha(plan["source_proof"])
            and checkpoint["canonical_identity_verified"] is True and checkpoint["revision"] == REVISION
            and checkpoint["weights_verified"] == checkpoint["safetensors"] and len(checkpoint["safetensors"]) == 16
            and plan["checkpoint_identity"]["proof_sha256"] == io.file_sha(plan["checkpoint_proof"]),
            "Actual allocation/source/checkpoint identity differs")
    for name, digest in SOURCE.items():
        require(proof["installed_package"]["installed_source_sha256"][name] == digest
                and plan["source_identity"]["files"]["public:" + name] == digest
                and plan["source_identity"]["files"]["installed:" + name] == digest,
                "Source lineage/actual installed identity differs")
        require(io.file_sha(plan["source_root"] + "/" + name) == digest
                and io.file_sha(plan["site_root"] + "/" + name.removeprefix("src/")) == digest,
                "Source bytes differ")
    # No held-out analysis, score or response is read until both matrices close.
    training_receipt, training = mode_gate(io, args.plan_sha256, "training")
    heldout_receipt, heldout = mode_gate(io, args.plan_sha256, "heldout")
    outer = outer_gate(io, args.plan_sha256, quality["driver_status_sha256"], training_receipt, heldout_receipt)
    freeze = io.read(REMOTE + "/heldout-freeze-receipt.json")
    require(freeze.get("complete") is True and freeze["plan_sha256"] == args.plan_sha256
            and freeze["training_receipt_sha256"] == io.file_sha(REMOTE + "/training/receipt.json")
            and freeze["protocol_sha256"] == io.file_sha(REMOTE + "/heldout-prepared/protocol.json")
            and freeze["selected_requests_sha256"] == io.file_sha(REMOTE + "/heldout-prepared/selected-requests.json")
            and freeze["review_sha256"] == io.file_sha(plan["heldout_review"])
            and freeze["no_heldout_model_queries_before_freeze"] is True
            and max(v["finished_unix_ns"] for v in training) <= freeze["created_unix_ns"],
            "Post-training held-out lock differs")
    owner = io.read(plan["owner_proof"])
    identities, analyses, previous_finish = set(), {}, int(owner["pod_created_unix"] * 1_000_000_000)
    with tempfile.TemporaryDirectory(prefix="causal-raw-audit-") as temporary:
        io.work = Path(temporary).resolve()
        with frozen_modules(io) as modules:
            modules["fresh_owner"].fresh_owner_gate(plan["owner_proof"], "/work/results/allocation-proof.json",
                plan["source_proof"], plan["native_manifest"], plan["quality_receipt"])
            # Reconstruct the preparation from the canonical source inputs too.
            m = modules["causal_probe"]
            rebuilt_training = io.work / "training-prepared"
            m.prepare(types.SimpleNamespace(prior_preflight=Path(plan["preflight_inputs"]), prior_preflight_sha256=PREFLIGHT_SHA,
                checkpoint_proof=Path(plan["checkpoint_proof"]), source_proof=Path(plan["source_proof"]),
                production_root=Path(plan["source_root"]), output=rebuilt_training))
            require(io.read(rebuilt_training / "protocol.json") == io.read(REMOTE + "/training-prepared/protocol.json"),
                    "Canonical training preparation regeneration differs")
            # Recreate the label-independent subset and exact protocol before
            # reading any held-out analysis contents or raw score responses.
            reconstructed = io.work / "heldout-prepared"
            modules["heldout_causal_v2"].prepare(types.SimpleNamespace(full_requests=Path(plan["full_inputs"]), full_requests_sha256=FULL_SHA,
                training_analyses=[Path(v["analysis"]["path"]) for v in training], checkpoint_proof_sha256=io.file_sha(plan["checkpoint_proof"]),
                source_proof_sha256=io.file_sha(plan["source_proof"]), output=reconstructed))
            modules["fresh_owner"].selected_gate(reconstructed)
            require(io.read(reconstructed / "protocol.json") == io.read(plan["heldout_prepared"] + "/protocol.json")
                    and io.read(reconstructed / "selected-requests.json") == io.read(plan["heldout_prepared"] + "/selected-requests.json"),
                    "Deterministic canonical held-out subset/protocol regeneration differs")
            # Reuse the unchanged independent raw-dataset/tokenizer functions.
            require(sha(OLD_AUDIT / "audit_quality_repeats.py") == QUALITY_AUDITOR_SHA,
                    "Independent task-input auditor differs")
            from audit_quality_repeats import load_raw_auditor
            raw_auditor = load_raw_auditor()
            metadata = json.loads(args.preparation.read_bytes())
            require(sha(args.preparation) == quality["preparation_sha256"],
                    "Canonical preparation metadata differs")
            data = raw_auditor.datasets(args.datasets)
            raw_auditor.validate_rows(io.read(plan["full_inputs"]), data, False)
            raw_auditor.validate_rows(io.read(plan["preflight_inputs"]), data, True)
            tokenization = {"heldout": raw_auditor.reencode_prompts(args.tokenizer_dir,
                io.read(reconstructed / "selected-requests.json"), metadata, data, False),
                "training": raw_auditor.reencode_prompts(args.tokenizer_dir, io.read(plan["preflight_inputs"]), metadata, data, True)}
            for launch in training + heldout:
                directory, arm, mode, budget = launch["directory"], launch["arm"], launch["mode"], launch["budget"]
                native_gate(io, modules, plan, launch, io.work)
                server = io.read(directory + "/server-launch.json")
                process = server["process"]; identity = process["pid"], str(process["start_ticks"])
                require(identity not in identities and process["pid"] == process["pgrp"]
                        and launch["closure"]["causal"]["identity"] == process
                        and server["command"] == server_command(plan, budget) and server["allocation"] == plan["allocation"],
                        "Fresh owned server command/identity differs")
                identities.add(identity)
                env = server["environment"]
                expected = {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "PYTHONPATH": plan["site_root"], "OMP_NUM_THREADS": "1",
                    "NVSHMEM_QP_DEPTH": "4096", "VLLM_PLUGINS": "placement_aware_router",
                    "PLACEMENT_PROBE": TOOLS + "/causal_worker_probe.py", "CAUSAL_PROBE_DIR": directory + "/probe",
                    "CAUSAL_PROBE_ARM": arm, "CAUSAL_SOURCE_PROOF": plan["source_proof"], "CAUSAL_EXPECTED_BUDGET": str(budget)}
                if arm != "stock": expected.update(PLACEMENT_KEEP="6" if arm == "keep6p8" else "4", PLACEMENT_LOAD_PENALTY="8")
                require(env == expected, "Archived causal server environment differs")
                imports = server["cpu_import_identity"]
                expected_modules = {"placement_router", "placement_router.plugin", "placement_router.kernel", "placement_router.vendor", "placement_router.vendor.fused_topk_router"}
                require(imports["entry_points"] == [["placement_aware_router", "placement_router.plugin:register"]]
                        and set(imports["module_paths"]) == set(imports["module_sha256"]) == expected_modules,
                        "Actual plugin entrypoint differs")
                for module, path in imports["module_paths"].items():
                    relative = module.replace(".", "/") + ("/__init__.py" if module in ("placement_router", "placement_router.vendor") else ".py")
                    require(path == plan["site_root"] + "/" + relative
                            and imports["module_sha256"][module] == SOURCE["src/" + relative], "Actual imported module bytes differ")
                log_path = directory + "/server.log"
                require(io.file_sha(log_path) == launch["server_log_sha256"], "Server log changed")
                log = io.resolve(log_path)[0].read_text(errors="replace")
                require("placement-router: routing left unchanged" not in log
                        and ("placement-router: placement-aware routing active" in log) == (arm != "stock"), "Routing activation/fallback differs")
                warmup = warmup_gate(io, modules, plan, launch)
                transport = transport_gate(io, directory + "/client", monotonic=mode == "heldout")
                require(warmup["first_submitted_unix_ns"] >= previous_finish
                        and transport["first_submitted_unix_ns"] >= warmup["last_returned_unix_ns"]
                        and transport["last_returned_unix_ns"] <= launch["finished_unix_ns"], "Fresh launch/client intervals overlap")
                if mode == "heldout": require(transport["first_submitted_unix_ns"] >= freeze["created_unix_ns"], "Held-out queries preceded lock")
                previous_finish = launch["finished_unix_ns"]
                require(io.file_sha(directory + "/client/result.json") == launch["client_result_sha256"], "Client result changed")
                output = io.work / (mode + directory.rsplit("/", 1)[1] + "-analysis.json")
                scientific = modules["causal_probe" if mode == "training" else "heldout_causal_v2"]
                result = scientific_replay(scientific, io, io.resolve(plan[mode + "_prepared"] + "/protocol.json")[0].parent,
                    io.resolve(directory + "/client/result.json")[0].parent, output)
                require(result == io.read(launch["analysis"]["path"]), "Full raw regenerated analysis differs")
                analyses[(mode, arm, budget)] = {"analysis": result, "transport": transport,
                    "launch_sha256": io.file_sha(directory + "/launch.json")}
    io.work = None
    require(previous_finish <= int(outer["finished_unix"] * 1_000_000_000), "Outer completion precedes final owned launch closure")
    return {"complete": True, "all_checks_pass": True, "training_launches": 6, "heldout_launches": 3,
        "plan_sha256": args.plan_sha256, "fresh_owner_uid": UID, "campaign": "par-gap-completion-20261018",
        "old_partial_results_reused": False,
        "archive": {k:v for k,v in archived.items() if k != "final_freeze"},
        "canonical_preparation_metadata_sha256": sha(args.preparation),
        "source_audit_sha256": sha(Path(__file__)), "frozen_sources_sha256": FROZEN,
        "quality_full9_audit_sha256": sha(args.quality_audit), "training_receipt_sha256": io.file_sha(REMOTE + "/training/receipt.json"),
        "outer": outer, "independent_canonical_reencoding": tokenization,
        "heldout_receipt_sha256": io.file_sha(REMOTE + "/heldout/receipt.json"),
        "training": {f"{a}-budget{b}": analyses["training", a, b] for a, b in TRAINING},
        "heldout": {a: analyses["heldout", a, 16384] for a in ARMS},
        "checked_input_manifest": io.checked,
        "scope": "All6 training + all3 heldout raw replay; fixed256-item serial target-absent re-prefill policy and teacher candidate-conditioned policy. One launch perarm; no population/equivalence or throughput claim.",
        "release_authority": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("files", "plan", "bindings", "quality-audit", "datasets", "tokenizer-dir", "preparation", "marker", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-sha256", "quality-audit-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite prior audit")
    require(not args.output.resolve().is_relative_to(args.files.resolve()), "Audit output cannot modify the raw archive scope")
    try:
        value = audit(args)
    except BaseException as error:
        value = {"complete": False, "all_checks_pass": False, "effects": None, "error": str(error),
                 "traceback": traceback.format_exc(), "scope": "Full causal matrix failed; no partial held-out effect published"}
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
    raise SystemExit(0 if value["complete"] else 2)


if __name__ == "__main__":
    main()
