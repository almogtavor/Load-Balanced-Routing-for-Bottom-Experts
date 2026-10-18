"""Fresh four-H100 ownership and immutable prior-campaign provenance.

Old Linux PIDs belong to a deleted pod and are never queried in the new PID
namespace. Prior results are provenance only, never statistical observations.
"""
from pathlib import Path
import uuid

from common_v2 import file_sha, read, require

CAMPAIGN = "par-gap-completion-20261018"
POD = "par-gap-causal-h100-20261018"
IMAGE = "vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b"
OLD_UID = "c42b0a8c-eb60-4837-9bd9-e83e4fceaa3c"
NATIVE_MANIFEST_SHA = "699c77c5d7562123231073df39589dfe41d7512f321de3010502681f280e2a18"
PRIOR = {
    "quality_receipt": "641b6f66d62aee3c866acfac40ad84b7feeb666b6d8db714f8e6434d544a0ffb",
    "archive_marker": "7fa8a5fba7d232b129e0f08b769a02b5b00c946d6ef84eec1b6368205a6a0f89",
    "release_receipt": "ff552a703c486d741824e473e0c689daa1912ede533a6c1802a9182aa6f0fc7e",
}
SELECTED_SHA = "600b8775d9a604d80f9277b7327c62d7f2a23d745507a16578dc3401da4b5609"


def prior_gate(path, _lib, process):
    owner = read(process["owner_proof"])
    paths = owner["prior_inputs"]
    require(set(paths) == set(PRIOR) and str(Path(path).resolve()) == str(Path(paths["quality_receipt"]).resolve()),
            "Wrong archived quality provenance paths")
    for key, expected in PRIOR.items():
        require(file_sha(paths[key]) == expected, "Prior immutable provenance changed: " + key)
    receipt, marker, released = (read(paths[k]) for k in ("quality_receipt", "archive_marker", "release_receipt"))
    require(receipt.get("complete") is True and receipt.get("execution_complete") is True
            and receipt.get("cleanup_complete") is True and receipt.get("cleanup_errors") == [],
            "Archived nine-launch quality campaign incomplete")
    rows = receipt["launches"]
    expected = {(r, a) for r in (1, 2, 3) for a in ("stock", "keep6p8", "keep4p8")}
    require(len(rows) == 9 and {(r["replicate"], r["arm"]) for r in rows} == expected
            and all(r["complete"] is True and r["ep"] == 4 for r in rows), "Archived quality matrix differs")
    require(marker.get("final_verified") is True and marker["source"]["pod_uid"] == OLD_UID
            and released.get("deleted") is True and released.get("absence_verified") is True
            and released["pod_uid"] == OLD_UID and released["archive_marker_sha256"] == PRIOR["archive_marker"]
            and released["archive_sha256"] == marker["archive_sha256"], "Old pod closure/whole-archive proof differs")
    return receipt


def native_gate(path):
    require(file_sha(path) == NATIVE_MANIFEST_SHA, "Frozen nineteen-file native source manifest changed")
    value = read(path)
    require(value.get("complete") is True and value["native_file_count"] == len(value["files"]) == 19,
            "Require all nineteen pinned native files")
    for filename, expected in value["files"].items():
        require(file_sha(filename) == expected, "Fresh installed native source differs: " + filename)
    return value["files"]


def fresh_owner_gate(path, allocation_path, source_path, native_path, quality_path):
    owner = read(path)
    allocation = read(allocation_path)
    require(owner.get("schema") == "fresh-four-h100-causal-owner-v1" and owner.get("complete") is True
            and owner.get("campaign") == CAMPAIGN and owner.get("old_partial_results_reused") is False,
            "Fresh campaign owner proof missing or wrong")
    identity = owner["pod_identity"]
    require(set(identity) == {"namespace", "pod", "uid", "node", "image"}
            and identity["namespace"] == "itay" and identity["pod"] == POD and identity["image"] == IMAGE
            and identity["node"] and str(uuid.UUID(identity["uid"])) == identity["uid"] and identity["uid"] != OLD_UID,
            "Wrong fresh pod owner/namespace/image")
    require(all(allocation.get(k) == identity[k] for k in ("namespace", "pod", "uid", "node", "image"))
            and allocation.get("verified_live") is True and allocation.get("gpu_count") == 4
            and "H100" in allocation.get("gpu_name", ""), "Exactly four fresh H100s required")
    rows = [r.split(",") for r in allocation["gpu_inventory"]]
    require(len(rows) == 4 and {r[0].strip() for r in rows} == {"0", "1", "2", "3"}
            and len({r[1].strip() for r in rows}) == 4 and all("H100" in r[2] for r in rows),
            "Duplicate/missing physical GPU inventory")
    source = read(source_path)
    require(all(source["pod_identity"].get(k) == identity[k] for k in identity), "Fresh source owner differs")
    require(owner["allocation_proof_sha256"] == file_sha(allocation_path)
            and owner["source_proof_sha256"] == file_sha(source_path)
            and owner["native_manifest_sha256"] == NATIVE_MANIFEST_SHA, "Fresh source/native proof binding changed")
    prior_gate(quality_path, None, {"owner_proof": str(path)})
    return native_gate(native_path)


def selected_gate(prepared):
    prepared = Path(prepared)
    value = read(prepared / "protocol.json")
    require(file_sha(prepared / "selected-requests.json") == SELECTED_SHA
            and value["selected_requests_sha256"] == SELECTED_SHA
            and value["items"] == 256 and value["candidates"] == 1024
            and value["prefix_requests"] == 29611 and value["total_requests"] == 29615
            and value["prefix_concurrency"] == 1 and value["teacher_requests"] == 4,
            "Fresh held-out policy/subset/request count differs from frozen reference")
