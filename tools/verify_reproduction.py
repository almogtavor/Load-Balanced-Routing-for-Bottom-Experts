#!/usr/bin/env python3
"""Check published snapshot bytes; this does not replay raw scientific acceptance."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    snapshot = root / "reproduction/2026-10-18"
    manifest = json.loads((snapshot / "source-manifest.json").read_text())
    for name, record in manifest["files"].items():
        path = (snapshot / name).resolve()
        if not path.is_relative_to(snapshot.resolve()) or not path.is_file():
            raise ValueError(f"Missing or invalid snapshot path: {name}")
        data = path.read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"Frozen source changed: {name}")
    report = root / "results/2026-10-18/causal-hellaswag.json"
    if hashlib.sha256(report.read_bytes()).hexdigest() != "a32d9c61949ee40a3f45a9cc0a7c0af768fd095987bb1135639f5a44797e8563":
        raise ValueError("Published causal statistics report changed")
    print(f"Verified {len(manifest['files'])} frozen files and the exact causal report.")
    print("Raw archives, GPU execution and scientific acceptance were not replayed.")


if __name__ == "__main__":
    main()
