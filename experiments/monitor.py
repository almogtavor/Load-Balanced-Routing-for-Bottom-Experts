"""Lightweight GPU and cgroup telemetry, timestamped separately from benchmark results."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time


def read(path: str):
    try:
        return pathlib.Path(path).read_text().strip()
    except OSError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--seconds", type=float, default=10800)
    args = parser.parse_args()
    if args.interval <= 0 or args.seconds <= 0:
        parser.error("interval and seconds must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stop = time.monotonic() + args.seconds
    fields = "index,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem"
    with args.output.open("a", buffering=1) as output:
        while time.monotonic() < stop:
            start = time.monotonic()
            row = {"unix_seconds": time.time(), "gpu_fields": fields.split(","),
                   "cpu_stat": read("/sys/fs/cgroup/cpu.stat"),
                   "memory_current": read("/sys/fs/cgroup/memory.current"),
                   "memory_events": read("/sys/fs/cgroup/memory.events")}
            try:
                result = subprocess.run(["nvidia-smi", f"--query-gpu={fields}",
                                         "--format=csv,noheader,nounits"],
                                        capture_output=True, text=True, timeout=5)
                row["gpu_csv"] = result.stdout.strip().splitlines()
                row["query_returncode"] = result.returncode
            except subprocess.TimeoutExpired:
                row["query_timeout"] = True
            output.write(json.dumps(row) + "\n")
            time.sleep(max(0, args.interval - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
