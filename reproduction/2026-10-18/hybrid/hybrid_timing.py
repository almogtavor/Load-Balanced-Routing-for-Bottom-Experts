"""Uninstrumented real NIXL two-leg streaming timing, memory-only on the clock."""
from __future__ import annotations
from native_request_id import external_item_id, producer_internal_id
from native_accounting import zero_cached_transfer, zero_cached_usage, begin_round, finish_round
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import threading
import time
import traceback
import urllib.error
import urllib.request
from common_v2 import base_body, choices, encoded, require, sha, write_new
from hybrid_client import paired_body, transfer_params


def stream_handoff(plan, row, body, producer_wire, parser, barrier, cohort):
    value = {"complete": False, "request_index": row["request_id"], "chunks": [], "done": False}
    raw_p, raw_d = bytearray(), bytearray()
    consumer_wire = b""
    deadline = time.monotonic() + plan["request_timeout"]
    def remaining():
        result = min(deadline - time.monotonic(), plan["stop_unix"] - time.time())
        require(result > 0, "Absolute two-leg request deadline reached")
        return result
    try:
        barrier.wait(timeout=60)
        value["request_start_ns"] = time.perf_counter_ns()
        value["cohort_start_ns"] = cohort["start_ns"]
        deadline = time.monotonic() + plan["request_timeout"]
        headers = {"Content-Type": "application/json", "X-data-parallel-rank": str(row["request_id"] % 4),
                   "X-Request-Id": body["request_id"]}
        request = urllib.request.Request(plan["prefill_url"] + "/v1/completions", producer_wire, headers)
        with urllib.request.urlopen(request, timeout=remaining()) as response:
            value["producer_status"] = response.status
            require(response.status == 200, "Producer HTTP failure")
            # Incremental reads preserve partial bytes if a later read fails.
            while True:
                if response.fp is None or getattr(response, "length", None) == 0:
                    break
                response.fp.raw._sock.settimeout(remaining())
                part = response.read1(65536)
                if not part:
                    break
                raw_p.extend(part)
                require(len(raw_p) <= 2 << 20, "Oversized producer response")
        producer = json.loads(raw_p)
        require(producer["id"]=="cmpl-"+body["request_id"], "Producer response/application request identity differs")
        p_choice = choices(producer, [row["token_ids"]])[0]
        require(len(p_choice["token_ids"]) == 1 and p_choice["finish_reason"] == "length"
                and producer["usage"]["prompt_tokens"] == len(row["token_ids"])
                and producer["usage"]["completion_tokens"] == 1, "Producer token accounting differs")
        params = transfer_params(producer, len(row["token_ids"]))
        require(producer_internal_id(producer) == params["remote_request_id"], "Native transfer full internal ID differs")
        _, consumer = paired_body(body, params)
        consumer.update(stream=True, stream_options={"include_usage": True})
        zero_cached_transfer(params, json.loads(producer_wire)["kv_transfer_params"], consumer["kv_transfer_params"])
        consumer_wire = encoded(consumer)
        value["producer_finished_ns"] = time.perf_counter_ns()
        request = urllib.request.Request(plan["decode_url"] + "/v1/completions", consumer_wire, headers)
        with urllib.request.urlopen(request, timeout=remaining()) as response:
            value["consumer_status"] = response.status
            require(response.status == 200 and (response.headers.get("Content-Type") or "").startswith("text/event-stream"),
                    "Consumer must return HTTP200 SSE")
            fields = []
            while True:
                response.fp.raw._sock.settimeout(remaining())
                line = response.readline()
                now = time.perf_counter_ns()
                raw_d.extend(line)
                require(line, "EOF before consumer DONE")
                require(len(raw_d) <= 16 << 20, "Oversized consumer response")
                decoded = line.decode().rstrip("\r\n")
                if not decoded and fields:
                    parser("\n".join(fields), value, now, row["token_ids"])
                    fields = []
                    if value["done"]:
                        break
                elif decoded.startswith("data:"):
                    fields.append(decoded[5:].lstrip(" "))
        value["finished_ns"] = time.perf_counter_ns()
        usage = value.get("usage", {})
        require(usage.get("prompt_tokens") == len(row["token_ids"]) and usage.get("completion_tokens") == plan["output_tokens"]
                and sum(len(x["token_ids"]) for x in value["chunks"]) == plan["output_tokens"]
                and value.get("finish_reason") == "length" and value.get("prompt_ids_verified") is True,
                "Fixed-output token accounting/finish/ID contract differs")
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        zero_cached_usage(usage, len(row["token_ids"]), plan["output_tokens"])
        first = value["chunks"][0]["arrival_ns"]
        value.update(complete=True, cached_prompt_tokens=cached,
                     ttft_ms=(first - value["request_start_ns"]) / 1e6,
                     cohort_ttft_ms=(first - cohort["start_ns"]) / 1e6,
                     latency_ms=(value["done_ns"] - value["request_start_ns"]) / 1e6,
                     transfer_params=params)
    except BaseException as error:
        value["error"] = traceback.format_exc()
        if isinstance(error, urllib.error.HTTPError):
            try:
                data = error.read(2 << 20)
                (raw_d if consumer_wire else raw_p).extend(data)
            except BaseException:
                pass
            value["http_error_status"] = error.code
    return value, bytes(raw_p), bytes(raw_d), consumer_wire


def application_round(plan, prompts, parser, directory, phase, number):
    # Fixed256 logical prompts,64 workers: four client waves are an explicit
    # deployment policy. Worker start latency and whole-cohort latency both kept.
    values = []
    for row in prompts:
        body = base_body("task-model", row["token_ids"], f"{plan['launch_id']}-{phase}-{number}-{row['request_id']:03d}", plan["output_tokens"])
        body["ignore_eos"] = True
        producer, _ = paired_body(body, None)
        values.append((row, body, encoded(producer)))
    cohort = {"start_ns": time.perf_counter_ns(), "start_unix_ns": time.time_ns()}
    results = []
    # Each wave has a common barrier. No raw file writes occur until the whole
    #256-request clock has stopped; every failed/in-flight result is retained.
    with cf.ThreadPoolExecutor(max_workers=64) as pool:
        for offset in range(0, len(values), 64):
            barrier = threading.Barrier(65)
            jobs = [pool.submit(stream_handoff, plan, row, body, wire, parser, barrier, cohort)
                    for row, body, wire in values[offset:offset + 64]]
            try:
                barrier.wait(timeout=60)
            except threading.BrokenBarrierError:
                cohort["barrier_error"] = "All64 workers did not reach the common wave barrier"
            results.extend(job.result() for job in jobs)
            if not all(r[0]["complete"] for r in results[-64:]):
                break
    finish = time.perf_counter_ns()
    finish_unix_ns = time.time_ns()
    directory.mkdir(parents=True, exist_ok=False)
    records = []
    for (row, body, producer_wire), (record, raw_p, raw_d, consumer_wire) in zip(values, results):
        name = f"request{row['request_id']:03d}"
        for suffix, data in (("prefill.request.json", producer_wire), ("prefill.response.json", raw_p),
                             ("decode.request.json", consumer_wire), ("decode.sse", raw_d)):
            path = directory / (name + "." + suffix)
            with path.open("xb") as stream:
                stream.write(data)
        record["files_sha256"] = {name + "." + suffix: sha(data) for suffix, data in
            (("prefill.request.json", producer_wire), ("prefill.response.json", raw_p),
             ("decode.request.json", consumer_wire), ("decode.sse", raw_d))}
        records.append(record)
    elapsed = (finish - cohort["start_ns"]) / 1e9
    complete = len(results) == len(prompts) and all(r["complete"] for r in records) and "barrier_error" not in cohort
    summary = {"complete": complete, "phase": phase, "round": number, "elapsed_seconds": elapsed,
               "logical_requests": 256, "concurrency": 64, "client_waves": 4,
               "input_tokens": 256 * 1024, "output_tokens": 256 * plan["output_tokens"],
               "application_tokens_per_second": 256 * (1024 + plan["output_tokens"]) / elapsed if complete else None,
               "rate_token_scope": "One logical input+consumeroutput; discard producerone-token probe and do not double-count transferredinput",
               "cohort": cohort, "finished_ns": finish, "finished_unix_ns": finish_unix_ns, "records": records,
               "clock_scope": "All256 real two-leg requests in four64-request waves through allSSEcompletion; disk writes excluded",
               "generation": "forced128ignoreEOStrue; speed workload only"}
    write_new(directory / "application-round.json", summary)
    require(complete, "Hybrid timing round incomplete; raw current-wave results retained, later waves not submitted")
    return summary


def run_round(plan, prompts, parser, directory, phase, number):
    directory=Path(directory);metrics_folder=directory.with_name(directory.name+'-native-metrics')
    summary=None
    try:
        native=begin_round(plan,phase,number,metrics_folder)
        summary=application_round(plan,prompts,parser,directory,phase,number)
        require(all(r['transport']['finished_unix_ns']<=summary['cohort']['start_unix_ns'] for r in native['before'].values()),
                'Native baseline GET overlaps completion clock')
        summary={**summary,'application_round_sha256':sha((directory/'application-round.json').read_bytes()),
                 'native_metrics':finish_round(plan,native,summary['finished_unix_ns']),
                 'cached_token_scope':'Native zero remote-producer prefix hits; NIXL coverage independently checked outsideclock'}
        write_new(directory/'round.json',summary)
        return summary
    except BaseException:
        directory.mkdir(parents=True,exist_ok=True)
        failure={'complete':False,'phase':phase,'round':number,'application_round_complete':bool(summary and summary.get('complete')),
                 'error':traceback.format_exc(),'effects':None,'native_metrics_directory':str(metrics_folder)}
        if not (directory/'round.json').exists():write_new(directory/'round.json',failure)
        raise
