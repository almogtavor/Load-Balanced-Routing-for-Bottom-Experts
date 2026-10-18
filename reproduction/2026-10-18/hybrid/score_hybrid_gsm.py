#!/usr/bin/env python3
"""Exact GSM8K extraction after a successful raw+native phase audit."""
import argparse
import re
from pathlib import Path
from common_v2 import file_sha, read, require, write_new


def normalize(text):
    return re.sub(r"\.$", "", str(text).strip().replace(",", "").replace("$", "")).lower()


def extract(text):
    strict = re.findall(r"The answer is (\-?[0-9\.\,]+).", text)
    flexible = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    return {"strict": normalize(strict[0]) if strict else None,
            "flexible": normalize(next(v for v in flexible[-1] if v)) if flexible else None}


def main(args):
    audit, start, result = read(args.phase_audit), read(args.result / "start.json"), read(args.result / "result.json")
    require(audit.get("complete") is True and audit["campaign_result_sha256"] == file_sha(args.result / "result.json")
            and start["mode"] in ("smoke", "generate"), "Successful same-campaign phase audit required")
    require(file_sha(args.prepared) == start["input_file_sha256"], "Frozen prompts changed")
    rows = [r for r in read(args.prepared) if r["task"] == "gsm8k"]
    if start["mode"] == "smoke":
        rows = [r for r in rows if r["split"] == "train"][:4]
    else:
        require(len(rows) == 1319 and all(r["split"] == "test" for r in rows)
                and start["max_tokens"] == 2048, "Full held-out claim requires all1319 canonical test prompts/max2048")
    records = [read(args.result / f"records/{i:06d}.json") for i in range(result["requests"])]
    require(len(rows) == len(records) and [r["id"] for r in rows] == [r["id"] for r in records], "GSM item identity/order differs")
    scored = []
    for row, record in zip(rows, records):
        choice = record["choice"]
        answers = extract(choice["text"])
        scored.append({"id": row["id"], "gold": row["gold"], "answers": answers,
                       "correct": answers["flexible"] == row["gold"], "strict_correct": answers["strict"] == row["gold"],
                       "truncated": choice["finish_reason"] == "length", "generated_tokens": len(choice["token_ids"]),
                       "empty": not choice["text"].strip(), "parse_failed": answers["flexible"] is None})
    n = len(scored)
    write_new(args.output, {"complete": True, "prefill_arm": start["arm"], "decode_arm": "stock", "n": n,
              "correct": sum(r["correct"] for r in scored), "strict_correct": sum(r["strict_correct"] for r in scored),
              "accuracy_percent": 100 * sum(r["correct"] for r in scored) / n,
              "length_finished": sum(r["truncated"] for r in scored), "empty": sum(r["empty"] for r in scored),
              "parse_failed": sum(r["parse_failed"] for r in scored), "records": scored,
              "phase_audit_sha256": file_sha(args.phase_audit), "result_sha256": file_sha(args.result / "result.json"),
              "scope": "Observed launch accuracy in actual disaggregated phase policy; no launch-population interval",
              "helper_sha256": file_sha(__file__)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("prepared", "result", "phase-audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
