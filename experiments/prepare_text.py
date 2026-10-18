"""Archive dataset bytes and natural text with SHA256 provenance for a run.

The upstream converted-parquet branch can change. Reproduce a completed run from
its archived files, or verify a new download against the recorded source hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    url = ("https://huggingface.co/datasets/Salesforce/wikitext/resolve/"
           "refs%2Fconvert%2Fparquet/wikitext-2-raw-v1/test/0000.parquet")
    request = urllib.request.Request(url, headers={"User-Agent": "placement-router-experiments"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    parquet = args.output / "wikitext-2-raw-test.parquet"
    parquet.write_bytes(payload)
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet, columns=["text"])["text"].to_pylist()
    text = "\n".join(row for row in rows if row and row.strip())
    target = args.output / "wikitext-2-raw-test.txt"
    target.write_text(text, encoding="utf-8")
    metadata = {
        "dataset": "Salesforce/wikitext", "config": "wikitext-2-raw-v1", "split": "test",
        "source_url": url, "rows": len(rows), "text_characters": len(text),
        "parquet_sha256": hashlib.sha256(payload).hexdigest(),
        "text_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "text_file": str(target),
    }
    (args.output / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
