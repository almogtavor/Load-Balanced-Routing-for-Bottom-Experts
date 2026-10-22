"""One setting of the accuracy check: GSM8K, HellaSwag and MMLU through lm-eval, served by the same
`vllm serve` (data-parallel, DeepEP) that `serve_run.py` times, the plugin set by `SETTING`.

Per-item samples go to `runs/eval_<tag>_<setting>_r<rep>/`, so settings are compared item by item
against the base setting of the same job, and `base2` - the same engine launched again - is the floor
any difference has to clear.
"""

from __future__ import annotations

import os
import subprocess
import sys

# GSM8K's 5-shot prompt runs past serve_run's default context; `or`, since the job exports "".
os.environ["MAX_LEN"] = os.environ.get("MAX_LEN") or "4096"
import serve_run as S  # noqa: E402

TASKS = os.environ.get("EVAL_TASKS") or "gsm8k,hellaswag,mmlu"
LIMIT = os.environ.get("EVAL_LIMIT")
# Gemma-4-26B is instruction-tuned only: plain completions score it at ~44 on GSM8K.
CHAT = os.environ.get("EVAL_CHAT")


def main() -> None:
    keep, load_penalty, mass = S.setting_env()
    print(f"[{S.TAG}/{S.SETTING}] eval keep={keep} load_penalty={load_penalty} mass={mass} "
          f"dp={S.DP} a2a={S.A2A} tasks={TASKS} limit={LIMIT}", flush=True)
    # Its own venv over the image's packages: whatever lm-eval pulls in must not change the vLLM
    # the server is about to run on.
    py = "/tmp/lme/bin/python"
    if not os.path.exists(py):
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", "/tmp/lme"],
                       check=True)
    subprocess.run([py, "-m", "pip", "install", "-q", "lm-eval[api]"], check=True)
    out = S.OUT / f"eval_{S.TAG}_{S.SETTING}_r{S.REP}"
    p = S.serve()
    try:
        subprocess.run([
            py, "-m", "lm_eval", "--model", "local-completions",
            "--model_args", f"model={S.MODEL},base_url={S.URL}/v1/completions,num_concurrent=128,"
                            f"max_retries=3,tokenized_requests={bool(CHAT)},tokenizer={S.MODEL},"
                            f"max_length={S.MAX_LEN}",
            "--tasks", TASKS, "--log_samples", "--output_path", str(out),
            *(["--limit", LIMIT] if LIMIT else []),
            *(["--apply_chat_template", "--fewshot_as_multiturn"] if CHAT else [])], check=True)
    finally:
        S.kill(p)
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
