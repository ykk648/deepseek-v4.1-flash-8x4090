#!/usr/bin/env python3
import argparse
import json
import time
import urllib.request

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8011")
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tokens", type=int, default=32000)
    parser.add_argument("--passkey", default="73918426")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    unit = "Routine archive note: no action is required for this entry.\n"
    fact = f"The passkey for Project Cedar is {args.passkey}. Remember it exactly.\n"
    question = "\nWhat is the passkey for Project Cedar? Reply with only the digits."
    unit_tokens = len(tokenizer.encode(unit, add_special_tokens=False))
    fixed_tokens = len(tokenizer.encode(fact + question, add_special_tokens=False))
    repeats = max(1, (args.tokens - fixed_tokens) // unit_tokens)
    split = repeats // 4
    prompt = unit * split + fact + unit * (repeats - split) + question

    payload = json.dumps(
        {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 128,
        }
    ).encode()
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - started
    answer = result["choices"][0]["message"]["content"]
    passed = args.passkey in answer
    print(
        json.dumps(
            {
                "prompt_tokens": result["usage"]["prompt_tokens"],
                "completion_tokens": result["usage"]["completion_tokens"],
                "elapsed_seconds": round(elapsed, 3),
                "answer": answer,
                "passed": passed,
            },
            ensure_ascii=False,
        )
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
