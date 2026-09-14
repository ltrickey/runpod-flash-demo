"""Submit a job to train_worker and poll until it finishes.

Training and the SAM preprocessing pass both run far longer than an HTTP
request will stay open, so they go through the queue endpoint's async /run
route and get polled via /status, rather than /runsync.

usage:
    # quick smoke test: mask 14 images, prove SAM + volume writes work
    python train_client.py --stage prepare --n-images 14

    # full preprocessing pass over HAM10000 (~10k images, resumable)
    python train_client.py --stage prepare

    # train both arms from the prepared dataset
    python train_client.py --stage train

Needs TRAIN_ENDPOINT_ID and RUNPOD_API_KEY in the environment or .env.
"""

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

POLL_SECONDS = 20
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


def submit(endpoint_id, headers, payload):
    response = requests.post(
        f"https://api.runpod.ai/v2/{endpoint_id}/run",
        json={"input": {"input_data": payload}},
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["id"]


def poll(endpoint_id, headers, job_id):
    """Block until the job reaches a terminal state, printing status changes."""
    started = time.time()
    last_status = None

    while True:
        response = requests.get(
            f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}",
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        status = body.get("status")

        elapsed = timedelta(seconds=int(time.time() - started))
        if status != last_status:
            print(f"[{elapsed}] {status}")
            last_status = status
        else:
            print(f"[{elapsed}] {status}", end="\r", flush=True)

        if status in TERMINAL:
            print()
            return body

        time.sleep(POLL_SECONDS)


def main():
    parser = argparse.ArgumentParser(description="Run a train_worker job and wait for it.")
    parser.add_argument("--stage", default="both", choices=["prepare", "train", "both", "evaluate"])
    parser.add_argument("--n-images", type=int, default=0, help="0 means all (~10k)")
    # Default to None and omit from the payload rather than duplicating the
    # worker's defaults here. Two copies of a default is a trap: the client
    # silently overrides the worker, so changing the worker appears to have
    # no effect -- which looks exactly like a stale deploy.
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--variants",
        default="masked,raw",
        help="Comma-separated training arms (default both)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the prepared dataset and manifest before preparing again",
    )
    parser.add_argument("--endpoint-id", default=None, help="Defaults to TRAIN_ENDPOINT_ID")
    args = parser.parse_args()

    load_dotenv()

    endpoint_id = args.endpoint_id or os.environ.get("TRAIN_ENDPOINT_ID")
    if not endpoint_id:
        raise SystemExit("set TRAIN_ENDPOINT_ID in .env or pass --endpoint-id")

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY must be set (environment or .env)")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "stage": args.stage,
        "n_images": args.n_images,
        "reset": args.reset,
        "variants": [v.strip() for v in args.variants.split(",") if v.strip()],
    }
    # only send what was explicitly asked for, so the worker's defaults apply
    if args.epochs is not None:
        payload["epochs"] = args.epochs
    if args.batch_size is not None:
        payload["batch_size"] = args.batch_size

    print(f"submitting stage={args.stage} to {endpoint_id}")
    job_id = submit(endpoint_id, headers, payload)
    print(f"job {job_id}\n")

    body = poll(endpoint_id, headers, job_id)
    output = body.get("output")

    print(json.dumps(output if output is not None else body, indent=2)[:4000])

    # Persist each stage's raw output so training_report.py can assemble a
    # report without re-running anything. These jobs are expensive; their
    # results should outlive the terminal scrollback.
    if isinstance(output, dict):
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        saved = results_dir / f"train_{args.stage}.json"
        saved.write_text(json.dumps(output, indent=2))
        print(f"\nsaved {saved}")

    if body.get("status") != "COMPLETED" or (
        isinstance(output, dict) and output.get("status") == "error"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
