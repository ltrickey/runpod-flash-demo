"""Cycle an endpoint's workers and prove which code they're running.

A Flash deploy cuts a new release, but a worker can keep serving the previous
release's code while the API reports `version` as current and `isStale` as
false. Runpod's API docs are explicit that `isStale` compares *configuration*:
"True when the worker is running an older endpoint configuration than the
current one". Under Flash your code is not part of that configuration -- only a
hash of it is, in an env var -- so those fields cannot tell you what a worker
loaded. This was observed twice: a train_worker started 2.5 minutes after its
release failed inside the previous version's file, and a load-balancer worker
served four-day-old pipeline.py for an entire demo.

The fix is to drain the workers and then prove the new code with a request the
old code could not possibly answer:

    1. scale to 0
    2. WAIT until both /health and the worker list report zero -- the step that
       is easy to skip and the reason a cycle silently does nothing
    3. restore the worker range
    4. probe with something only the current code answers correctly

usage:
    python scripts/cycle_and_verify.py lesion_pipeline
    python scripts/cycle_and_verify.py classify_worker --max 3
    python scripts/cycle_and_verify.py train_worker --skip-probe
    python scripts/cycle_and_verify.py --endpoint-id abc123 --skip-probe

Needs RUNPOD_API_KEY in the environment or .env.
"""

import argparse
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

REST = "https://rest.runpod.io/v1/endpoints"
API = "https://api.runpod.ai/v2"
CATALOG = "https://api.runpod.io/v2/serverless"

# This project's deployed endpoints, so the common case is one argument.
ENDPOINTS = {
    "lesion_pipeline": "uvu4lc1mmlihc0",
    "classify_worker": "8064syplx1zsn0",
    "segment_worker": "4h6wodvsn0zap7",
    "train_worker": "f11k4djb0vnl82",
}

DRAIN_POLL_SECONDS = 10
DRAIN_TIMEOUT_SECONDS = 600
PROBE_RETRIES = 12
PROBE_WAIT_SECONDS = 15


def now():
    return time.strftime("%H:%M:%S", time.gmtime())


def probe_pipeline(endpoint_id, headers):
    """An unknown arm. Current code rejects it by name before any GPU work."""
    response = requests.post(
        f"https://{endpoint_id}.api.runpod.ai/analyze",
        json={"input_data": {"image_base64": "AAAA", "arm": "bogus"}},
        headers=headers,
        timeout=180,
    )
    body = response.text[:200]
    worker = response.headers.get("x-runpod-worker-id")
    ok = response.status_code == 200 and "unknown arm: bogus" in body
    return ok, response.status_code, worker, body


def probe_classify(endpoint_id, headers):
    """An unknown model_variant. Old code ignored the field and 'succeeded'."""
    response = requests.post(
        f"{API}/{endpoint_id}/runsync",
        json={"input": {"input_data": {"image_base64": "AAAA", "model_variant": "bogus"}}},
        headers=headers,
        timeout=300,
    )
    output = response.json().get("output") or {}
    body = json.dumps(output)[:200]
    ok = "unknown model_variant" in body
    return ok, response.status_code, response.json().get("workerId"), body


def probe_train(endpoint_id, headers):
    """batch_size=0 raises straight away; the traceback names the code that ran.

    Writes nothing: it fails on the first batch, after loading one checkpoint.
    """
    response = requests.post(
        f"{API}/{endpoint_id}/runsync",
        json={"input": {"input_data": {"stage": "evaluate", "batch_size": 0}}},
        headers=headers,
        timeout=600,
    )
    payload = response.json()
    output = payload.get("output") or {}
    frames = [
        line.strip()
        for line in (output.get("traceback") or "").splitlines()
        if line.strip().startswith('File "/app')
    ]
    ok = any("lesion_training" in frame for frame in frames)
    return ok, response.status_code, payload.get("workerId"), " | ".join(frames[:4]) or json.dumps(output)[:200]


PROBES = {
    "lesion_pipeline": probe_pipeline,
    "classify_worker": probe_classify,
    "train_worker": probe_train,
}


def config(endpoint_id, headers):
    d = requests.get(f"{REST}/{endpoint_id}", headers=headers, timeout=60).json()
    return d.get("workersMin"), d.get("workersMax"), d.get("idleTimeout")


def patch(endpoint_id, headers, body):
    """Scale an endpoint. The REST API takes FLAT keys -- a nested
    {"workers": {...}} body returns 400 and silently changes nothing."""
    response = requests.patch(f"{REST}/{endpoint_id}", json=body, headers=headers, timeout=60)
    print(f"{now()} PATCH {body} -> {response.status_code}")
    response.raise_for_status()


def worker_counts(endpoint_id, headers):
    """Both views, because they can disagree; a cycle is only safe when both say 0."""
    health = requests.get(f"{API}/{endpoint_id}/health", headers=headers, timeout=60).json()
    listed = requests.get(f"{CATALOG}/{endpoint_id}/workers", headers=headers, timeout=60).json()
    counted = sum(v for v in (health.get("workers") or {}).values() if isinstance(v, int))
    return counted, listed.get("workers", [])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("endpoint", nargs="?", choices=sorted(ENDPOINTS), help="Known endpoint name")
    parser.add_argument("--endpoint-id", help="Raw endpoint id, for anything not in the list")
    parser.add_argument("--min", type=int, default=0, help="workersMin to restore (default 0)")
    parser.add_argument(
        "--max",
        type=int,
        default=1,
        help="workersMax to restore (default 1: with one worker, the probe speaks for the "
        "whole endpoint)",
    )
    parser.add_argument("--skip-probe", action="store_true", help="Cycle only, don't verify")
    args = parser.parse_args()

    endpoint_id = args.endpoint_id or ENDPOINTS.get(args.endpoint)
    if not endpoint_id:
        raise SystemExit("give a known endpoint name or --endpoint-id")

    load_dotenv()
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("RUNPOD_API_KEY must be set (environment or .env)")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    print(f"{now()} endpoint {args.endpoint or endpoint_id} ({endpoint_id})")
    print(f"{now()} config before (min, max, idleTimeout): {config(endpoint_id, headers)}")

    patch(endpoint_id, headers, {"workersMin": 0, "workersMax": 0})

    deadline = time.time() + DRAIN_TIMEOUT_SECONDS
    while True:
        counted, listed = worker_counts(endpoint_id, headers)
        print(f"{now()} health workers={counted}  listed workers={len(listed)}")
        if counted == 0 and not listed:
            print(f"{now()} DRAINED")
            break
        if time.time() > deadline:
            raise SystemExit(
                f"{now()} still not drained after {DRAIN_TIMEOUT_SECONDS}s -- not restoring "
                "blindly, since a half-drained endpoint can keep serving old code"
            )
        time.sleep(DRAIN_POLL_SECONDS)

    patch(endpoint_id, headers, {"workersMin": args.min, "workersMax": args.max})
    print(f"{now()} config after (min, max, idleTimeout): {config(endpoint_id, headers)}")

    probe = PROBES.get(args.endpoint)
    if args.skip_probe or probe is None:
        reason = "--skip-probe" if args.skip_probe else f"no probe defined for {args.endpoint or endpoint_id}"
        print(f"{now()} not verifying ({reason}). Cycling alone does not prove which code is live.")
        return

    for attempt in range(1, PROBE_RETRIES + 1):
        try:
            ok, status, worker, body = probe(endpoint_id, headers)
        except requests.RequestException as error:
            print(f"{now()} probe {attempt}: {type(error).__name__}, retrying")
            time.sleep(PROBE_WAIT_SECONDS)
            continue
        if status in (502, 503):
            print(f"{now()} probe {attempt}: http {status} (worker still starting), retrying")
            time.sleep(PROBE_WAIT_SECONDS)
            continue
        print(f"{now()} probe {attempt}: http={status} worker={worker}")
        print(f"          {body}")
        print(f"{now()} VERDICT: {'CURRENT code is live' if ok else 'NOT current -- cycle again or check the logs'}")
        sys.exit(0 if ok else 1)

    raise SystemExit(f"{now()} endpoint never answered the probe")


if __name__ == "__main__":
    main()
