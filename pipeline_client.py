"""Shared helpers for calling the deployed lesion pipeline.

Used by demo_client.py (one image) and eval_sweep.py (a whole directory).
"""

import os
import time

import requests

LOCAL_ROUTE = "/pipeline/analyze"
DEPLOYED_ROUTE = "/analyze"
COLD_START_RETRIES = 6
RETRY_WAIT_SECONDS = 20
REQUEST_TIMEOUT_SECONDS = 300

# The two complete pipelines: preprocessing paired with the model trained for
# it. Never mix them -- that measures a distribution mismatch instead.
ARMS = ("masked", "raw")


def resolve_target(url, path_override=None):
    """Work out the route and auth headers for this endpoint.

    Deployed endpoints need bearer auth and serve routes at the root; a local
    `flash dev` server needs neither and namespaces routes under the endpoint
    name.
    """
    is_local = "localhost" in url or "127.0.0.1" in url
    path = path_override or (LOCAL_ROUTE if is_local else DEPLOYED_ROUTE)

    if is_local:
        return path, {}

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit(
            "RUNPOD_API_KEY must be set (environment or .env) to call a deployed endpoint"
        )
    return path, {"Authorization": f"Bearer {api_key}"}


def call_pipeline(url, path, headers, image_base64, arm="raw"):
    """POST to the pipeline, retrying through cold-start 502s.

    A fully cold call provisions a CPU worker for the pipeline and GPU workers
    for SAM (weights pulled from Hugging Face) and the classifier (weights read
    from the network volume). Measured from zero workers, the gateway holds the
    request open and answers after ~55s rather than timing out. Cold calls do
    sometimes come back 502 instead -- the cause is unconfirmed, and it is not
    simply a fixed gateway deadline -- so retrying rides it out.
    """
    payload = {"input_data": {"image_base64": image_base64, "arm": arm}}

    for attempt in range(1, COLD_START_RETRIES + 1):
        response = requests.post(
            f"{url}{path}", json=payload, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
        )
        if response.status_code == 502 and attempt < COLD_START_RETRIES:
            print(f"  workers still warming (502), retry {attempt}/{COLD_START_RETRIES - 1}...")
            time.sleep(RETRY_WAIT_SECONDS)
            continue

        response.raise_for_status()
        result = response.json()
        if result.get("status") != "success":
            raise SystemExit(f"pipeline returned an error: {result}")
        return result

    raise SystemExit("pipeline never became available")
