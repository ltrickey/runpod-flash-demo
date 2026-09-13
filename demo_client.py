"""Demo client for the lesion analysis pipeline.

Reads a local image, runs it through the pipeline, and writes a self-contained
markdown report to --out-dir with the original image, what SAM produced, and
the resulting classification.

By default it runs the image BOTH ways -- masked (background blacked out) and
crop-only -- because the comparison is the interesting part: masking pushes
the image off the classifier's training distribution and measurably lowers
accuracy. Use --single to make just one call.

usage:
    python demo_client.py sample_images/mel_ISIC_0024351.jpg
    python demo_client.py sample_images/mel_ISIC_0024351.jpg --url https://<id>.api.runpod.ai
    python demo_client.py sample_images/mel_ISIC_0024351.jpg --single --no-mask

Against a deployed endpoint you also need RUNPOD_API_KEY (read from the
environment or .env) -- deployed endpoints require bearer auth, local flash
dev does not. The route differs too: flash dev namespaces routes under the
endpoint name (/pipeline/analyze) while a deployed load-balanced endpoint
serves them at the root (/analyze). Both are handled automatically; override
with --path.
"""

import argparse
import base64
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

LOCAL_ROUTE = "/pipeline/analyze"
DEPLOYED_ROUTE = "/analyze"
COLD_START_RETRIES = 6
RETRY_WAIT_SECONDS = 20
REQUEST_TIMEOUT_SECONDS = 300


@dataclass
class Run:
    """One pipeline call and the image it produced."""

    label: str
    image_filename: str
    result: dict

    @property
    def prediction(self) -> str:
        return self.result.get("label", "")

    @property
    def confidence(self) -> float:
        return self.result.get("confidence", 0.0)

    @property
    def class_scores(self) -> dict:
        return self.result.get("all_scores", {})

    @property
    def segmentation(self) -> dict:
        return self.result.get("segmentation", {})


def round_value(value, digits=3):
    return round(value, digits) if isinstance(value, (int, float)) else value


def markdown_table(headers, rows):
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
        *["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows],
    ]


# -- talking to the pipeline ------------------------------------------------


def resolve_target(url, path_override):
    """Work out the route and auth headers for this endpoint."""
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


def call_pipeline(url, path, headers, image_base64, apply_mask, segment=True):
    """POST to the pipeline, retrying through cold-start 502s.

    A fully cold /analyze (both GPU workers provisioning, plus pulling SAM and
    BEiT from Hugging Face) takes longer than the load balancer's ~40s gateway
    timeout, which surfaces as a 502. Retrying rides it out while the workers
    finish warming.
    """
    payload = {
        "input_data": {
            "image_base64": image_base64,
            "apply_mask": apply_mask,
            "segment": segment,
        }
    }

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


def execute_runs(modes, url, path, headers, image_base64, stem, out_dir):
    """Call the pipeline once per mode, saving each returned image."""
    runs = []
    for label, apply_mask, segment in modes:
        print(f"calling pipeline ({label})...")
        result = call_pipeline(url, path, headers, image_base64, apply_mask, segment)

        filename = f"{stem}_{label.replace('-', '_')}.png"
        segmentation = result["segmentation"]
        image_data = segmentation.pop("segmented_image_base64")
        (out_dir / filename).write_bytes(base64.b64decode(image_data))

        runs.append(Run(label=label, image_filename=filename, result=result))
        print(f"  {label}: {result['label']} ({round_value(result['confidence'])})")

    return runs


# -- report sections --------------------------------------------------------


def header_section(stem, image_path, url, expected):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return [
        f"# Lesion analysis — `{stem}`",
        "",
        f"- **Run:** {timestamp}",
        f"- **Endpoint:** `{url}`",
        f"- **Source image:** `{image_path}`",
        f"- **Expected class (from filename):** `{expected}`",
        "",
    ]


def result_section(runs, expected):
    rows = [
        [
            run.label,
            f"`{run.prediction}`",
            round_value(run.confidence),
            "yes" if run.prediction == expected else "no",
        ]
        for run in runs
    ]
    return [
        "## Result",
        "",
        *markdown_table(["mode", "prediction", "confidence", "correct"], rows),
        "",
    ]


def segmentation_section(segmentation):
    """SAM metrics are identical across modes, so report them once."""
    rows = [
        ["coherent mask found", segmentation.get("applied")],
        ["SAM IoU score", round_value(segmentation.get("score", 0), 4)],
        ["coverage (fraction of frame)", round_value(segmentation.get("coverage", 0), 4)],
        ["solidity (fill of own bbox)", round_value(segmentation.get("solidity", 0), 4)],
        ["bbox", f"`{segmentation.get('bbox')}`"],
    ]
    lines = ["## Segmentation (SAM)", "", *markdown_table(["metric", "value"], rows), ""]

    if segmentation.get("applied") is False:
        lines += [
            "> No coherent lesion mask passed the coverage/solidity filters, so "
            "the original image was passed through to the classifier unchanged. "
            "Both modes are therefore identical for this image.",
            "",
        ]
    return lines


def images_section(runs, stem, original_suffix):
    headers = ["original"] + [run.label for run in runs]
    cells = [f"![original]({stem}_original{original_suffix})"]
    cells += [f"![{run.label}]({run.image_filename})" for run in runs]
    return ["## Images", "", *markdown_table(headers, [cells]), ""]


def probabilities_section(runs, expected):
    rows = []
    for class_name in sorted(runs[0].class_scores):
        label = f"`{class_name}`" + (" **(expected)**" if class_name == expected else "")
        rows.append([label] + [round_value(run.class_scores.get(class_name, 0), 4) for run in runs])

    headers = ["class"] + [run.label for run in runs]
    return ["## Class probabilities", "", *markdown_table(headers, rows), ""]


def explanation_section():
    return [
        "## Why the two modes differ",
        "",
        "Measured over 35 held-out HAM10000 images (`eval_sweep.py`), masking "
        "the background scores 62.9% (mean confidence 0.788) against 77.1% "
        "(0.884) for cropping without masking. The likely cause is a "
        "train/inference mismatch — a black-background cutout is unlike "
        "anything in the classifier's training data — though that "
        "checkpoint's model card does not document what it was trained on, so "
        "the mechanism is a hypothesis rather than a verified fact. Note this "
        "compares masking against cropping, not against skipping segmentation "
        "altogether.",
        "",
    ]


def build_report(runs, image_path, url):
    stem = image_path.stem
    expected = stem.split("_")[0]

    lines = [
        *header_section(stem, image_path, url, expected),
        *result_section(runs, expected),
        *segmentation_section(runs[0].segmentation),
        *images_section(runs, stem, image_path.suffix),
        *probabilities_section(runs, expected),
    ]
    if len(runs) > 1:
        lines += explanation_section()

    return "\n".join(lines)


# -- entry point ------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a lesion image through the pipeline and write a markdown report."
    )
    parser.add_argument("image_path", help="Path to a lesion image (JPEG/PNG)")
    parser.add_argument(
        "--url",
        default="http://localhost:8888",
        help="Base URL of the running Flash server (default: local flash dev)",
    )
    parser.add_argument(
        "--out-dir",
        default="results",
        help="Directory for the report and images (default: results)",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Route path to call (default: /pipeline/analyze locally, /analyze when deployed)",
    )
    parser.add_argument(
        "--single", action="store_true", help="Make one call instead of comparing both modes"
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="With --single, crop to the lesion without blacking out the background",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()

    path, headers = resolve_target(args.url, args.path)

    image_path = Path(args.image_path)
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.single:
        modes = [("crop-only" if args.no_mask else "masked", not args.no_mask, True)]
    else:
        # (label, apply_mask, segment)
        modes = [("masked", True, True), ("crop-only", False, True), ("raw", False, False)]

    runs = execute_runs(modes, args.url, path, headers, image_base64, image_path.stem, out_dir)

    # copy the original alongside so the report renders standalone
    shutil.copyfile(image_path, out_dir / f"{image_path.stem}_original{image_path.suffix}")

    report_path = out_dir / f"{image_path.stem}_report.md"
    report_path.write_text(build_report(runs, image_path, args.url))
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
