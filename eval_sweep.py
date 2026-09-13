"""Evaluation sweep: run every image in a directory through the pipeline both
ways (masked and crop-only) and report accuracy per mode.

Exists to check whether the masked-vs-crop-only accuracy difference observed
on the original 7 sample images holds on a larger set. Writes raw results to
JSON so the numbers can be re-derived without re-running (which costs GPU time).

usage:
    python eval_sweep.py --url https://<id>.api.runpod.ai --dir eval_images
"""

import argparse
import base64
import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from demo_client import call_pipeline, resolve_target


# (label, apply_mask, segment) -- "raw" bypasses SAM entirely, which is the
# baseline that shows whether segmentation earns its place at all.
MODES = (
    ("masked", True, True),
    ("crop-only", False, True),
    ("raw", False, False),
)


def evaluate(image_paths, url, path, headers):
    """Return {image_stem: {expected, modes: {mode: {...}}}}."""
    records = {}

    for index, image_path in enumerate(image_paths, 1):
        stem = image_path.stem
        expected = stem.split("_")[0]
        image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

        record = {"expected": expected, "modes": {}}
        for label, apply_mask, segment in MODES:
            result = call_pipeline(url, path, headers, image_base64, apply_mask, segment)
            record["modes"][label] = {
                "prediction": result["label"],
                "confidence": result["confidence"],
                "segmentation_applied": result["segmentation"].get("applied"),
            }

        records[stem] = record
        summary = "  ".join(
            f"{label}={record['modes'][label]['prediction']:5s}"
            f"({record['modes'][label]['confidence']:.3f})"
            for label, _, _ in MODES
        )
        print(f"[{index}/{len(image_paths)}] {stem:26s} exp={expected:5s} {summary}")

    return records


def summarise(records):
    totals = defaultdict(int)
    per_class = defaultdict(lambda: defaultdict(int))
    confidence_sums = defaultdict(float)
    segmented_count = 0

    for record in records.values():
        expected = record["expected"]
        per_class[expected]["n"] += 1
        if record["modes"]["masked"]["segmentation_applied"]:
            segmented_count += 1

        for mode, outcome in record["modes"].items():
            confidence_sums[mode] += outcome["confidence"]
            if outcome["prediction"] == expected:
                totals[mode] += 1
                per_class[expected][mode] += 1

    return totals, per_class, confidence_sums, segmented_count


def main():
    parser = argparse.ArgumentParser(description="Run an accuracy sweep over a directory of images.")
    parser.add_argument("--url", required=True, help="Base URL of the deployed Flash server")
    parser.add_argument("--dir", default="eval_images", help="Directory of labelled images")
    parser.add_argument("--path", default=None, help="Override the route path")
    parser.add_argument("--out", default="eval_results.json", help="Where to write raw results")
    args = parser.parse_args()

    load_dotenv()
    path, headers = resolve_target(args.url, args.path)

    image_paths = sorted(Path(args.dir).glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"no .jpg images found in {args.dir}")

    records = evaluate(image_paths, args.url, path, headers)
    Path(args.out).write_text(json.dumps(records, indent=2))

    totals, per_class, confidence_sums, segmented_count = summarise(records)
    n = len(records)

    print(f"\n{'=' * 58}")
    print(f"{n} images | SAM found a coherent mask on {segmented_count}")
    print(f"{'=' * 58}")
    mode_labels = [label for label, _, _ in MODES]

    print(f"{'mode':<12} {'correct':>12} {'accuracy':>10} {'mean conf':>12}")
    for mode in mode_labels:
        accuracy = totals[mode] / n
        print(f"{mode:<12} {totals[mode]:>7}/{n:<4} {accuracy:>9.1%} {confidence_sums[mode] / n:>11.3f}")

    header = f"\n{'class':<8} {'n':>3}" + "".join(f"{label:>11}" for label in mode_labels)
    print(header)
    for class_name in sorted(per_class):
        counts = per_class[class_name]
        row = f"{class_name:<8} {counts['n']:>3}" + "".join(
            f"{counts[label]:>11}" for label in mode_labels
        )
        print(row)

    print(f"\nRaw results written to {args.out}")


if __name__ == "__main__":
    main()
