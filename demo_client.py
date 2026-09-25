"""Run one image through both arms of the pipeline and show what each did.

Two complete pipelines are compared:

    masked  -- SAM blacks out the background, classified by the model trained
               on masked images (the architecture this project set out to
               replicate)
    raw     -- SAM skipped entirely, classified by the model trained on raw
               images

Each model only ever sees the preprocessing it was trained for, so the
comparison is between two fair pipelines rather than a model being fed
unfamiliar input.

Saves the image each arm actually classified, so the masked and raw inputs can
be shown side by side.

usage:
    python demo_client.py sample_images/bcc_ISIC_0024431.jpg
    python demo_client.py sample_images/bcc_ISIC_0024431.jpg --url https://uvu4lc1mmlihc0.api.runpod.ai
"""

import argparse
import base64
from pathlib import Path

from dotenv import load_dotenv

from pipeline_client import ARMS, call_pipeline, resolve_target


def main():
    parser = argparse.ArgumentParser(description="Classify one image with both pipeline arms.")
    parser.add_argument("image_path", help="Path to a lesion image (JPEG/PNG)")
    parser.add_argument(
        "--url",
        default="http://localhost:8888",
        help="Base URL of the Flash server (default: local flash dev)",
    )
    parser.add_argument("--out-dir", default="results", help="Where to save the classified images")
    args = parser.parse_args()

    load_dotenv()
    path, headers = resolve_target(args.url)

    image_path = Path(args.image_path)
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    expected = image_path.stem.split("_")[0]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{image_path.name}  (expected: {expected})\n")

    for arm in ARMS:
        result = call_pipeline(args.url, path, headers, image_base64, arm)

        # the pipeline returns whatever image it actually classified, so the
        # saved file shows exactly what the model saw
        classified = result["segmentation"].pop("segmented_image_base64")

        # A masked-arm run where SAM found no coherent mask produces the
        # original image untouched. Name it differently so it can't be
        # mistaken for -- or charted as -- a real masked/raw comparison.
        suffix = arm
        if arm == "masked" and not result["segmentation"]["applied"]:
            suffix = "masked-fallback"
        saved = out_dir / f"{image_path.stem}_{suffix}.png"
        saved.write_bytes(base64.b64decode(classified))

        mark = "✓" if result["label"] == expected else "✗"
        # On the masked arm, SAM sometimes finds no coherent mask and the
        # original image passes through untouched. Say so: otherwise this
        # line claims a masked result for an image that was never masked,
        # and the saved PNG is identical to the raw one.
        note = ""
        if arm == "masked" and not result["segmentation"]["applied"]:
            note = "   (no coherent mask -- passed through unmasked)"
        print(
            f"  {mark} {arm:7s} {result['label']:6s} "
            f"{result['confidence']:.3f}   -> {saved}{note}"
        )


if __name__ == "__main__":
    main()
