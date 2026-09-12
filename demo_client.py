# thin demo client: reads a local image file and calls the lesion analysis
# pipeline's /analyze route (either the local flash dev server or a deployed one).
# Saves the segmented/masked crop alongside the original so the demo can show
# what each pipeline stage actually produced, not just the final label.
#
# usage:
#   python demo_client.py sample_images/mel_ISIC_0024351.jpg #no --url flag = default to localhost
#   python demo_client.py sample_images/mel_ISIC_0024351.jpg --url http://localhost:8888
#   python demo_client.py sample_images/mel_ISIC_0024351.jpg --out-dir results
import argparse
import base64
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description="Call the lesion analysis pipeline with a local image.")
    parser.add_argument("image_path", help="Path to a lesion image (JPEG/PNG)")
    parser.add_argument(
        "--url",
        default="http://localhost:8888",
        help="Base URL of the running Flash server (default: local flash dev)",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory to save the segmented/masked crop image into (default: current directory)",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    response = requests.post(
        f"{args.url}/pipeline/analyze",
        json={"input_data": {"image_base64": image_base64}},
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()

    segmentation = result.get("segmentation", {})
    segmented_image_base64 = segmentation.pop("segmented_image_base64", None)

    print(json.dumps(result, indent=2))

    if segmented_image_base64:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        crop_path = out_dir / f"{image_path.stem}_segmented.png"
        with open(crop_path, "wb") as f:
            f.write(base64.b64decode(segmented_image_base64))
        print(f"\nOriginal image:  {image_path}")
        print(f"Segmented crop:  {crop_path}")


if __name__ == "__main__":
    main()
