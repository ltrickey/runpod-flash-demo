# thin demo client: reads a local image file and calls the lesion analysis
# pipeline's /analyze route (either the local flash dev server or a deployed one).
# Saves the segmented/masked crop alongside the original so the demo can show
# what each pipeline stage actually produced, not just the final label.
#
# usage:
#   python demo_client.py sample_images/mel_ISIC_0024351.jpg #no --url flag = default to localhost
#   python demo_client.py sample_images/mel_ISIC_0024351.jpg --url http://localhost:8888
#   python demo_client.py sample_images/mel_ISIC_0024351.jpg --out-dir results
#   python demo_client.py sample_images/mel_ISIC_0024351.jpg --url https://<id>.api.runpod.ai
#
# Against a deployed endpoint you also need RUNPOD_API_KEY (read from the
# environment or .env) -- deployed endpoints require bearer auth, local
# flash dev does not. Note the route differs too: flash dev namespaces routes
# under the endpoint name (/pipeline/analyze) while a deployed load-balanced
# endpoint serves them at the root (/analyze). This is handled automatically;
# override with --path if needed.
import argparse
import base64
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


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
    parser.add_argument(
        "--path",
        default=None,
        help="Route path to call (default: /pipeline/analyze locally, /analyze when deployed)",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Crop to the lesion without blacking out the background. Masking "
        "pushes the image off the classifier's training distribution and "
        "measurably lowers accuracy -- use this to compare the two.",
    )
    args = parser.parse_args()

    load_dotenv()

    is_local = "localhost" in args.url or "127.0.0.1" in args.url
    path = args.path or ("/pipeline/analyze" if is_local else "/analyze")

    headers = {}
    if not is_local:
        api_key = os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            raise SystemExit(
                "RUNPOD_API_KEY must be set (environment or .env) to call a deployed endpoint"
            )
        headers["Authorization"] = f"Bearer {api_key}"

    image_path = Path(args.image_path)
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    response = requests.post(
        f"{args.url}{path}",
        json={
            "input_data": {
                "image_base64": image_base64,
                "apply_mask": not args.no_mask,
            }
        },
        headers=headers,
        timeout=300,
    )
    response.raise_for_status()
    result = response.json()

    segmentation = result.get("segmentation", {})
    segmented_image_base64 = segmentation.pop("segmented_image_base64", None)

    print(json.dumps(result, indent=2))

    if segmented_image_base64:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_cropped" if args.no_mask else "_masked"
        crop_path = out_dir / f"{image_path.stem}{suffix}.png"
        with open(crop_path, "wb") as f:
            f.write(base64.b64decode(segmented_image_base64))
        print(f"\nOriginal image:  {image_path}")
        print(f"Segmented crop:  {crop_path}")


if __name__ == "__main__":
    main()
