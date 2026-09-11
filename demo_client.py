# thin demo client: reads a local image file and calls the lesion analysis
# pipeline's /analyze route (either the local flash dev server or a deployed one).
#
# usage:
#   python demo_client.py sample_images/mel_ISIC_0024351.jpg
#   python demo_client.py sample_images/mel_ISIC_0024351.jpg --url http://localhost:8888
import argparse
import base64
import json

import requests


def main():
    parser = argparse.ArgumentParser(description="Call the lesion analysis pipeline with a local image.")
    parser.add_argument("image_path", help="Path to a lesion image (JPEG/PNG)")
    parser.add_argument(
        "--url",
        default="http://localhost:8888",
        help="Base URL of the running Flash server (default: local flash dev)",
    )
    args = parser.parse_args()

    with open(args.image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    response = requests.post(
        f"{args.url}/pipeline/analyze",
        json={"input_data": {"image_base64": image_base64}},
        timeout=120,
    )
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
