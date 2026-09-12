# orchestration: SAM segmentation -> BEiT classification.
# thin load-balanced client that chains segment_worker and classify_worker,
# following the CPU->GPU->CPU pipeline pattern from
# flash-examples/01_getting_started/03_mixed_workers/pipeline.py.

from runpod_flash import Endpoint

pipeline = Endpoint(name="lesion_pipeline", cpu="cpu3c-1-2", workers=(1, 3))


@pipeline.post("/analyze")
async def analyze(input_data: dict) -> dict:
    """
    Full pipeline: segment a lesion with SAM, then classify it with BEiT.

    Input:
        image_base64: str - base64-encoded source image
        apply_mask: bool - black out non-lesion pixels before classifying
            (default True). Set False to crop to the lesion without masking;
            see segment_worker for why this materially changes accuracy.

    Returns:
        label: str - predicted HAM10000 diagnostic class
        confidence: float - softmax probability of the predicted class
        all_scores: dict[str, float] - probability for every class
        segmentation: dict - what the SAM stage produced: the lesion image
            (base64 PNG), its bbox, SAM's IoU score, the coverage/solidity
            selection metrics, `applied` (False when no coherent mask was
            found and the original image was passed through unchanged), and
            `mask_applied` (whether pixels were actually blacked out)
    """
    from classify_worker import classify
    from segment_worker import segment

    segment_result = await segment(
        {
            "image_base64": input_data["image_base64"],
            "apply_mask": input_data.get("apply_mask", True),
        }
    )
    if segment_result.get("status") != "success":
        return {"status": "error", "stage": "segment", "error": segment_result.get("error")}

    classify_result = await classify(
        {"image_base64": segment_result["segmented_image_base64"]}
    )
    if classify_result.get("status") != "success":
        return {"status": "error", "stage": "classify", "error": classify_result.get("error")}

    return {
        "status": "success",
        "label": classify_result["label"],
        "confidence": classify_result["confidence"],
        "all_scores": classify_result["all_scores"],
        "segmentation": {
            "applied": segment_result["segmentation_applied"],
            "mask_applied": segment_result["mask_applied"],
            "bbox": segment_result["bbox"],
            "score": segment_result["score"],
            "coverage": segment_result["coverage"],
            "solidity": segment_result["solidity"],
            "segmented_image_base64": segment_result["segmented_image_base64"],
        },
    }


@pipeline.get("/health")
async def health() -> dict:
    """Health check for the lesion analysis pipeline."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import asyncio
    import base64
    import io

    from PIL import Image, ImageDraw

    def make_test_image() -> str:
        img = Image.new("RGB", (256, 256), (224, 172, 142))
        draw = ImageDraw.Draw(img)
        draw.ellipse((90, 90, 166, 166), fill=(90, 60, 45))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    test_payload = {"image_base64": make_test_image()}
    print("Testing full lesion analysis pipeline with a synthetic image")
    result = asyncio.run(analyze(test_payload))
    print(f"Result: {result}")
