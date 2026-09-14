# orchestration: SAM segmentation -> ViT classification.
# thin load-balanced client that chains segment_worker and classify_worker,
# following the CPU->GPU->CPU pipeline pattern from
# flash-examples/01_getting_started/03_mixed_workers/pipeline.py.

from runpod_flash import Endpoint

pipeline = Endpoint(name="lesion_pipeline", cpu="cpu3c-1-2", workers=(1, 3))


@pipeline.post("/analyze")
async def analyze(input_data: dict) -> dict:
    """
    Full pipeline: classify a lesion using one of two complete arms.

    An arm is a preprocessing choice *and* the model trained for it, always
    paired:

        arm="raw"     -> SAM skipped, raw-trained model      (default)
        arm="masked"  -> SAM blacks out the background,
                         masked-trained model

    They are paired on purpose. Feeding a masked image to the raw-trained
    model measures a distribution mismatch, not the value of segmentation,
    and makes masking look worse than it is. Keeping each model on its own
    preprocessing is the fair comparison -- and the one that shows
    segmentation does not earn its place here (results/training_report.md).

    Input:
        image_base64: str - base64-encoded source image
        arm: "raw" | "masked" - which complete pipeline to run (default "raw")

    Returns:
        label: str - predicted HAM10000 diagnostic class
        confidence: float - softmax probability of the predicted class
        all_scores: dict[str, float] - probability for every class
        segmentation: dict - what the SAM stage did. `requested` is False when
            bypassed; `applied` is False when bypassed or when no coherent
            mask was found. `segmented_image_base64` is always the image that
            was actually classified, so callers can render it uniformly.
    """
    import traceback

    # Everything is wrapped: without this, any unexpected exception surfaces as
    # the load balancer's bare "Internal Server Error" with no detail, and the
    # worker log API returns 404 for LB workers, so there is nowhere else to
    # read what went wrong. Returning the traceback in the body is the only
    # reliable way to debug this tier.
    try:
        from classify_worker import classify
        from segment_worker import segment

        image_base64 = input_data["image_base64"]
        arm = input_data.get("arm", "raw")
        if arm not in ("raw", "masked"):
            return {"status": "error", "error": f"unknown arm: {arm}"}
        segment_requested = arm == "masked"

        if segment_requested:
            segment_result = await segment({"image_base64": image_base64})
            if segment_result.get("status") != "success":
                return {
                    "status": "error",
                    "stage": "segment",
                    "error": segment_result.get("error"),
                }

            classified_image = segment_result["segmented_image_base64"]
            segmentation = {
                "requested": True,
                "applied": segment_result["segmentation_applied"],
                "bbox": segment_result["bbox"],
                "score": segment_result["score"],
                "coverage": segment_result["coverage"],
                "solidity": segment_result["solidity"],
                "segmented_image_base64": classified_image,
            }
        else:
            classified_image = image_base64
            segmentation = {
                "requested": False,
                "applied": False,
                "bbox": None,
                "score": None,
                "coverage": None,
                "solidity": None,
                "segmented_image_base64": classified_image,
            }

        classify_result = await classify(
            {"image_base64": classified_image, "model_variant": arm}
        )
        if classify_result.get("status") != "success":
            return {
                "status": "error",
                "stage": "classify",
                "error": classify_result.get("error"),
            }

        return {
            "status": "success",
            "arm": arm,
            "label": classify_result["label"],
            "confidence": classify_result["confidence"],
            "all_scores": classify_result["all_scores"],
            "segmentation": segmentation,
        }

    except Exception as e:
        return {
            "status": "error",
            "stage": "pipeline",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


@pipeline.get("/health")
async def health() -> dict:
    """Health check for the lesion analysis pipeline."""
    return {"status": "healthy"}
