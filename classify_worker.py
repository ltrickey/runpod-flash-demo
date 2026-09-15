# gpu serverless worker -- classifies a lesion image with the ViT fine-tuned
# by train_worker.py (7-class: akiec, bcc, bkl, df, mel, nv, vasc).
#
# Serves either trained arm, selected per request. The pipeline pairs each
# model with the preprocessing it was trained for -- a masked-trained model
# only ever sees masked input, a raw-trained model only raw. Mixing them
# measures a distribution mismatch rather than the preprocessing itself, and
# would make masking look worse than it fairly is.
#
# Default is the raw arm. SAM masking costs 14.2 points of balanced accuracy
# that retraining cannot recover; with expert ground-truth masks the cost is
# still 7.0, so about half was poor segmentation and half is masking itself
# (see results/training_report.md).
#
# Weights are read from the network volume rather than Hugging Face, which is
# why this endpoint carries volume= and datacenter=. The volume is
# datacenter-scoped, so attaching it pins inference to EU-RO-1 instead of the
# eleven datacenters this endpoint could otherwise schedule across. The
# alternative -- downloading the weights and baking them into the deploy
# artifact -- avoids the pin but adds ~350MB to every deploy and a manual
# re-fetch after each retrain.
#
# The model is loaded inside the function on every call. Caching it in a
# module-level global does speed things up (measured on the SAM worker:
# ~12s -> ~1.4s once warm), but only when deployed -- under `flash dev` it
# raises NameError, because live provisioning ships the decorated function's
# source without the surrounding module's state. That split is written up in
# the README; caching is left out here to keep the worker simple. The same
# constraint rules out helpers defined at module level in this file. A cache,
# or any shared code, can instead live in a separate module imported inside
# the function body -- the way train_worker.py uses lesion_training/ -- though
# a module-level cache has not been tested here.
from runpod_flash import DataCenter, Endpoint, GpuGroup, NetworkVolume

# same volume train_worker.py writes checkpoints to, matched by name
volume = NetworkVolume(
    name="lesion-training",
    size=50,
    datacenter=DataCenter.EU_RO_1,
)


@Endpoint(
    name="classify_worker",
    gpu=[GpuGroup.ADA_24, GpuGroup.AMPERE_24, GpuGroup.AMPERE_16],
    workers=(0, 3),  # scales to 0 when idle (no charges); use (1, 3) to keep
    # one worker always warm and skip the first-request cold start/model load
    # at the cost of continuous GPU billing.
    idle_timeout=300,
    volume=volume,
    datacenter=DataCenter.EU_RO_1,
    dependencies=["transformers", "pillow"],
)
async def classify(input_data: dict) -> dict:
    """
    Classify a lesion image with the fine-tuned ViT.

    Input:
        image_base64: str - base64-encoded lesion image, normally the output
            of segment_worker.segment
        model_variant: "raw" | "masked" - which trained arm to load
            (default "raw")

    Returns:
        label: str - predicted HAM10000 diagnostic class
        confidence: float - softmax probability of the predicted class
        all_scores: dict[str, float] - probability for every class
    """
    import base64
    import io
    import os

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    variant = input_data.get("model_variant", "raw")
    if variant not in ("raw", "masked"):
        return {"status": "error", "error": f"unknown model_variant: {variant}"}
    checkpoint = f"/runpod-volume/models/vit-base-p32-{variant}"

    try:
        if not os.path.isdir(checkpoint):
            return {
                "status": "error",
                "error": (
                    f"no checkpoint at {checkpoint} -- run train_worker "
                    "(stage=prepare, then stage=train) to create it"
                ),
            }

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForImageClassification.from_pretrained(checkpoint).to(device)
        model.eval()
        processor = AutoImageProcessor.from_pretrained(checkpoint)

        image_bytes = base64.b64decode(input_data["image_base64"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu()
        top_idx = int(probs.argmax())
        id2label = model.config.id2label

        return {
            "status": "success",
            "model_variant": variant,
            "label": id2label[top_idx],
            "confidence": float(probs[top_idx]),
            "all_scores": {id2label[i]: float(probs[i]) for i in range(len(probs))},
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}
