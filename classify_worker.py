# gpu serverless worker -- classifies a segmented skin lesion using BEiT
# fine-tuned on HAM10000 (7-class: akiec, bcc, bkl, df, mel, nv, vasc).
# run with: flash dev
# test directly: python classify_worker.py
#
# NOTE on caching: flash-examples/docs/cli/workflows.md documents this exact
# module-level global pattern to avoid reloading the model on every call.
# Confirmed it does NOT work under `flash dev` (NameError -- live/on-demand
# provisioning ships only the decorated function's isolated source, not
# surrounding module state; see runpod_flash.endpoint._is_live_provisioning,
# which is True for flash dev and False only for flash build/deploy). It's
# re-enabled here to test after `flash deploy`, where the whole file is
# baked into a real container image and imported normally, so module-level
# state should persist across requests on the same warm worker. If it still
# doesn't work post-deploy, revert to loading fresh inside the function body
# (see git history) and reload every call.
from runpod_flash import Endpoint, GpuGroup

_MODEL = None
_PROCESSOR = None
_DEVICE = None


@Endpoint(
    name="classify_worker",
    gpu=[GpuGroup.ADA_24, GpuGroup.AMPERE_24, GpuGroup.AMPERE_16],
    workers=(0, 3),  # scales to 0 when idle (no charges); use (1, 3) to keep
    # one worker always warm and skip the first-request cold start/model load
    # at the cost of continuous GPU billing.
    idle_timeout=300,
    dependencies=["transformers", "pillow"],
)
async def classify(input_data: dict) -> dict:
    """
    Classify a segmented lesion image with the HAM10000-finetuned BEiT model.

    Input:
        image_base64: str - base64-encoded segmented lesion image (e.g. the
            output of segment_worker.segment)

    Returns:
        label: str - predicted HAM10000 diagnostic class
        confidence: float - softmax probability of the predicted class
        all_scores: dict[str, float] - probability for every class
    """
    import base64
    import io

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    global _MODEL, _PROCESSOR, _DEVICE

    try:
        if _MODEL is None:
            model_id = "ALM-AHME/beit-large-patch16-224-finetuned-Lesion-Classification-HAM10000-AH-60-20-20"
            _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            _MODEL = AutoModelForImageClassification.from_pretrained(model_id).to(
                _DEVICE
            )
            _MODEL.eval()
            _PROCESSOR = AutoImageProcessor.from_pretrained(model_id)

        image_bytes = base64.b64decode(input_data["image_base64"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inputs = _PROCESSOR(images=image, return_tensors="pt").to(_DEVICE)

        with torch.no_grad():
            outputs = _MODEL(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu()
        top_idx = int(probs.argmax())
        id2label = _MODEL.config.id2label

        return {
            "status": "success",
            "label": id2label[top_idx],
            "confidence": float(probs[top_idx]),
            "all_scores": {id2label[i]: float(probs[i]) for i in range(len(probs))},
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    import asyncio
    import base64
    import io

    from PIL import Image, ImageDraw

    def make_test_image() -> str:
        img = Image.new("RGB", (224, 224), (224, 172, 142))
        draw = ImageDraw.Draw(img)
        draw.ellipse((60, 60, 164, 164), fill=(90, 60, 45))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    test_payload = {"image_base64": make_test_image()}
    print("Testing BEiT classify worker with a synthetic lesion image")
    result = asyncio.run(classify(test_payload))
    if result["status"] == "success":
        print(f"Success! label={result['label']} confidence={result['confidence']:.4f}")
    else:
        print(f"Error: {result}")
