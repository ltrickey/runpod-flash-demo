# gpu serverless worker -- classifies a segmented skin lesion using BEiT
# fine-tuned on HAM10000 (7-class: akiec, bcc, bkl, df, mel, nv, vasc).
# run with: flash dev
# test directly: python classify_worker.py
#
# The model is loaded inside the function on every call. Caching it in a
# module-level global does speed this up (measured ~12s -> ~1.4s once warm),
# but only when deployed -- under `flash dev` it raises NameError, because
# live provisioning ships the decorated function's source in isolation
# without the surrounding module. That split is written up in the README;
# it's left out of the code here to keep both paths working and the worker
# simple. The same constraint is why this function body is self-contained
# rather than calling module-level helpers.
from runpod_flash import Endpoint, GpuGroup


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

    try:
        model_id = "ALM-AHME/beit-large-patch16-224-finetuned-Lesion-Classification-HAM10000-AH-60-20-20"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForImageClassification.from_pretrained(model_id).to(device)
        model.eval()
        processor = AutoImageProcessor.from_pretrained(model_id)

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
