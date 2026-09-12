# gpu serverless worker -- segments a skin lesion from an image using SAM.
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
    name="segment_worker",
    gpu=[GpuGroup.ADA_24, GpuGroup.AMPERE_24, GpuGroup.AMPERE_16],
    workers=(0, 3),  # scales to 0 when idle (no charges); use (1, 3) to keep
    # one worker always warm and skip the first-request cold start/model load
    # at the cost of continuous GPU billing.
    idle_timeout=300,
    dependencies=["transformers", "pillow"],
)
async def segment(input_data: dict) -> dict:
    """
    Segment a lesion from an image using SAM, prompted with a center point.

    Input:
        image_base64: str - base64-encoded source image (JPEG/PNG)

    Returns:
        segmented_image_base64: str - base64-encoded PNG crop of the lesion,
            bounded to the highest-confidence mask
        bbox: [x0, y0, x1, y1] - bounding box of the mask within the source image
        score: float - SAM's IoU confidence score for the chosen mask
    """
    import base64
    import io

    import torch
    from PIL import Image
    from transformers import SamModel, SamProcessor

    global _MODEL, _PROCESSOR, _DEVICE

    try:
        if _MODEL is None:
            model_id = "facebook/sam-vit-base"
            _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            _MODEL = SamModel.from_pretrained(model_id).to(_DEVICE)
            _MODEL.eval()
            _PROCESSOR = SamProcessor.from_pretrained(model_id)

        image_bytes = base64.b64decode(input_data["image_base64"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        width, height = image.size
        center_point = [[[width // 2, height // 2]]]

        inputs = _PROCESSOR(
            image, input_points=center_point, return_tensors="pt"
        ).to(_DEVICE)

        with torch.no_grad():
            outputs = _MODEL(**inputs)

        masks = _PROCESSOR.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        scores = outputs.iou_scores[0, 0]

        best_idx = int(scores.argmax())
        best_mask = masks[0][0, best_idx].numpy()
        best_score = float(scores[best_idx])

        ys, xs = best_mask.nonzero()
        if len(xs) == 0 or len(ys) == 0:
            return {"status": "error", "error": "SAM returned an empty mask"}

        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        cropped = image.crop(bbox)

        buffer = io.BytesIO()
        cropped.save(buffer, format="PNG")
        segmented_image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "segmented_image_base64": segmented_image_base64,
            "bbox": list(bbox),
            "score": best_score,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


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
    print("Testing SAM segment worker with a synthetic lesion image")
    result = asyncio.run(segment(test_payload))
    if result["status"] == "success":
        print(f"Success! bbox={result['bbox']} score={result['score']:.4f}")
    else:
        print(f"Error: {result}")
