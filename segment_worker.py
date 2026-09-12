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

    SAM returns three candidate masks at different granularities. Picking the
    highest IoU naively selects the "whole frame" mask on most dermoscopy
    images, which makes segmentation a no-op. Instead, candidates are filtered
    on two criteria before preferring highest IoU:

      - coverage: fraction of the frame the mask occupies. Rejects the
        whole-image mask (>90%) and tiny specks (<3%).
      - solidity: fraction of its own bounding box the mask fills. Rejects
        fragmented masks -- scattered hairs/streaks can pass the coverage
        check while being useless for classification (a real lesion fills
        ~0.4-0.8 of its bbox; streak masks measure ~0.07-0.2).

    When no candidate is coherent, the original image is passed through
    unmasked with segmentation_applied=False, rather than handing the
    classifier garbage.

    On apply_mask: blacking out the background pushes the image off the
    distribution the classifier was fine-tuned on (raw dermoscopy photos),
    which measurably *hurts* accuracy -- on the sample set, masking dropped
    7/7 correct to 5/7 and lowered confidence on every image it touched.
    apply_mask=False keeps real pixels and only crops to the lesion bbox,
    which is the better-performing mode. The toggle exists so the two can be
    compared directly.

    Input:
        image_base64: str - base64-encoded source image (JPEG/PNG)
        apply_mask: bool - black out non-lesion pixels (default True). When
            False, crop to the mask's bounding box but keep real pixels.

    Returns:
        segmented_image_base64: str - base64-encoded PNG of the lesion,
            cropped to the mask's bounding box, with non-lesion pixels
            blacked out when apply_mask is True
        segmentation_applied: bool - False if no coherent mask was found and
            the original image was passed through
        mask_applied: bool - whether non-lesion pixels were actually blacked out
        bbox: [x0, y0, x1, y1] - bounding box of the mask within the source image
        score: float - SAM's IoU confidence score for the chosen mask
        coverage / solidity: float - the two selection metrics, for inspection
    """
    import base64
    import io

    import numpy as np
    import torch
    from PIL import Image
    from transformers import SamModel, SamProcessor

    global _MODEL, _PROCESSOR, _DEVICE

    min_coverage, max_coverage, min_solidity = 0.03, 0.90, 0.40
    apply_mask = input_data.get("apply_mask", True)

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
        total_px = width * height
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

        viable = []
        for i in range(masks[0].shape[1]):
            mask = masks[0][0, i].numpy().astype(bool)
            area = int(mask.sum())
            if area == 0:
                continue

            coverage = area / total_px
            ys, xs = mask.nonzero()
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            solidity = area / ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

            if min_coverage <= coverage <= max_coverage and solidity >= min_solidity:
                viable.append((float(scores[i]), mask, bbox, coverage, solidity))

        buffer = io.BytesIO()

        if viable:
            score, mask, bbox, coverage, solidity = max(viable, key=lambda c: c[0])
            if apply_mask:
                pixels = np.array(image)
                pixels[~mask] = 0
                Image.fromarray(pixels).crop(bbox).save(buffer, format="PNG")
            else:
                image.crop(bbox).save(buffer, format="PNG")
            segmentation_applied = True
            mask_applied = apply_mask
        else:
            score, bbox, coverage, solidity = 0.0, (0, 0, width, height), 1.0, 1.0
            image.save(buffer, format="PNG")
            segmentation_applied = False
            mask_applied = False

        segmented_image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "segmented_image_base64": segmented_image_base64,
            "segmentation_applied": segmentation_applied,
            "mask_applied": mask_applied,
            "bbox": list(bbox),
            "score": score,
            "coverage": coverage,
            "solidity": solidity,
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
