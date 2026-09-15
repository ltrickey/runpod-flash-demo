# gpu serverless worker -- segments a skin lesion from an image using SAM.
# run with: flash dev
#
# The model is loaded inside the function on every call. Caching it in a
# module-level global does speed this up (measured ~12s -> ~1.4s once warm),
# but only when deployed -- under `flash dev` it raises NameError, because
# live provisioning ships the decorated function's source without the
# surrounding module's state. That split is written up in the README; caching
# is left out here to keep the worker simple. A cache in a separate module,
# imported inside the function body, would avoid the NameError (not tested
# here). The same constraint rules out helpers defined at module level in
# this file.
from runpod_flash import Endpoint, GpuGroup


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

    Input:
        image_base64: str - base64-encoded source image (JPEG/PNG)

    Returns:
        segmented_image_base64: str - base64-encoded PNG of the lesion,
            with non-lesion pixels blacked out, at the original framing
            (not cropped)
        segmentation_applied: bool - False if no coherent mask was found and
            the original image was passed through
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

    min_coverage, max_coverage, min_solidity = 0.03, 0.90, 0.40

    try:
        model_id = "facebook/sam-vit-base"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SamModel.from_pretrained(model_id).to(device)
        model.eval()
        processor = SamProcessor.from_pretrained(model_id)

        image_bytes = base64.b64decode(input_data["image_base64"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        width, height = image.size
        total_px = width * height
        center_point = [[[width // 2, height // 2]]]

        inputs = processor(
            image, input_points=center_point, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        masks = processor.image_processor.post_process_masks(
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
            # Black out non-lesion pixels but keep the original framing. NOT
            # cropped to the bbox: cropping would change scale and framing as
            # well, so masked vs raw would differ in two ways at once and the
            # comparison could not attribute any difference to masking. The
            # bbox is still reported as metadata.
            #
            # Lesion pixels keep their RGB. Himel et al.'s wording ("converted
            # to binary masking", white = lesion / black = everything else)
            # more likely means the ViT is fed the bare silhouette, with all
            # colour discarded. Keeping the pixels is the more generous
            # reading: it hands the classifier strictly more information than
            # a silhouette would, so it gives segmentation its best shot.
            # README covers both readings.
            pixels = np.array(image)
            pixels[~mask] = 0
            Image.fromarray(pixels).save(buffer, format="PNG")
            segmentation_applied = True
        else:
            score, bbox, coverage, solidity = 0.0, (0, 0, width, height), 1.0, 1.0
            image.save(buffer, format="PNG")
            segmentation_applied = False

        segmented_image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "segmented_image_base64": segmented_image_base64,
            "segmentation_applied": segmentation_applied,
            "bbox": list(bbox),
            "score": score,
            "coverage": coverage,
            "solidity": solidity,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}
