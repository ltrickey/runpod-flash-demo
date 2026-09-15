"""How good the pipeline's SAM masks are, against the reference masks."""

import os
import zipfile

import numpy as np
from PIL import Image

from lesion_training import common, masking

HIMEL_REPORTED_IOU = 0.9601


def segmentation_iou():
    """Score the pipeline's zero-shot SAM masks against the published ones.

    Himel et al. report IoU 96.01% from a segmenter *trained* on these masks.
    This pipeline prompts SAM zero-shot with a centre point, so this measures
    the gap between the two approaches on identical images.

    Scored on the same seeded test split as every accuracy number, so the IoU
    and the classification results describe the same images.

    Fallbacks (no coherent mask) are reported both ways: counted as IoU 0,
    which is what the pipeline effectively delivers, and excluded, which is
    how well SAM does when it commits to a lesion at all. Reporting only the
    second would flatter it.
    """
    if not os.path.exists(common.GT_MASK_ZIP):
        return {"error": f"no mask archive at {common.GT_MASK_ZIP} -- run stage=prepare_gt"}

    archive = zipfile.ZipFile(common.GT_MASK_ZIP)
    index = common.gt_mask_index(archive)
    test_records = common.split_by_lesion(common.load_manifest())["test"]

    device = common.device()
    sam, processor = masking.load_sam(device)

    ious, dices, fell_back, missing = [], [], 0, 0
    for record in test_records:
        name = index.get(record["image_id"])
        if name is None:
            missing += 1
            continue

        image = Image.open(f"{common.DATA_ROOT}/raw/{record['file']}").convert("RGB")
        gt = common.load_gt_mask(archive, name, image.size)

        predicted = masking.predict_mask(image, sam, processor, device)
        if predicted is None:
            fell_back += 1
            ious.append(0.0)
            dices.append(0.0)
            continue

        intersection = float(np.logical_and(predicted, gt).sum())
        union = float(np.logical_or(predicted, gt).sum())
        total = float(predicted.sum() + gt.sum())
        ious.append(intersection / union if union else 0.0)
        dices.append(2 * intersection / total if total else 0.0)

    committed = [value for value in ious if value > 0]
    return {
        "n": len(ious),
        "no_ground_truth_mask": missing,
        "fell_back": fell_back,
        "fallback_rate": (fell_back / len(ious)) if ious else None,
        "mean_iou": (sum(ious) / len(ious)) if ious else None,
        "mean_dice": (sum(dices) / len(dices)) if dices else None,
        "mean_iou_excluding_fallbacks": (
            sum(committed) / len(committed) if committed else None
        ),
        "himel_reported_iou": HIMEL_REPORTED_IOU,
    }
