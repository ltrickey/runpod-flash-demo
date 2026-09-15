"""SAM mask selection and application.

Deliberately identical to segment_worker.segment: the training distribution has
to match what the deployed pipeline produces, which is the entire point of this
experiment. If the selection thresholds change there, they must change here too.
"""

import numpy as np
import torch
from PIL import Image

SAM_MODEL = "facebook/sam-vit-base"
MIN_COVERAGE, MAX_COVERAGE, MIN_SOLIDITY = 0.03, 0.90, 0.40


def load_sam(device):
    from transformers import SamModel, SamProcessor

    sam = SamModel.from_pretrained(SAM_MODEL).to(device)
    sam.eval()
    return sam, SamProcessor.from_pretrained(SAM_MODEL)


def select_mask(masks, scores, total_px):
    """Highest-IoU mask that is neither the whole frame nor fragmented."""
    viable = []

    for i in range(masks.shape[1]):
        mask = masks[0, i].numpy().astype(bool)
        area = int(mask.sum())
        if area == 0:
            continue

        coverage = area / total_px
        ys, xs = mask.nonzero()
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        solidity = area / ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        if MIN_COVERAGE <= coverage <= MAX_COVERAGE and solidity >= MIN_SOLIDITY:
            viable.append((float(scores[i]), mask, bbox))

    return max(viable, key=lambda item: item[0]) if viable else None


def predict_mask(image, sam, processor, device):
    """Prompt SAM at the image centre; the selected boolean mask, or None."""
    width, height = image.size
    inputs = processor(
        image, input_points=[[[width // 2, height // 2]]], return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = sam(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]

    chosen = select_mask(masks, outputs.iou_scores[0, 0], width * height)
    return None if chosen is None else chosen[1]


def black_out(image, keep):
    """Zero every pixel outside *keep*. No crop -- matching segment_worker, so
    masked and raw copies differ in masking alone, not framing or scale."""
    pixels = np.array(image)
    pixels[~keep] = 0
    return Image.fromarray(pixels)
