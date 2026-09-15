"""Paths, labels, manifest access and the lesion-grouped split.

Everything every stage has to agree on lives here. split_by_lesion above all:
training, evaluate and segmentation_iou all score against its "test" fold, so
a second copy drifting would silently compare different images.
"""

import io
import json
import re
from collections import defaultdict

import numpy as np
from PIL import Image

VOLUME = "/runpod-volume"
DATA_ROOT = f"{VOLUME}/data"
MANIFEST = f"{DATA_ROOT}/manifest.json"

# masked   = SAM, what the deployed pipeline actually produces
# raw      = untouched
# gtmasked = reference ground-truth masks, the ceiling for masking
VARIANTS = ("masked", "raw", "gtmasked")

# Tschandl's lesion masks for HAM10000, one per image, from the same Harvard
# Dataverse record as the images themselves. 10.8MB, no auth. They exist to
# remove mask quality as a confound.
GT_MASK_URL = "https://dataverse.harvard.edu/api/access/datafile/3838943"
GT_MASK_ZIP = f"{VOLUME}/ham10000_segmentations.zip"

# The dataset spells diagnoses out; HAM10000 short codes are the labels.
DX_TO_CLASS = {
    "actinic_keratoses": "akiec",
    "basal_cell_carcinoma": "bcc",
    "benign_keratosis-like_lesions": "bkl",
    "dermatofibroma": "df",
    "melanoma": "mel",
    "melanocytic_Nevi": "nv",
    "vascular_lesions": "vasc",
}
CLASSES = sorted(set(DX_TO_CLASS.values()))


def checkpoint_dir(variant):
    # named after the architecture actually used, not the base model this
    # started life with
    return f"{VOLUME}/models/vit-base-p32-{variant}"


def device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_manifest():
    with open(MANIFEST) as handle:
        return json.load(handle)


def save_manifest(records):
    with open(MANIFEST, "w") as handle:
        json.dump(records, handle)


def split_by_lesion(records):
    """Group-aware split, so no lesion appears in more than one fold."""
    lesions = sorted({r["lesion_id"] for r in records})
    rng = np.random.default_rng(42)
    rng.shuffle(lesions)

    n_val = max(1, int(len(lesions) * 0.15))
    fold_of = {}
    for index, lesion in enumerate(lesions):
        if index < n_val:
            fold_of[lesion] = "validation"
        elif index < 2 * n_val:
            fold_of[lesion] = "test"
        else:
            fold_of[lesion] = "train"

    folds = defaultdict(list)
    for record in records:
        folds[fold_of[record["lesion_id"]]].append(record)
    return folds


def gt_mask_index(archive):
    """Map ISIC image id -> member name inside the reference mask archive."""
    index = {}
    for name in archive.namelist():
        if not name.lower().endswith(".png"):
            continue
        found = re.search(r"(ISIC_\d+)", name)
        if found:
            index[found.group(1)] = name
    return index


def load_gt_mask(archive, name, size):
    """Boolean lesion mask for one image, at *size* (width, height)."""
    mask = Image.open(io.BytesIO(archive.read(name))).convert("L")
    if mask.size != size:
        # NEAREST keeps the mask binary; anything else would blur the
        # boundary into grey and silently shave the lesion edge.
        mask = mask.resize(size, Image.NEAREST)
    return np.array(mask) > 127
