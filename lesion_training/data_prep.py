"""Build the training dataset on the network volume."""

import os
import shutil
import urllib.request
import zipfile
from collections import defaultdict

import numpy as np
from PIL import Image

from lesion_training import common, masking


def prepare(n_images=0, reset=False):
    """Stream HAM10000 and write the SAM-masked and raw copy of every image.

    Two copies of each image are saved so the training runs see identical
    source images and differ only in preprocessing. That is what makes the
    masked-vs-raw comparison mean anything: without it, a difference could
    just as easily be the sample.

    Resumable: images already on the volume are skipped, so a run that hits
    the execution timeout can simply be submitted again. Worth having when
    there is no way to attach to a running job.
    """
    from datasets import load_dataset

    device = common.device()
    sam, processor = masking.load_sam(device)

    # Reset BEFORE creating directories -- doing it after would delete the
    # directories just created and every save would fail.
    if reset:
        # start clean -- resume-by-default is right for timeouts, but wrong
        # when the previous run produced a bad dataset
        for variant in common.VARIANTS:
            if os.path.isdir(f"{common.DATA_ROOT}/{variant}"):
                shutil.rmtree(f"{common.DATA_ROOT}/{variant}")
        if os.path.exists(common.MANIFEST):
            os.remove(common.MANIFEST)

    for variant in common.VARIANTS:
        for class_name in common.CLASSES:
            os.makedirs(f"{common.DATA_ROOT}/{variant}/{class_name}", exist_ok=True)

    records = common.load_manifest() if os.path.exists(common.MANIFEST) else []
    seen = {record["image_id"] for record in records}

    counts = defaultdict(int)
    for record in records:
        counts[record["label"]] += 1

    # The stream is ordered by class, so a global cap would take every image
    # from the first class. Cap per class instead, which only matters for
    # capped runs -- n_images=0 takes everything.
    per_class_cap = (n_images // len(common.CLASSES)) if n_images else 0

    dataset = load_dataset("marmal88/skin_cancer", split="train", streaming=True)
    for example in dataset:
        dx = example["dx"]
        if dx not in common.DX_TO_CLASS or example["image_id"] in seen:
            continue

        class_name = common.DX_TO_CLASS[dx]
        if per_class_cap:
            if counts[class_name] >= per_class_cap:
                if all(counts[c] >= per_class_cap for c in common.CLASSES):
                    break
                continue

        image = example["image"].convert("RGB")
        keep = masking.predict_mask(image, sam, processor, device)
        masked = image if keep is None else masking.black_out(image, keep)

        filename = f"{class_name}/{example['image_id']}.png"
        image.save(f"{common.DATA_ROOT}/raw/{filename}", format="PNG")
        masked.save(f"{common.DATA_ROOT}/masked/{filename}", format="PNG")

        counts[class_name] += 1
        records.append(
            {
                "image_id": example["image_id"],
                "file": filename,
                "label": class_name,
                # grouping key -- several images can share one lesion, and
                # splitting without grouping leaks near-duplicates across folds
                "lesion_id": example["lesion_id"],
                "masked": keep is not None,
            }
        )

        # checkpoint the manifest periodically; a timeout mid-run then costs
        # at most the last 200 images
        if len(records) % 200 == 0:
            common.save_manifest(records)

    common.save_manifest(records)

    return {
        # identifies which sampling logic actually ran, so a stale worker
        # serving older code is obvious from the response rather than inferred
        # from odd class counts
        "sampling": "per_class" if per_class_cap else "all",
        "per_class_cap": per_class_cap,
        "images": len(records),
        "per_class": dict(counts),
        "masked": sum(1 for r in records if r["masked"]),
        "passed_through": sum(1 for r in records if not r["masked"]),
    }


def download_gt_masks():
    """Fetch the reference mask archive to the volume, once."""
    if os.path.exists(common.GT_MASK_ZIP):
        return

    # Dataverse 403s urllib's default "Python-urllib/3.12" agent, so send a
    # browser one. Download to a temp path and rename only on success: a
    # partial file left at GT_MASK_ZIP would be treated as cached on the next
    # run and fail as a corrupt zip forever.
    request = urllib.request.Request(
        common.GT_MASK_URL, headers={"User-Agent": "Mozilla/5.0"}
    )
    partial = f"{common.GT_MASK_ZIP}.partial"
    with urllib.request.urlopen(request, timeout=300) as response:
        with open(partial, "wb") as handle:
            shutil.copyfileobj(response, handle)
    if not zipfile.is_zipfile(partial):
        os.remove(partial)
        raise RuntimeError(f"downloaded mask archive is not a zip: {common.GT_MASK_URL}")
    os.replace(partial, common.GT_MASK_ZIP)


def prepare_gt():
    """Write a ground-truth-masked copy of every image already prepared.

    Backfills rather than re-preparing: the raw images are already on the
    volume, so this reads those, applies the reference mask, and writes the
    `gtmasked` variant. No HF streaming and no SAM, which makes it minutes
    rather than hours -- and it guarantees all three variants come from
    byte-identical source images.

    Resumable the same way prepare() is: images already written are skipped,
    so a timeout costs nothing.
    """
    if not os.path.exists(common.MANIFEST):
        raise RuntimeError("no manifest -- run stage=prepare first")

    download_gt_masks()
    archive = zipfile.ZipFile(common.GT_MASK_ZIP)
    index = common.gt_mask_index(archive)

    for class_name in common.CLASSES:
        os.makedirs(f"{common.DATA_ROOT}/gtmasked/{class_name}", exist_ok=True)

    records = common.load_manifest()

    written, skipped, missing, coverages = 0, 0, 0, []
    for record in records:
        target = f"{common.DATA_ROOT}/gtmasked/{record['file']}"
        if os.path.exists(target):
            skipped += 1
            record["gt_masked"] = True
            continue

        name = index.get(record["image_id"])
        if name is None:
            missing += 1
            record["gt_masked"] = False
            continue

        image = Image.open(f"{common.DATA_ROOT}/raw/{record['file']}").convert("RGB")
        keep = common.load_gt_mask(archive, name, image.size)
        if not keep.any():
            missing += 1
            record["gt_masked"] = False
            continue

        # black out only -- identical to the SAM variant, so gtmasked differs
        # from masked in mask quality alone and in nothing else
        masking.black_out(image, keep).save(target, format="PNG")

        record["gt_masked"] = True
        coverages.append(float(np.mean(keep)))
        written += 1

    common.save_manifest(records)

    return {
        "written": written,
        "already_present": skipped,
        "no_ground_truth_mask": missing,
        "total": len(records),
        "mean_coverage": (sum(coverages) / len(coverages)) if coverages else None,
    }
