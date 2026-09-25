"""Download the evaluation set used by eval_sweep.py.

Pulls N images per diagnostic class from the HAM10000 test split and writes
them to eval_images/, named `<class>_<ISIC id>.jpg` so eval_sweep.py can read
the expected label from the filename.

Images already present in sample_images/ are skipped, so the evaluation set
stays disjoint from the demo fixtures.

eval_images/ is gitignored (they're ~2MB of redownloadable data, and Flash
ships the project directory to every worker, where they'd never be read), so
run this once after cloning:

    python fetch_eval_images.py
    python fetch_eval_images.py --per-class 10 --split train
"""

import argparse
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

# Same mapping the training pipeline uses. Imported rather than copied so the
# eval set can never be labelled by a different rule than the training data.
from lesion_training.common import DX_TO_CLASS

DATASET = "marmal88/skin_cancer"


def existing_image_ids(*directories):
    """ISIC ids already on disk, so downloads don't duplicate them."""
    ids = set()
    for directory in directories:
        for path in Path(directory).glob("*.jpg"):
            # filenames look like `mel_ISIC_0024351.jpg`
            parts = path.stem.split("_", 1)
            if len(parts) == 2:
                ids.add(parts[1])
    return ids


def collect(per_class, split, skip_ids):
    """Stream the split until every class has `per_class` unseen images."""
    dataset = load_dataset(DATASET, split=split, streaming=True)
    buckets = defaultdict(list)

    for example in dataset:
        dx = example["dx"]
        if dx not in DX_TO_CLASS or example["image_id"] in skip_ids:
            continue
        if len(buckets[dx]) < per_class:
            buckets[dx].append(example)
        if all(len(buckets[d]) >= per_class for d in DX_TO_CLASS):
            break

    return buckets


def main():
    parser = argparse.ArgumentParser(description="Download the eval_sweep.py image set.")
    parser.add_argument("--per-class", type=int, default=5, help="Images per class (default 5)")
    parser.add_argument("--split", default="test", help="Dataset split (default test)")
    parser.add_argument("--out-dir", default="eval_images", help="Output directory")
    parser.add_argument(
        "--exclude-dir",
        default="sample_images",
        help="Directory whose images should not be duplicated",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    skip_ids = existing_image_ids(args.exclude_dir, out_dir)
    print(f"skipping {len(skip_ids)} image(s) already present")

    buckets = collect(args.per_class, args.split, skip_ids)

    saved = 0
    for dx, examples in sorted(buckets.items()):
        class_name = DX_TO_CLASS[dx]
        for example in examples:
            path = out_dir / f"{class_name}_{example['image_id']}.jpg"
            example["image"].convert("RGB").save(path, format="JPEG", quality=92)
            saved += 1
        print(f"  {class_name}: {len(examples)}")

    print(f"\nsaved {saved} images to {out_dir}/")
    if saved < args.per_class * len(DX_TO_CLASS):
        print("note: some classes had fewer unseen images than requested")


if __name__ == "__main__":
    main()
