# gpu serverless worker -- fine-tunes a ViT on lesion images, once per
# preprocessing variant.
#
# Uses ViT-B/32 (google/vit-base-patch32-224-in21k), the architecture family
# Himel et al. report using. Note their 96-97% accuracy is on a BINARY
# benign/malignant split of HAM10000; this trains the 7-class problem the
# rest of the pipeline uses, so the numbers are not comparable to theirs.
#
# Why this exists: masking the background before classification measurably
# hurt accuracy (see README). The hypothesis is a train/inference mismatch --
# the classifier had never seen masked input. This tests that by training on
# masked images so the distributions match.
#
# It trains THREE models, on copies of the same images that differ only in
# preprocessing, with everything else held identical. A single masked-trained
# model would not settle anything: a poor result could equally mean "masking
# is bad" or "a generic ImageNet backbone can't learn dermoscopy from this
# much data". Only the differences between the arms isolate the variables.
#
#   raw      -- untouched
#   masked   -- SAM, what the deployed pipeline actually produces
#   gtmasked -- Tschandl's expert masks, one per HAM10000 image
#
# gtmasked exists to answer the strongest objection to this experiment: SAM
# finds no coherent mask on ~29% of images, so "masking hurts" might only mean
# "our masks were bad". With expert masks that confound is gone -- gtmasked is
# the best case masking can possibly achieve here, so if it still loses to
# raw, the loss is inherent to masking rather than to SAM.
#
# Runs on Flash serverless rather than a Pod. That works because the
# execution timeout is configurable up to 7 days and a network volume
# (mounted at /runpod-volume) persists checkpoints independently of the
# worker. Two real constraints come with it: no interactive debugging, and
# the volume is datacenter-scoped so the endpoint is pinned to EU-RO-1.
#
# Single worker on purpose: concurrent writes to one network volume can
# corrupt data, and the account worker quota is nearly spent by the
# inference endpoints.
#
# Stages are selectable so the expensive SAM pass is not repeated on every
# training run:
#   stage="prepare"    -> mask the source train split with SAM (9,577
#                         images, or n_images capped), write masked + raw
#   stage="prepare_gt" -> backfill the ground-truth-masked variant from the
#                         raw images already on the volume (no SAM, minutes)
#   stage="train"      -> fine-tune one model per variant from that dataset;
#                         pass variants=["gtmasked"] to train just one
#   stage="both"       -> prepare, prepare_gt, then train
#   stage="evaluate"   -> score every trained model against every image
#                         variant (the full 3x3), no training
#   stage="segmentation_iou" -> IoU/Dice of our zero-shot SAM masks against
#                         the published ones, on the same test split
from runpod_flash import DataCenter, Endpoint, GpuGroup, NetworkVolume

volume = NetworkVolume(
    name="lesion-training",
    size=50,
    datacenter=DataCenter.EU_RO_1,
)


@Endpoint(
    name="train_worker",
    # The volume pins us to one datacenter, so keep the GPU pools wide --
    # a single pool in a single datacenter may simply fail to schedule.
    # AMPERE_16 excluded to leave headroom; ViT-B/32 itself is small.
    gpu=[GpuGroup.ADA_24, GpuGroup.AMPERE_24],
    workers=(0, 1),
    volume=volume,
    datacenter=DataCenter.EU_RO_1,
    # 6h. The default is 600s and the ceiling is 7 days. A SAM pass over the
    # full 9,577-image train split measured far slower than estimated, and a
    # cold start alone costs ~9min (GPU provisioning + volume mount + artifact
    # pull), so give it room -- prepare is resumable, but each resume pays
    # that cold start again.
    execution_timeout_ms=21_600_000,
    idle_timeout=60,
    dependencies=["transformers", "datasets", "accelerate", "pillow"],
)
async def train(input_data: dict) -> dict:
    """
    Fine-tune a ViT on HAM10000 images, once per variant.

    Input:
        stage: "prepare" | "prepare_gt" | "train" | "both" | "evaluate"
            | "segmentation_iou" (default "both")
        n_images: int - cap on source images, applied as n_images/7 per
            class; 0 for the whole train split (9,577)
        variants: list - which arms to train
            (default ["masked", "raw", "gtmasked"])
        epochs: int - training epochs (default 10)
        base_model: str - checkpoint to fine-tune (default ViT-B/32 in21k)
        batch_size: int - per-device batch size (default 32)
        learning_rate: float - default 2e-5

    Returns a dict describing what each stage did: the dataset manifest
    counts, and per-variant test accuracy and balanced accuracy.
    """
    import json
    import os
    import shutil
    from collections import defaultdict

    import numpy as np
    import torch
    from PIL import Image

    VOLUME = "/runpod-volume"
    DATA_ROOT = f"{VOLUME}/data"
    MANIFEST = f"{DATA_ROOT}/manifest.json"
    # masked   = SAM, what the deployed pipeline actually produces
    # raw      = untouched
    # gtmasked = expert ground-truth masks, the ceiling for masking
    VARIANTS = ("masked", "raw", "gtmasked")

    # Tschandl's expert lesion masks for HAM10000, one per image, from the
    # same Harvard Dataverse record as the images themselves. 10.8MB, no auth.
    # These exist to remove mask quality as a confound.
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

    stage = input_data.get("stage", "both")

    # 0 means "everything": the source dataset (marmal88/skin_cancer) train
    # split, 9,577 images. Otherwise the cap is applied per class, which also
    # rebalances: nv is 67% of the source split but capped to 500 like the
    # common classes. The reported results used 3500 -> 2,548 images.
    n_images = int(input_data.get("n_images", 0))

    # Himel et al. used 100 epochs, but on 6,000 augmented images for a binary
    # task. On an earlier, larger run (~2,850 training / 644 validation
    # images) that overfit: a 30-epoch run scored WORSE than a 5-epoch one on
    # both arms (raw balanced 73.7% -> 66.7%). More epochs also means
    # load_best_model_at_end picks from more checkpoints against the
    # validation set, which invites selection overfitting on top of training
    # overfitting. 10 is a middle ground between the two lengths measured.
    epochs = int(input_data.get("epochs", 10))

    # ViT-B/32 to match Himel et al., who used "the Google-based ViT patch-32
    # variant". Also ~4x fewer tokens than patch-16 at 224px (7x7 vs 14x14),
    # so training is markedly faster. The -in21k checkpoint ships without a
    # classification head, which suits fine-tuning to 7 classes.
    base_model = input_data.get("base_model", "google/vit-base-patch32-224-in21k")

    # batch size and learning rate follow Himel et al.
    batch_size = int(input_data.get("batch_size", 32))
    learning_rate = float(input_data.get("learning_rate", 2e-5))

    # -- masking ---------------------------------------------------------
    # Deliberately identical to segment_worker.segment: the training
    # distribution has to match what the deployed pipeline produces, which
    # is the entire point of this experiment. If the selection thresholds
    # change there, they must change here too.

    def select_mask(masks, scores, total_px):
        """Highest-IoU mask that is neither the whole frame nor fragmented."""
        min_coverage, max_coverage, min_solidity = 0.03, 0.90, 0.40
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

            if min_coverage <= coverage <= max_coverage and solidity >= min_solidity:
                viable.append((float(scores[i]), mask, bbox))

        return max(viable, key=lambda item: item[0]) if viable else None

    def mask_image(image, sam, processor, device):
        """Return (masked PIL image, was_masked)."""
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
        if chosen is None:
            return image, False

        _, mask, _bbox = chosen
        # black out only -- no crop, matching segment_worker exactly
        pixels = np.array(image)
        pixels[~mask] = 0
        return Image.fromarray(pixels), True

    def prepare_gt():
        """Write a ground-truth-masked copy of every image already prepared.

        Backfills rather than re-preparing: the raw images are already on the
        volume, so this reads those, applies the expert mask, and writes the
        `gtmasked` variant. No HF streaming and no SAM, which makes it minutes
        rather than hours -- and it guarantees all three variants come from
        byte-identical source images.

        Resumable the same way prepare() is: images already written are
        skipped, so a timeout costs nothing.
        """
        import io
        import re
        import urllib.request
        import zipfile

        if not os.path.exists(MANIFEST):
            raise RuntimeError("no manifest -- run stage=prepare first")

        if not os.path.exists(GT_MASK_ZIP):
            # Dataverse 403s urllib's default "Python-urllib/3.12" agent, so
            # send a browser one. Download to a temp path and rename only on
            # success: a partial file left at GT_MASK_ZIP would be treated as
            # cached on the next run and fail as a corrupt zip forever.
            request = urllib.request.Request(
                GT_MASK_URL, headers={"User-Agent": "Mozilla/5.0"}
            )
            partial = f"{GT_MASK_ZIP}.partial"
            with urllib.request.urlopen(request, timeout=300) as response:
                with open(partial, "wb") as handle:
                    shutil.copyfileobj(response, handle)
            if not zipfile.is_zipfile(partial):
                os.remove(partial)
                raise RuntimeError(f"downloaded mask archive is not a zip: {GT_MASK_URL}")
            os.replace(partial, GT_MASK_ZIP)

        archive = zipfile.ZipFile(GT_MASK_ZIP)
        index = {}
        for name in archive.namelist():
            if not name.lower().endswith(".png"):
                continue
            found = re.search(r"(ISIC_\d+)", name)
            if found:
                index[found.group(1)] = name

        for class_name in CLASSES:
            os.makedirs(f"{DATA_ROOT}/gtmasked/{class_name}", exist_ok=True)

        with open(MANIFEST) as handle:
            records = json.load(handle)

        written, skipped, missing, coverages = 0, 0, 0, []
        for record in records:
            target = f"{DATA_ROOT}/gtmasked/{record['file']}"
            if os.path.exists(target):
                skipped += 1
                record["gt_masked"] = True
                continue

            name = index.get(record["image_id"])
            if name is None:
                missing += 1
                record["gt_masked"] = False
                continue

            image = Image.open(f"{DATA_ROOT}/raw/{record['file']}").convert("RGB")
            mask = Image.open(io.BytesIO(archive.read(name))).convert("L")
            if mask.size != image.size:
                # NEAREST keeps the mask binary; anything else would blur the
                # boundary into grey and silently shave the lesion edge.
                mask = mask.resize(image.size, Image.NEAREST)

            keep = np.array(mask) > 127
            if not keep.any():
                missing += 1
                record["gt_masked"] = False
                continue

            # black out only -- no crop, matching segment_worker and the SAM
            # variant exactly, so gtmasked differs from masked in mask quality
            # alone and in nothing else
            pixels = np.array(image)
            pixels[~keep] = 0
            Image.fromarray(pixels).save(target, format="PNG")

            record["gt_masked"] = True
            coverages.append(float(keep.mean()))
            written += 1

        with open(MANIFEST, "w") as handle:
            json.dump(records, handle)

        return {
            "written": written,
            "already_present": skipped,
            "no_ground_truth_mask": missing,
            "total": len(records),
            "mean_coverage": (sum(coverages) / len(coverages)) if coverages else None,
        }

    def prepare():
        """Stream HAM10000 and write BOTH variants of every image.

        Two copies of each image are saved -- SAM-masked and untouched -- so
        the two training runs see identical source images and differ only in
        preprocessing. That is what makes the masked-vs-raw comparison mean
        anything: without it, a difference could just as easily be the sample.

        Resumable: images already on the volume are skipped, so a run that
        hits the execution timeout can simply be submitted again. Worth having
        when there is no way to attach to a running job.
        """
        from datasets import load_dataset
        from transformers import SamModel, SamProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
        sam.eval()
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

        # Reset BEFORE creating directories -- doing it after would delete the
        # directories just created and every save would fail.
        if input_data.get("reset"):
            # start clean -- resume-by-default is right for timeouts, but
            # wrong when the previous run produced a bad dataset
            for variant in VARIANTS:
                if os.path.isdir(f"{DATA_ROOT}/{variant}"):
                    shutil.rmtree(f"{DATA_ROOT}/{variant}")
            if os.path.exists(MANIFEST):
                os.remove(MANIFEST)

        for variant in VARIANTS:
            for class_name in CLASSES:
                os.makedirs(f"{DATA_ROOT}/{variant}/{class_name}", exist_ok=True)

        records = []
        if os.path.exists(MANIFEST):
            with open(MANIFEST) as handle:
                records = json.load(handle)
        seen = {record["image_id"] for record in records}

        counts = defaultdict(int)
        for record in records:
            counts[record["label"]] += 1

        # The stream is ordered by class, so a global cap would take every
        # image from the first class. Cap per class instead, which only
        # matters for small runs -- n_images=0 takes everything.
        per_class_cap = (n_images // len(CLASSES)) if n_images else 0

        dataset = load_dataset("marmal88/skin_cancer", split="train", streaming=True)
        for example in dataset:
            dx = example["dx"]
            if dx not in DX_TO_CLASS or example["image_id"] in seen:
                continue

            class_name = DX_TO_CLASS[dx]
            if per_class_cap:
                if counts[class_name] >= per_class_cap:
                    if all(counts[c] >= per_class_cap for c in CLASSES):
                        break
                    continue

            image = example["image"].convert("RGB")
            masked, was_masked = mask_image(image, sam, processor, device)

            filename = f"{class_name}/{example['image_id']}.png"
            image.save(f"{DATA_ROOT}/raw/{filename}", format="PNG")
            masked.save(f"{DATA_ROOT}/masked/{filename}", format="PNG")

            counts[class_name] += 1
            records.append(
                {
                    "image_id": example["image_id"],
                    "file": filename,
                    "label": class_name,
                    # grouping key -- several images can share one lesion, and
                    # splitting without grouping leaks near-duplicates across
                    # folds (the flaw we suspect in the community checkpoint)
                    "lesion_id": example["lesion_id"],
                    "masked": was_masked,
                }
            )

            # checkpoint the manifest periodically; a timeout mid-run then
            # costs at most the last 200 images
            if len(records) % 200 == 0:
                with open(MANIFEST, "w") as handle:
                    json.dump(records, handle)

        with open(MANIFEST, "w") as handle:
            json.dump(records, handle)

        return {
            # identifies which sampling logic actually ran, so a stale worker
            # serving older code is obvious from the response rather than
            # inferred from odd class counts
            "sampling": "per_class" if per_class_cap else "all",
            "per_class_cap": per_class_cap,
            "images": len(records),
            "per_class": dict(counts),
            "masked": sum(1 for r in records if r["masked"]),
            "passed_through": sum(1 for r in records if not r["masked"]),
        }

    # -- training --------------------------------------------------------

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

    def segmentation_iou():
        """Score the pipeline's zero-shot SAM masks against the published ones.

        Himel et al. report IoU 96.01% from a segmenter *trained* on these
        masks. This pipeline prompts SAM zero-shot with a centre point, so
        this measures the gap between the two approaches on identical images.

        Scored on the same seeded test split as every accuracy number, so the
        IoU and the classification results describe the same 408 images.

        Fallbacks (no coherent mask) are reported both ways: counted as IoU 0,
        which is what the pipeline effectively delivers, and excluded, which
        is how well SAM does when it commits to a lesion at all. Reporting
        only the second would flatter it.
        """
        import io
        import re
        import zipfile

        from transformers import SamModel, SamProcessor

        if not os.path.exists(GT_MASK_ZIP):
            return {"error": f"no mask archive at {GT_MASK_ZIP} -- run stage=prepare_gt"}

        archive = zipfile.ZipFile(GT_MASK_ZIP)
        index = {}
        for name in archive.namelist():
            if not name.lower().endswith(".png"):
                continue
            found = re.search(r"(ISIC_\d+)", name)
            if found:
                index[found.group(1)] = name

        with open(MANIFEST) as handle:
            records = json.load(handle)
        test_records = split_by_lesion(records)["test"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
        sam.eval()
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

        ious, dices, fell_back, missing = [], [], 0, 0
        for record in test_records:
            name = index.get(record["image_id"])
            if name is None:
                missing += 1
                continue

            image = Image.open(f"{DATA_ROOT}/raw/{record['file']}").convert("RGB")
            truth = Image.open(io.BytesIO(archive.read(name))).convert("L")
            if truth.size != image.size:
                truth = truth.resize(image.size, Image.NEAREST)
            gt = np.array(truth) > 127

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
            if chosen is None:
                fell_back += 1
                ious.append(0.0)
                dices.append(0.0)
                continue

            _, predicted, _bbox = chosen
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
            "himel_reported_iou": 0.9601,
        }

    def score_model(model_variant, data_variant):
        """Score a trained checkpoint against one image variant.

        The training runs only ever test a model on the preprocessing it was
        trained for -- the diagonal of a 2x2. This fills the off-diagonal, so
        the two candidate explanations can be separated:

          - a raw-trained model scored on masked images measures the
            *mismatch* cost (input unlike anything it trained on)
          - a masked-trained model scored on masked images measures what is
            achievable once the mismatch is removed

        If removing the mismatch still leaves masked behind raw, the loss is
        information, not distribution.
        """
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        checkpoint = f"{VOLUME}/models/vit-base-p32-{model_variant}"
        if not os.path.isdir(checkpoint):
            return {"error": f"no checkpoint at {checkpoint} -- train it first"}

        data_dir = f"{DATA_ROOT}/{data_variant}"
        processor = AutoImageProcessor.from_pretrained(checkpoint)
        model = AutoModelForImageClassification.from_pretrained(checkpoint)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()

        with open(MANIFEST) as handle:
            records = json.load(handle)
        # the same seeded split as training, so "test" means the same images
        test_records = split_by_lesion(records)["test"]

        label2id = {name: index for index, name in enumerate(CLASSES)}
        correct = 0
        per_class_right = defaultdict(int)
        per_class_total = defaultdict(int)

        for start in range(0, len(test_records), batch_size):
            chunk = test_records[start : start + batch_size]
            images = [
                Image.open(f"{data_dir}/{record['file']}").convert("RGB")
                for record in chunk
            ]
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                predicted = model(**inputs).logits.argmax(dim=-1).cpu().tolist()

            for record, prediction in zip(chunk, predicted):
                expected = label2id[record["label"]]
                per_class_total[expected] += 1
                if prediction == expected:
                    correct += 1
                    per_class_right[expected] += 1

        recalls = [
            per_class_right[index] / per_class_total[index]
            for index in per_class_total
        ]
        return {
            "model_trained_on": model_variant,
            "scored_against": data_variant,
            "n": len(test_records),
            "accuracy": correct / len(test_records),
            "balanced_accuracy": float(np.mean(recalls)),
        }

    def run_training(variant):
        """Fine-tune on one variant. Everything except the image directory is
        identical between runs, so the difference in results is the masking."""
        from transformers import (
            AutoImageProcessor,
            AutoModelForImageClassification,
            Trainer,
            TrainingArguments,
        )

        data_dir = f"{DATA_ROOT}/{variant}"
        # named after the architecture actually used, not the base model this
        # started life with
        output_dir = f"{VOLUME}/models/vit-base-p32-{variant}"

        with open(MANIFEST) as handle:
            records = json.load(handle)
        folds = split_by_lesion(records)

        processor = AutoImageProcessor.from_pretrained(base_model)
        label2id = {name: index for index, name in enumerate(CLASSES)}

        # A plain torch Dataset rather than datasets.Dataset on purpose.
        # datasets fingerprints every Dataset by dill-pickling it, and dill
        # cannot serialise objects defined in Flash's exec'd function
        # namespace -- it assumes the code lives in an importable module.
        # Trainer accepts any torch Dataset, so this sidesteps it entirely.
        class LesionImages(torch.utils.data.Dataset):
            def __init__(self, fold_records):
                self.records = fold_records

            def __len__(self):
                return len(self.records)

            def __getitem__(self, index):
                record = self.records[index]
                image = Image.open(f"{data_dir}/{record['file']}").convert("RGB")
                encoded = processor(images=image, return_tensors="pt")
                return {
                    "pixel_values": encoded["pixel_values"][0],
                    "labels": label2id[record["label"]],
                }

        splits = {name: LesionImages(fold) for name, fold in folds.items()}

        model = AutoModelForImageClassification.from_pretrained(
            base_model,
            num_labels=len(CLASSES),
            id2label={index: name for name, index in label2id.items()},
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

        def compute_metrics(prediction):
            # HAM10000 is heavily imbalanced (~67% nv), so plain accuracy
            # flatters a model that just predicts the majority class.
            # balanced_accuracy is the mean per-class recall, which doesn't.
            predicted = np.argmax(prediction.predictions, axis=1)
            labels = prediction.label_ids
            recalls = [
                (predicted[labels == index] == index).mean()
                for index in range(len(CLASSES))
                if (labels == index).any()
            ]
            return {
                "accuracy": float((predicted == labels).mean()),
                "balanced_accuracy": float(np.mean(recalls)),
            }

        arguments = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            # select on balanced accuracy, not raw -- with ~37% nv in this
            # set, plain accuracy would favour a checkpoint that leans on the
            # majority class
            metric_for_best_model="balanced_accuracy",
            logging_steps=25,
            bf16=torch.cuda.is_available(),
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=arguments,
            train_dataset=splits["train"],
            eval_dataset=splits["validation"],
            compute_metrics=compute_metrics,
        )
        trainer.train()

        test_metrics = trainer.evaluate(splits["test"])
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)

        return {
            "variant": variant,
            "base_model": base_model,
            "epochs": epochs,
            "fold_sizes": {name: len(records) for name, records in folds.items()},
            "test_accuracy": test_metrics.get("eval_accuracy"),
            "test_balanced_accuracy": test_metrics.get("eval_balanced_accuracy"),
            "output_dir": output_dir,
        }

    try:
        result = {"status": "success", "stage": stage}

        if stage in ("prepare", "both"):
            result["prepare"] = prepare()

        # Backfills the ground-truth-masked variant from the raw images that
        # prepare() just wrote, so it has to run after it.
        if stage in ("prepare_gt", "both"):
            result["prepare_gt"] = prepare_gt()

        if stage in ("train", "both"):
            # default to training both arms; the comparison between them is
            # the experiment, a single arm on its own proves little
            variants = input_data.get("variants", list(VARIANTS))
            result["train"] = {v: run_training(v) for v in variants}

        if stage == "segmentation_iou":
            result["segmentation_iou"] = segmentation_iou()

        if stage == "evaluate":
            # full 3x3: every trained model against every image variant
            result["evaluate"] = {
                f"{model_variant}_model_on_{data_variant}_images": score_model(
                    model_variant, data_variant
                )
                for model_variant in VARIANTS
                for data_variant in VARIANTS
            }

        return result

    except Exception as error:
        import traceback

        return {"status": "error", "error": str(error), "traceback": traceback.format_exc()}
