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
#   gtmasked -- Tschandl's reference masks, one per HAM10000 image
#
# gtmasked exists to answer the strongest objection to this experiment: SAM
# finds no coherent mask on ~29% of images, so "masking hurts" might only mean
# "our masks were bad". With reference masks that confound is gone -- gtmasked
# is the best case masking can possibly achieve here, so if it still loses to
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
# This file is only the endpoint and stage dispatch; the stages live in
# lesion_training/. They're imported inside the function body so both paths
# get them: `flash deploy` bundles the whole project directory, and
# `flash dev` ships the function source plus the local modules that source
# imports. Module-level state in this file would not reach a `flash dev`
# worker.
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
    import traceback

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

    try:
        # inside the try, so a worker missing the package returns a traceback
        # instead of an opaque failure
        from lesion_training import common, data_prep, segmentation_eval, vit

        result = {"status": "success", "stage": stage}

        if stage in ("prepare", "both"):
            result["prepare"] = data_prep.prepare(
                n_images=n_images, reset=bool(input_data.get("reset"))
            )

        # Backfills the ground-truth-masked variant from the raw images that
        # prepare() just wrote, so it has to run after it.
        if stage in ("prepare_gt", "both"):
            result["prepare_gt"] = data_prep.prepare_gt()

        if stage in ("train", "both"):
            # default to training every arm; the comparison between them is
            # the experiment, a single arm on its own proves little
            variants = input_data.get("variants", list(common.VARIANTS))
            result["train"] = {
                v: vit.run_training(v, base_model, epochs, batch_size, learning_rate)
                for v in variants
            }

        if stage == "segmentation_iou":
            result["segmentation_iou"] = segmentation_eval.segmentation_iou()

        if stage == "evaluate":
            # full 3x3: every trained model against every image variant
            result["evaluate"] = {
                f"{model_variant}_model_on_{data_variant}_images": vit.score_model(
                    model_variant, data_variant, batch_size
                )
                for model_variant in common.VARIANTS
                for data_variant in common.VARIANTS
            }

        return result

    except Exception as error:
        return {"status": "error", "error": str(error), "traceback": traceback.format_exc()}
