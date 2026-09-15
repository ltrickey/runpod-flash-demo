"""Fine-tune the ViT classifiers and score them against any image variant."""

import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from lesion_training import common


class LesionImages(torch.utils.data.Dataset):
    """One fold of one image variant, encoded for the ViT.

    A plain torch Dataset rather than datasets.Dataset. datasets fingerprints
    every Dataset by dill-pickling it, which failed on objects defined inside
    Flash's exec'd function namespace back when this code lived in the endpoint
    body. Trainer accepts any torch Dataset, so there is no reason to go back.
    """

    def __init__(self, records, data_dir, processor, label2id):
        self.records = records
        self.data_dir = data_dir
        self.processor = processor
        self.label2id = label2id

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(f"{self.data_dir}/{record['file']}").convert("RGB")
        encoded = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"][0],
            "labels": self.label2id[record["label"]],
        }


def label_maps():
    label2id = {name: index for index, name in enumerate(common.CLASSES)}
    return label2id, {index: name for name, index in label2id.items()}


def compute_metrics(prediction):
    # The source split is heavily imbalanced (~67% nv), and even after the
    # per-class cap the rare classes are a few dozen images, so plain accuracy
    # flatters a model that leans on the common classes. balanced_accuracy is
    # the mean per-class recall, which doesn't.
    predicted = np.argmax(prediction.predictions, axis=1)
    labels = prediction.label_ids
    recalls = [
        (predicted[labels == index] == index).mean()
        for index in range(len(common.CLASSES))
        if (labels == index).any()
    ]
    return {
        "accuracy": float((predicted == labels).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
    }


def run_training(variant, base_model, epochs, batch_size, learning_rate):
    """Fine-tune on one variant. Everything except the image directory is
    identical between runs, so the difference in results is the masking."""
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        Trainer,
        TrainingArguments,
    )

    data_dir = f"{common.DATA_ROOT}/{variant}"
    output_dir = common.checkpoint_dir(variant)
    folds = common.split_by_lesion(common.load_manifest())

    processor = AutoImageProcessor.from_pretrained(base_model)
    label2id, id2label = label_maps()
    splits = {
        name: LesionImages(fold, data_dir, processor, label2id)
        for name, fold in folds.items()
    }

    model = AutoModelForImageClassification.from_pretrained(
        base_model,
        num_labels=len(common.CLASSES),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

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
        # select on balanced accuracy, not raw -- plain accuracy would favour
        # a checkpoint that leans on the common classes
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


def score_model(model_variant, data_variant, batch_size):
    """Score a trained checkpoint against one image variant.

    A training run only ever tests a model on the preprocessing it was trained
    for -- the diagonal of the grid. This fills the off-diagonal, so the two
    candidate explanations can be separated:

      - a raw-trained model scored on masked images measures the *mismatch*
        cost (input unlike anything it trained on)
      - a masked-trained model scored on masked images measures what is
        achievable once the mismatch is removed

    If removing the mismatch still leaves masked behind raw, the loss is
    information, not distribution.
    """
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    checkpoint = common.checkpoint_dir(model_variant)
    if not os.path.isdir(checkpoint):
        return {"error": f"no checkpoint at {checkpoint} -- train it first"}

    data_dir = f"{common.DATA_ROOT}/{data_variant}"
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModelForImageClassification.from_pretrained(checkpoint)
    device = common.device()
    model.to(device).eval()

    # the same seeded split as training, so "test" means the same images
    test_records = common.split_by_lesion(common.load_manifest())["test"]

    label2id, _ = label_maps()
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
        per_class_right[index] / per_class_total[index] for index in per_class_total
    ]
    return {
        "model_trained_on": model_variant,
        "scored_against": data_variant,
        "n": len(test_records),
        "accuracy": correct / len(test_records),
        "balanced_accuracy": float(np.mean(recalls)),
    }
