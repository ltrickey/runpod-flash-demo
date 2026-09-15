"""Stage implementations for train_worker.py.

    common             paths, labels, manifest, the lesion-grouped split
    masking            SAM mask selection and application
    data_prep          prepare (SAM + raw copies), prepare_gt (reference masks)
    vit                fine-tune and score the ViT classifiers
    segmentation_eval  IoU of SAM's masks against the reference masks

Imports within the package are absolute (`from lesion_training import ...`).
Flash's local-module resolver follows relative imports too, but raises if one
can't be resolved to a local file; absolute imports keep that path simple.
"""
