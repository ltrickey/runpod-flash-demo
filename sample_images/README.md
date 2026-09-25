# Sample Images

Seven real HAM10000 lesion photos, one per diagnostic class, used as demo
fixtures for the pipeline.

These are for *showing* the pipeline on a single image. For measuring
accuracy, use `../eval_images/` instead — 35 held-out images (5 per class,
no overlap with these seven) scored by `eval_sweep.py`. Seven images is far
too small a sample to draw conclusions from; an early 7-image run suggested
near-perfect accuracy that the larger set did not bear out.

| File | Class | Meaning | Status |
|------|-------|---------|--------|
| `akiec_*.jpg` | akiec | Actinic keratoses / intraepithelial carcinoma | Pre-malignant / in situ |
| `bcc_*.jpg`   | bcc   | Basal cell carcinoma | **Malignant** |
| `bkl_*.jpg`   | bkl   | Benign keratosis-like lesions | Benign |
| `df_*.jpg`    | df    | Dermatofibroma | Benign |
| `mel_*.jpg`   | mel   | Melanoma | **Malignant** |
| `nv_*.jpg`    | nv    | Melanocytic nevi | Benign |
| `vasc_*.jpg`  | vasc  | Vascular lesions | Benign |

Status is the dataset's diagnostic category, not a clinical judgement about any
individual image. This pipeline predicts the 7-class label; it does not decide
whether something is cancer.

Worth knowing when comparing against published binary results: across the full
10,015-image dataset the counts are nv 6,705 / mel 1,113 / bkl 1,099 / bcc 514 /
akiec 327 / vasc 142 / df 115 (counted from `HAM10000_metadata.tab` on Harvard
Dataverse). Himel et al. report a binary split of 6,705 benign vs 3,310
malignant -- exactly `nv` against every other class, which puts the benign
`bkl`, `df` and `vasc` (1,356 images) on the malignant side. Their task is
closer to "is this a mole?" than "is this cancer?".

Sourced from the [`marmal88/skin_cancer`](https://huggingface.co/datasets/marmal88/skin_cancer)
Hugging Face mirror of the HAM10000 dataset (Tschandl et al., 2018), used here under
its **CC BY-NC 4.0** license for non-commercial demo/educational purposes. Filenames
retain the original `ISIC_*` image IDs for traceability.

## Quick test

```bash
python3 - <<'PY'
import base64
with open("sample_images/mel_ISIC_0024351.jpg", "rb") as f:
    print(base64.b64encode(f.read()).decode())
PY
```

Pipe that base64 string into any of the `curl` examples in the top-level README.
