# Sample Images

Seven real HAM10000 lesion photos, one per diagnostic class, for demoing and
testing the pipeline (`segment_worker.py` -> `classify_worker.py` -> `pipeline.py`).

| File | Class | Meaning |
|------|-------|---------|
| `akiec_*.jpg` | akiec | Actinic keratoses / intraepithelial carcinoma |
| `bcc_*.jpg`   | bcc   | Basal cell carcinoma |
| `bkl_*.jpg`   | bkl   | Benign keratosis-like lesions |
| `df_*.jpg`    | df    | Dermatofibroma |
| `mel_*.jpg`   | mel   | Melanoma |
| `nv_*.jpg`    | nv    | Melanocytic nevi |
| `vasc_*.jpg`  | vasc  | Vascular lesions |

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
