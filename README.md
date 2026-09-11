# SAM → BEiT Skin Lesion Analysis Pipeline

A two-stage inference pipeline on Runpod Flash serverless GPUs, based on the SAM→ViT
segmentation-then-classification approach from [Himel et al. (2024)](https://onlinelibrary.wiley.com/doi/10.1155/2024/3022192), built with public,
Apache-2.0 licensed models:

- **Segmentation** — [`facebook/sam-vit-base`](https://huggingface.co/facebook/sam-vit-base):
  finds the lesion in the source photo and crops to it.
- **Classification** — [`ALM-AHME/beit-large-patch16-224-finetuned-Lesion-Classification-HAM10000-AH-60-20-20`](https://huggingface.co/ALM-AHME/beit-large-patch16-224-finetuned-Lesion-Classification-HAM10000-AH-60-20-20):
  a BEiT vision transformer fine-tuned on HAM10000, classifying the crop into one of
  7 diagnostic classes (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`).

## Pipeline Architecture

```
image_base64
    |
segment_worker.py (GPU, SAM)
    - decode image, prompt SAM at the image center
    - crop to the highest-confidence mask's bounding box
    |
classify_worker.py (GPU, BEiT)
    - classify the cropped lesion
    - softmax over 7 HAM10000 classes
    |
{"label": ..., "confidence": ..., "all_scores": {...}}
```

`pipeline.py` is a thin load-balanced orchestrator that calls both endpoints in
sequence and exposes the whole thing as a single `POST /analyze` call.

### Pipeline Files

| File | Type | Role |
|------|------|------|
| `segment_worker.py` | GPU, queue-based (class) | SAM segmentation |
| `classify_worker.py` | GPU, queue-based (class) | BEiT classification |
| `pipeline.py` | CPU, load-balanced | Orchestrates segment → classify |
| `demo_client.py` | local script | Reads an image file, calls `/pipeline/analyze` |
| `sample_images/` | test fixtures | One real HAM10000 photo per diagnostic class |

### Demo

```bash
python demo_client.py sample_images/mel_ISIC_0024351.jpg
```

### Test the Pipeline

```bash
# Full pipeline: segment + classify in one call
curl -X POST http://localhost:8888/pipeline/analyze \
  -H "Content-Type: application/json" \
  -d '{"input_data": {"image_base64": "<base64-encoded-image>"}}'

# Individual stages
curl -X POST http://localhost:8888/segment_worker/runsync \
  -H "Content-Type: application/json" \
  -d '{"input": {"input_data": {"image_base64": "<base64-encoded-image>"}}}'

curl -X POST http://localhost:8888/classify_worker/runsync \
  -H "Content-Type: application/json" \
  -d '{"input": {"input_data": {"image_base64": "<base64-encoded-crop>"}}}'
```

### GPU Configuration

Both GPU workers use a small-model-sized fallback list, since SAM-base and
BEiT-large comfortably fit on 16-24GB of VRAM:

```python
gpu=[GpuGroup.ADA_24, GpuGroup.AMPERE_24, GpuGroup.AMPERE_16]
```

---

## Quick Start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended Python package manager):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Set up the project:

```bash
uv venv && source .venv/bin/activate
uv sync
flash login              # Authenticate with Runpod
flash dev
```

Or with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
flash login              # Authenticate with Runpod
flash dev
```

Server starts at **http://localhost:8888**. Visit **http://localhost:8888/docs** for interactive Swagger UI.

Use `flash dev --auto-provision` to pre-deploy all endpoints on startup, eliminating cold-start delays on first request. Provisioned endpoints are cached and reused across restarts.

When you stop the server with Ctrl+C, all endpoints provisioned during the session are automatically cleaned up.

## Project Structure

```
runpod_trial/
├── segment_worker.py   # GPU worker: SAM lesion segmentation
├── classify_worker.py  # GPU worker: BEiT HAM10000 classification
├── pipeline.py          # LB orchestrator: segment -> classify
├── demo_client.py      # Local script: call the pipeline with an image file
├── sample_images/      # Real HAM10000 test images, one per class
├── .env.example        # Environment variable template
├── requirements.txt    # Python dependencies
└── README.md
```

## Worker Types

### Queue-Based (QB) Workers

QB workers process jobs from a queue. Each call to `/runsync` sends a job and waits
for the result. Use QB for compute-heavy tasks that may take seconds to minutes.
`segment_worker.py` and `classify_worker.py` are both class-based QB workers: the
model loads once in `__init__` and is reused across requests on a warm worker.

### Load-Balanced (LB) Workers

LB workers expose standard HTTP endpoints (GET, POST, etc.) behind a load balancer.
Use LB for low-latency API endpoints that need horizontal scaling. `pipeline.py` is
an LB worker that orchestrates the two GPU workers above.

### Client Mode

Call an existing endpoint or a pre-built image without writing handler code:

```python
from runpod_flash import Endpoint

# connect to an existing endpoint by id
ep = Endpoint(id="ep-abc123")
job = await ep.run({"prompt": "hello"})
await job.wait()
print(job.output)

# deploy and call a pre-built image
ep = Endpoint(name="vllm", image="runpod/worker-vllm:stable-cuda12.1.0")
result = await ep.post("/v1/completions", {"prompt": "hello"})
```

## Adding New Workers

Create a new `.py` file with an `Endpoint`. `flash dev` auto-discovers all
`Endpoint` functions in the project.

```python
# my_worker.py
from runpod_flash import Endpoint, GpuType

@Endpoint(name="my_worker", gpu=GpuType.NVIDIA_GEFORCE_RTX_4090, dependencies=["transformers"])
async def predict(input_data: dict) -> dict:
    from transformers import pipeline
    pipe = pipeline("sentiment-analysis")
    return pipe(input_data["text"])[0]
```

Then run `flash dev` -- the new worker appears automatically.

## GPU Types

| Config                                    | Hardware          | VRAM   |
| ----------------------------------------- | ----------------- | ------ |
| `GpuType.ANY`                             | Any available GPU | varies |
| `GpuType.NVIDIA_GEFORCE_RTX_4090`         | RTX 4090          | 24 GB  |
| `GpuType.NVIDIA_GEFORCE_RTX_5090`         | RTX 5090          | 32 GB  |
| `GpuType.NVIDIA_RTX_6000_ADA_GENERATION`  | RTX 6000 Ada      | 48 GB  |
| `GpuType.NVIDIA_L4`                       | L4                | 24 GB  |
| `GpuType.NVIDIA_A100_80GB_PCIe`           | A100 PCIe         | 80 GB  |
| `GpuType.NVIDIA_A100_SXM4_80GB`           | A100 SXM4         | 80 GB  |
| `GpuType.NVIDIA_H100_80GB_HBM3`           | H100              | 80 GB  |
| `GpuType.NVIDIA_H200`                     | H200              | 141 GB |
| `GpuType.NVIDIA_B200`                     | B200              | 180 GB |

## CPU Types

Pass a CPU instance type string to `cpu=`:
- `"cpu3c-1-2"` -- 1 vCPU, 2 GB RAM
- `"cpu3c-4-8"` -- 4 vCPU, 8 GB RAM
- `"cpu3g-2-8"` -- 2 vCPU, 8 GB RAM
- `"cpu5g-4-16"` -- 4 vCPU, 16 GB RAM

Or use `CpuInstanceType` enum values.

## Authentication

Run `flash login` to authenticate via browser. This stores your API key in `~/.runpod/config.toml`.

Alternatively, set the `RUNPOD_API_KEY` environment variable or add it to `.env`:
```bash
cp .env.example .env   # Then edit .env with your key
```

Get your API key from [Runpod Settings](https://www.runpod.io/console/user/settings).
Learn more from our [Documentation](https://docs.runpod.io/get-started/api-keys).

## Environment Variables

```bash
# Authentication (optional if using flash login)
RUNPOD_API_KEY=your_api_key

# Optional
FLASH_HOST=localhost   # Server host (default: localhost)
FLASH_PORT=8888        # Server port (default: 8888)
LOG_LEVEL=INFO         # Logging level (default: INFO)
```

## Deploy

```bash
flash deploy
```
