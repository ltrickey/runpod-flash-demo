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
| `segment_worker.py` | GPU, queue-based (function) | SAM segmentation |
| `classify_worker.py` | GPU, queue-based (function) | BEiT classification |
| `pipeline.py` | CPU, load-balanced | Orchestrates segment → classify |
| `demo_client.py` | local script | Reads an image file, calls `/pipeline/analyze` |
| `sample_images/` | test fixtures | One real HAM10000 photo per diagnostic class |

`segment_worker.py` and `classify_worker.py` are plain function endpoints
(not class-based): `pipeline.py` chains them by directly importing and
`await`-ing them, following the pattern in
`flash-examples/01_getting_started/03_mixed_workers`. A class-based
`@Endpoint` (model loaded once in `__init__`) can't be called this way —
Flash's cross-worker chaining only supports functions — nor can it be called
from `Endpoint(id=...)` client mode, since that requires speaking an
undocumented internal wire protocol. Each call currently reloads the model
from Hugging Face; a documented module-level caching pattern exists
(`flash-examples/docs/cli/workflows.md`, "Reduce cold starts") but only
applies to `flash deploy`'s real container image, not `flash dev`'s
live/on-demand testing mode — worth re-testing once deployed.

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

## Testing Locally with `flash dev`

`flash dev` runs a local dev server that auto-discovers every `@Endpoint` in
the project and exposes it over HTTP on your machine — but for GPU/CPU
queue-based workers (`segment_worker`, `classify_worker`) it **provisions
real, billable Runpod serverless workers**, not free local compute. "Dev"
here means "a local server you can iterate against," not "runs on your own
hardware." Each worker scales to 0 (`workers=(0, 3)`) when idle, so you're
only charged for actual GPU time during a call, but every test invocation
does cost a small amount.

### Steps

```bash
cd runpod_trial
uv venv && source .venv/bin/activate   # or: python -m venv .venv && source .venv/bin/activate
uv sync                                 # or: pip install -r requirements.txt
flash login                             # authenticate once, or set RUNPOD_API_KEY in .env
flash dev
```

This prints the routes it discovered:

```
POST  /classify_worker/runsync  classify  QB
POST  /pipeline/analyze         analyze   LB
GET   /pipeline/health          health    LB
POST  /segment_worker/runsync   segment   QB
```

Visit **http://localhost:8888/docs** for interactive Swagger UI, or drive it
from the terminal:

```bash
# Easiest: run the full pipeline against a real sample image
python demo_client.py sample_images/mel_ISIC_0024351.jpg

# Or call the full pipeline directly with curl
curl -X POST http://localhost:8888/pipeline/analyze \
  -H "Content-Type: application/json" \
  -d '{"input_data": {"image_base64": "'"$(base64 -i sample_images/bcc_ISIC_0024431.jpg)"'"}}'

# Or test one stage in isolation
curl -X POST http://localhost:8888/segment_worker/runsync \
  -H "Content-Type: application/json" \
  -d '{"input": {"input_data": {"image_base64": "'"$(base64 -i sample_images/bcc_ISIC_0024431.jpg)"'"}}}'
```

The first call to a given worker takes longer (a real GPU is being spun up
and SAM/BEiT are downloaded from Hugging Face); subsequent calls to a warm
worker are faster, though each call currently still reloads the model
in-process (see the caching note above).

Press **Ctrl+C** to stop the server — this cleans up the Runpod endpoints
that `flash dev` provisioned during the session, so nothing keeps running
(and billing) after you stop.

### Checking on things independently

The Runpod console (Serverless tab) shows dev-provisioned endpoints prefixed
with `live-` (e.g. `live-segment_worker`) — useful for confirming a worker
actually spun up, checking logs, or seeing GPU/worker status if a request
seems stuck.

---

## Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended Python package manager):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

See "Testing Locally with `flash dev`" above for setup and run steps.

Tip: `flash dev --auto-provision` pre-deploys all endpoints on startup instead
of lazily on first request, eliminating cold-start delay on your *first*
test call (provisioned endpoints are cached and reused across restarts) —
but note this also means the GPU workers spin up immediately, not just when
you actually call them.

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
`segment_worker.py` and `classify_worker.py` are both function-based QB workers
(see the caching note above for why, and the current tradeoff on model reloads).

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
