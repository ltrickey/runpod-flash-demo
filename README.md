# SAM → ViT Skin Lesion Analysis Pipeline

A two-stage inference pipeline on Runpod Flash serverless GPUs — SAM for
segmentation, then a vision transformer for classification — loosely inspired by
[Himel et al. (2024)](https://onlinelibrary.wiley.com/doi/10.1155/2024/3022192),
built with public, Apache-2.0 licensed models:

- **Segmentation** — [`facebook/sam-vit-base`](https://huggingface.co/facebook/sam-vit-base):
  finds the lesion in the source photo and blacks out everything else.
- **Classification** — a ViT-B/32 fine-tuned **in this repo** on HAM10000 by
  `train_worker.py`, predicting one of 7 diagnostic classes (`akiec`, `bcc`,
  `bkl`, `df`, `mel`, `nv`, `vasc`).

The classifier is trained here rather than pulled off the shelf. That is the
point of the project: the training data can be controlled, so the effect of
the segmentation stage can actually be measured instead of guessed at. It
starts from `google/vit-base-patch32-224-in21k` — generic ImageNet weights,
the architecture family Himel et al. report — and is fine-tuned on masked
lesion images so the classifier sees the same preprocessing the pipeline
produces.

**On "inspired by" rather than "implementing":** Himel et al. do describe how
the segmentation output reaches the classifier, but the description admits two
readings. Their algorithm block runs:

```
i.   I     = resize I_dermatoscopy into 224x224-pixel
ii.  I_seg = apply SegNetSAM on I
iii. I_seg = apply binary mask on I_seg
     ViT input: I_seg
```

and the prose says "After the SAM output is generated, it is converted to
binary masking. The processed output images are given to the ViT model for
classification." Elsewhere they describe binary masking as "the white area
represents the cancerous area and the black portion represents irrelevant
areas (skin, background, etc.)", and their segmentation figure shows exactly
that — a white silhouette on black.

So the ViT input is either:

| reading | ViT sees | consequence |
|---------|----------|-------------|
| **(a) silhouette** | the binary mask itself, white on black | lesion **shape only** — all colour and texture discarded |
| **(b) masked RGB** | lesion pixels kept, background blacked out | colour, texture and shape |

"Converted to binary masking", the white/black description, and the figure all
point at **(a)**; the phrase "apply binary mask on I_seg" is the one piece of
evidence for **(b)**, since applying a mask *to* an image normally preserves
the pixels underneath.

**This pipeline implements (b)**, which is the more generous reading — it keeps
strictly more information than (a), so it gives segmentation the better chance
of paying off. That choice is the design decision this project investigates.
Under reading (a) the classifier would be separating melanoma from naevi
without ever seeing pigment, which is a far stronger claim than this repo
tests.

Their headline 96-97% is also a **binary** benign/malignant result, while this
pipeline does 7-class, so the numbers are not comparable. That accuracy is
additionally inflated by augmenting *before* the train/test split, which leaks
near-duplicate images across the boundary.

## Pipeline Architecture

```
image_base64
    |
segment_worker.py (GPU, SAM)
    - decode image, prompt SAM at the image center
    - pick a coherent lesion mask (coverage + solidity filtered)
    - black out non-lesion pixels, keeping the original framing
    |
classify_worker.py (GPU, fine-tuned ViT)
    - classify the masked lesion
    - softmax over 7 HAM10000 classes
    |
{"label": ..., "confidence": ..., "all_scores": {...},
 "segmentation": {"applied": ..., "segmented_image_base64": ..., ...}}
```

### How the SAM mask is chosen

SAM returns three candidate masks at different granularities. Taking the
highest IoU naively selects the **whole-frame** mask on most dermoscopy
images, which silently makes segmentation a no-op (the classifier just sees
the original image). Candidates are therefore filtered on two metrics before
preferring highest IoU:

| Metric | Meaning | Accept |
|--------|---------|--------|
| coverage | fraction of the frame the mask occupies | 3%–90% |
| solidity | fraction of its own bbox the mask fills | ≥ 0.40 |

Solidity is what rejects **fragmented** masks: scattered hairs/streaks can
pass the coverage check while being useless to classify. Measured on the
sample images, real lesions score 0.42–0.78 solidity; streak masks score
0.07–0.21.

If no candidate is coherent, the original image is passed through unmasked
and `segmentation.applied` is `false` — better than handing the classifier
garbage. On the seven sample images, 5 segment cleanly and 2 (`bcc`, `nv`)
fall back.

A caveat worth knowing: HAM10000 images are already cropped and centered on
the lesion, so segmentation legitimately won't shrink them dramatically —
SAM's contribution here is masking out surrounding skin and vignetting. On
diffuse lesions with no crisp boundary, SAM's mask is weak regardless of
prompt (a box prompt was tested and largely returns whatever box you give
it, so the canonical point prompt is used instead).

### Measuring whether masking helps

Everything here is measured against the pipeline's own classifier, which is
fine-tuned in this repo (see below) — there is no third-party checkpoint to
attribute results to.

The pipeline exposes two complete **arms** so it can be measured against
itself. An arm is a preprocessing choice *and* the model trained for it,
always paired:

| `/analyze` input | behaviour |
|------------------|-----------|
| `arm: "raw"` (default) | SAM skipped entirely; raw-trained model sees the raw image |
| `arm: "masked"` | SAM blacks out the background; masked-trained model sees the masked image |

They are paired deliberately. Feeding a masked image to the raw-trained model
measures a distribution mismatch rather than the value of segmentation, and
makes masking look worse than it is.

`demo_client.py` runs both by default; `eval_sweep.py` scores them across
a directory of labelled images:

```bash
python fetch_eval_images.py          # one-time: download 35 held-out images
python eval_sweep.py --url <url> --dir eval_images
```

`eval_images/` is gitignored — the images are redownloadable, and Flash ships
the project directory to every worker, where they would never be read. Sweep
output lands in `results/`, also gitignored.

### Does training on masked images fix it?

If masking hurts only because the classifier never saw masked input, then
training on masked images should close the gap. `train_worker.py` tests that
by fine-tuning **two** models — one on SAM-masked images, one on the same
images untouched — with identical data, lesion-grouped split, and
hyperparameters. A single masked-trained model would settle nothing: a poor
score could equally mean "masking is bad" or "a generic ImageNet backbone
can't learn dermoscopy from this much data". Only the difference between the
arms isolates masking.

**Both arms are fine-tuned on skin cancer images.** Each starts from the same
generic ImageNet checkpoint (`google/vit-base-patch32-224-in21k`, the
architecture family Himel et al. report) and is then fine-tuned on HAM10000 —
one on the masked copies, one on the identical images unmasked. Each is then
evaluated on the preprocessing it was trained for. Neither is a plain
ImageNet model at evaluation time; the only variable is masking.

Starting from the generic checkpoint rather than the community HAM10000 one
is deliberate: that checkpoint's model card doesn't say what it was trained
on, so building on it would make any result hard to attribute.

This is what an earlier version of this project could not do. It classified
with an off-the-shelf community checkpoint fine-tuned on (probably unmasked)
data, so masked input was off-distribution for it by construction. That setup
can only answer "should I mask before calling *this* model?", not "is masking
inherently harmful?" — which is why the checkpoint was dropped and both arms
are now trained here. No third-party classifier remains in the pipeline.

**Results:** see [`results/training_report.md`](results/), generated by
`training_report.py` from the saved outputs of each stage. It carries the
dataset composition, per-arm scores, the full model × input grid, and a
decomposition of where the accuracy goes.

Numbers are deliberately not duplicated here. Earlier revisions of this README
carried measurements that silently went stale when the preprocessing changed,
which is exactly the failure the report format is meant to avoid.

Balanced accuracy (mean per-class recall) is reported alongside raw accuracy
because the dataset is imbalanced enough that a majority-class predictor would
look respectable on accuracy alone.

`pipeline.py` is a thin load-balanced orchestrator that calls both endpoints in
sequence and exposes the whole thing as a single `POST /analyze` call.

### Pipeline Files

| File | Type | Role |
|------|------|------|
| `segment_worker.py` | GPU, queue-based (function) | SAM segmentation |
| `classify_worker.py` | GPU, queue-based (function) | ViT classification (self-trained) |
| `pipeline.py` | CPU, load-balanced | Orchestrates segment → classify |
| `demo_client.py` | local script | Runs one image through both arms, saves what each classified |
| `eval_sweep.py` | local script | Scores a whole directory across both arms |
| `pipeline_client.py` | local module | Shared request/retry helpers for the two clients above |
| `training_report.py` | local script | Builds `results/training_report.md` from the stage outputs |
| `train_worker.py` | GPU, queue-based (function) | Fine-tunes masked and raw arms on a network volume |
| `train_client.py` | local script | Submits training jobs and polls to completion |
| `fetch_eval_images.py` | local script | Downloads the eval set |
| `sample_images/` | test fixtures | One real HAM10000 photo per diagnostic class |
| `eval_images/` | eval set | 35 held-out images (5 per class), gitignored |

`segment_worker.py` and `classify_worker.py` are plain function endpoints
(not class-based): `pipeline.py` chains them by directly importing and
`await`-ing them, following the pattern in
`flash-examples/01_getting_started/03_mixed_workers`. A class-based
`@Endpoint` (model loaded once in `__init__`) can't be called this way —
Flash's cross-worker chaining only supports functions — nor can it be called
from `Endpoint(id=...)` client mode, since that requires speaking an
undocumented internal wire protocol.

### The network volume is load-bearing

`classify_worker` reads its weights from `/runpod-volume/models/vit-base-p32-{raw,masked}`
— the same volume `train_worker` writes them to. **Deleting the `lesion-training`
volume breaks inference**, not just training; the endpoint returns
`no checkpoint at ...` until the training stages are re-run.

The volume is datacenter-scoped, so attaching it pins `classify_worker` to
`EU-RO-1` instead of the eleven datacenters it could otherwise schedule
across. That's a real capacity cost — during testing this endpoint was the
first to end up `THROTTLED` when the account's worker quota was tight. The
alternative is baking the weights into the deploy artifact, which avoids the
pin but adds ~350MB to every deploy and a manual re-fetch after each retrain.

### Model caching: works deployed, breaks in dev

Both workers load their model inside the endpoint function on **every call**.
That's deliberate, and the reason is worth knowing.

`flash-examples/docs/cli/workflows.md` ("Reduce cold starts") recommends
caching the model in a module-level global:

```python
_model = None

@Endpoint(...)
async def infer(payload: dict) -> dict:
    global _model
    if _model is None:
        _model = load_model()
```

**That pattern behaves differently depending on how you run it:**

| | Module-level globals | Why |
|---|---|---|
| `flash dev` | ✗ `NameError` | Live/on-demand provisioning ships only the decorated function's *isolated source*, without surrounding module state (see `runpod_flash.endpoint._is_live_provisioning`) |
| `flash deploy` | ✓ works | The whole file is baked into the container image and imported normally, so module state persists across requests on a warm worker |

It was tried here and measured in the deployed worker logs: the first request
logs `Loading weights: 100%|██████████| 314/314` and takes ~12s; the next
request logs no weight loading at all and takes ~1.4s — roughly a 9x speedup.

**It was then removed on purpose.** Keeping it would have meant `flash dev`
raising `NameError` on every call, breaking local development, in exchange
for a speedup that this demo doesn't need. The tradeoff is that every
`/analyze` call now pays the model load. If you want the speedup in a
deployed-only setup, add the globals back — just don't expect `flash dev` to
work afterwards.

This behaviour appears to be undocumented. `docs.runpod.io/flash/apps/build-app`
doesn't mention it, and the only `global` in `docs/cli/` is the snippet
recommending the pattern. The one place stating the underlying rule —
"only local variables, parameters, and internal imports work" — is
`flash-examples/CLAUDE.md`, which is auto-generated repo analysis rather than
documentation. It was determined here by testing (three `NameError`s: a module
constant, the model globals, and a module-level `import os`), then traced to
`runpod_flash/stubs/live_serverless.py`, which extracts the decorated
function's source via AST and ships it in isolation.

The same constraint is why the worker function bodies are self-contained
rather than decomposed into module-level helpers — those helpers would be
invisible to the function under `flash dev` for exactly the same reason.

### Demo

```bash
# against a local flash dev server
python demo_client.py sample_images/mel_ISIC_0024351.jpg

# against the deployed pipeline (needs RUNPOD_API_KEY in env or .env)
python demo_client.py sample_images/mel_ISIC_0024351.jpg \
  --url https://uvu4lc1mmlihc0.api.runpod.ai
```

`demo_client.py` runs the image through **both arms** and prints the verdict
from each, checked against the expected class (taken from the filename):

```
mel_ISIC_0024351.jpg  (expected: mel)

  ✓ masked  mel    0.611   -> results/mel_ISIC_0024351_masked.png
  ✓ raw     mel    0.736   -> results/mel_ISIC_0024351_raw.png
```

It also saves the image each arm actually classified, so the masked and raw
inputs can be shown side by side:

```
results/
├── <stem>_masked.png     # background blacked out, original framing kept
└── <stem>_raw.png        # SAM bypassed entirely
```

`--out-dir` changes where those land. `results/` is gitignored, since it's
generated output — `training_report.py` picks these images up automatically
and embeds them in the report's Examples section.

It also retries through cold-start 502s automatically, so the first call of a
demo won't fail while the GPU workers warm up.

### Deployed endpoints

| Endpoint | URL |
|----------|-----|
| `lesion_pipeline` (LB) | `https://uvu4lc1mmlihc0.api.runpod.ai` — `POST /analyze`, `GET /health` |
| `segment_worker` (QB) | `https://api.runpod.ai/v2/4h6wodvsn0zap7/runsync` |
| `classify_worker` (QB) | `https://api.runpod.ai/v2/8064syplx1zsn0/runsync` |

Two differences from local `flash dev` to watch for:

- **Auth**: deployed endpoints require `Authorization: Bearer $RUNPOD_API_KEY`;
  local `flash dev` does not.
- **Routes**: a deployed load-balanced endpoint serves its routes at the root
  (`/analyze`), while `flash dev` namespaces them under the endpoint name
  (`/pipeline/analyze`). `demo_client.py` handles both automatically.

**Cold-start caveat for live demos:** a fully cold `/analyze` (both GPU
workers provisioning, plus loading the models) consistently failed with
a 502 at around 40-45 seconds, which looks like a gateway timeout ahead of the
worker. Once the workers are up, calls succeed. `demo_client.py` and
`eval_sweep.py` retry through these automatically; if you're using `curl`,
send a throwaway request first to get the workers provisioned.

### After deploying, warm workers can keep serving old code

The single biggest time sink in this project. `flash deploy` cuts a new
release, but **already-running workers can keep executing the previous
version's code** — while the API reports them as current.

Concretely: after deploying a change that added a `model_variant` argument to
`classify_worker`, the endpoint kept ignoring that argument. The config showed
the new source fingerprint, `list-endpoint-workers` reported
`version: 19, isStale: false` for every worker, and the uploaded artifact was
verified to contain the new code. The worker was still running the old
function. The giveaway was behavioural, not metadata: requesting `raw` and
`masked` returned **identical confidences to six decimal places**, and the new
`model_variant` key was absent from the response.

The fix is to force the workers to cycle:

```bash
# flat keys -- a nested {"workers": {...}} body is rejected with 400
curl -X PATCH -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{"workersMin":0,"workersMax":0}' \
  https://rest.runpod.io/v1/endpoints/<id>
# wait for the worker count to reach 0, then restore
curl -X PATCH -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{"workersMin":0,"workersMax":3}' \
  https://rest.runpod.io/v1/endpoints/<id>
```

Two traps that made this much harder to diagnose than it should have been:

- **Check the PATCH status code.** The REST API takes flat `workersMin` /
  `workersMax`. A nested `{"workers": {"min": 0}}` body returns `400` with the
  scaling silently unapplied — which looks exactly like "I cycled the workers
  and it didn't help."
- **LB worker logs are unreadable.** `stream-worker-logs` returns
  `404 worker not found` for load-balancer workers that `list-endpoint-workers`
  simultaneously reports as `RUNNING`. With no logs, an unhandled exception in
  an LB route surfaces only as a bare `Internal Server Error`.

That second point is why `pipeline.py` wraps its whole handler in `try/except`
and returns the traceback in the response body. It isn't defensive style for
its own sake — it's the only reliable way to see what failed in this tier.

**Verify a deploy behaviourally, not by metadata.** Send a request whose
response provably distinguishes new code from old — a new field, or an
argument the old code ignores.

### Test the Pipeline

```bash
# Full pipeline, raw arm (the default): SAM skipped, raw-trained model
curl -X POST http://localhost:8888/pipeline/analyze \
  -H "Content-Type: application/json" \
  -d '{"input_data": {"image_base64": "<base64-encoded-image>"}}'

# Full pipeline, masked arm: SAM masks, masked-trained model
curl -X POST http://localhost:8888/pipeline/analyze \
  -H "Content-Type: application/json" \
  -d '{"input_data": {"image_base64": "<...>", "arm": "masked"}}'

# Individual stages
curl -X POST http://localhost:8888/segment_worker/runsync \
  -H "Content-Type: application/json" \
  -d '{"input": {"input_data": {"image_base64": "<base64-encoded-image>"}}}'

# model_variant selects which trained checkpoint to load ("raw" | "masked")
curl -X POST http://localhost:8888/classify_worker/runsync \
  -H "Content-Type: application/json" \
  -d '{"input": {"input_data": {"image_base64": "<...>", "model_variant": "masked"}}}'
```

`segment_worker` returns `segmented_image_base64` — non-lesion pixels blacked
out at the **original framing, not cropped to the bbox** — plus
`segmentation_applied`, `bbox`, `score`, `coverage` and `solidity`. Feed that
image straight into `classify_worker`.

### GPU Configuration

Both GPU workers use a small-model-sized fallback list, since SAM-base and
ViT-B/32 comfortably fit on 16-24GB of VRAM:

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

The first call to a given worker takes longest (a real GPU is being spun up
and the models loaded). Later calls skip any download but
still reload the model into memory each time — see the caching note above for
why that's deliberate.

Press **Ctrl+C** to stop the server. Flash's own docs say this cleans up the
endpoints provisioned during the session — but **if the process is killed
rather than interrupted, cleanup does not run and the endpoints are left
behind, still billable.** That happened repeatedly while building this. Check
with `flash undeploy list` and remove strays with
`flash undeploy <name> --force`.

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
├── classify_worker.py  # GPU worker: ViT HAM10000 classification
├── pipeline.py         # LB orchestrator: segment -> classify
├── train_worker.py     # GPU worker: fine-tunes the masked and raw arms
├── demo_client.py      # Local script: one image through both arms
├── eval_sweep.py       # Local script: score a directory across both arms
├── pipeline_client.py  # Local module: shared request/retry helpers
├── train_client.py     # Local script: submit training jobs and poll
├── training_report.py  # Local script: build the training report
├── fetch_eval_images.py # Local script: download the eval set
├── sample_images/      # 7 HAM10000 images, one per class (demo fixtures)
├── eval_images/        # 35 held-out images, 5 per class (gitignored)
├── results/            # Generated reports and images (gitignored)
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

This project doesn't use client mode — `pipeline.py` chains `segment_worker`/
`classify_worker` via direct import instead (see the caching note above).
Confirmed client mode can't invoke a **class-based** `@Endpoint` (it requires
speaking an undocumented internal protocol). It remains untested against the
**function-based** workers this project actually uses — direct import works,
so there was no reason to switch.

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
flash deploy                    # deploy to the default environment
flash deploy --env staging      # deploy to a named environment
```

### Read the whole output

A successful deploy builds and uploads the artifact every time, as documented:

```
✓ installed 3 packages  60.9s
✓ built runpod_trial  54 files, 3 deps, 85.2 MB
✓ uploaded  85.2 MB  5.0s
✓ deployed to production  3.6s
```

Don't pipe this through `tail -n` with a small `n`. The endpoint table and
example `curl` that follow run to ~14 more lines, so `tail -12` clips the
`built`/`uploaded` lines and makes a perfectly normal deploy look like it
skipped the build. (That misreading cost real debugging time here.) Use
`tee` if you want to keep the log:

```bash
flash deploy 2>&1 | tee /tmp/deploy.log
```

### Rollouts are gradual — use environments to iterate

A deploy creates a **new endpoint version**. Existing workers keep serving the
previous version and are marked `isStale: true`; they're replaced as they
cycle out. That's intentional zero-downtime rollout, not a bug — but it means
**a fresh deploy is not immediately live**, which is surprising the first time
you hit it (`✓ deployed` prints, and the old code answers your next request).

Don't fight it by forcing workers down and back up. Runpod's recommended
pattern is to **deploy to a separate environment** and test there:

```bash
flash env list                  # show environments
flash env create staging        # one-time
flash deploy --env staging      # deploy without touching production
flash env get staging           # URLs for the staging endpoints
```

Then promote to production once you're happy, and let the rollout proceed
normally.

### Verify what's actually live

```bash
flash app list                  # apps and their environments
flash env get production        # endpoint IDs and URLs
```

Endpoint versions and per-worker staleness are visible in the Runpod console
(Serverless → the endpoint → Workers), which is the quickest way to see
whether a rollout has finished.
