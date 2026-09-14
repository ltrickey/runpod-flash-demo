"""Assemble a markdown report from the training experiment's saved outputs.

`train_client.py` writes each stage's raw JSON to results/train_<stage>.json.
This reads those and produces results/training_report.md: dataset composition,
per-arm training scores, the full model x input-variant grid, and a
decomposition of where the accuracy goes.

Deliberately statistics-first. A couple of example images are embedded if
`demo_client.py` has left any in results/, but the point of this report is the
numbers -- the per-image reports already cover the visual side.

usage:
    python training_report.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path("results")
VARIANTS = ("masked", "raw")


def load(stage):
    path = RESULTS / f"train_{stage}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def pct(value):
    return f"{value * 100:.1f}%" if isinstance(value, (int, float)) else "—"


def table(headers, rows):
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
        *["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows],
    ]


def dataset_section(prepare):
    if not prepare:
        return ["## Dataset", "", "_No prepare output found._", ""]

    counts = prepare.get("per_class", {})
    total = prepare.get("images", 0)
    masked = prepare.get("masked", 0)
    passed = prepare.get("passed_through", 0)

    lines = [
        "## Dataset",
        "",
        f"- **{total}** images from HAM10000, each written twice: SAM-masked "
        "and untouched, so both training arms see identical source images.",
        f"- SAM found a coherent mask on **{masked}**; **{passed}** "
        f"({passed / total * 100:.0f}%) had none and were passed through "
        "unchanged — identical in both arms, which dilutes the contrast.",
        "",
        *table(
            ["class", "images"],
            [[name, counts[name]] for name in sorted(counts)],
        ),
        "",
        "HAM10000 is imbalanced by nature (`df` and `vasc` are genuinely rare),"
        " which is why balanced accuracy is reported alongside raw accuracy.",
        "",
    ]
    return lines


def training_section(train):
    if not train:
        return ["## Training", "", "_No train output found._", ""]

    any_arm = next(iter(train.values()))
    folds = any_arm.get("fold_sizes", {})

    rows = []
    for variant in VARIANTS:
        arm = train.get(variant)
        if arm:
            rows.append(
                [
                    f"trained on **{variant}**",
                    pct(arm.get("test_accuracy")),
                    pct(arm.get("test_balanced_accuracy")),
                ]
            )

    return [
        "## Training",
        "",
        f"- base model: `{any_arm.get('base_model')}`",
        f"- epochs: {any_arm.get('epochs')}",
        f"- split (grouped by lesion, so no lesion spans folds): "
        f"train {folds.get('train')} / val {folds.get('validation')} / "
        f"test {folds.get('test')}",
        "",
        "Each arm is scored on the preprocessing it was trained for:",
        "",
        *table(["arm", "accuracy", "balanced accuracy"], rows),
        "",
    ]


def grid_section(evaluate):
    if not evaluate:
        return ["## Model × input grid", "", "_No evaluate output found._", ""], None

    def cell(model_variant, data_variant):
        entry = evaluate.get(f"{model_variant}_model_on_{data_variant}_images", {})
        return entry.get("balanced_accuracy")

    rows = []
    for model_variant in VARIANTS:
        rows.append(
            [f"trained on **{model_variant}**"]
            + [pct(cell(model_variant, data)) for data in VARIANTS]
        )

    lines = [
        "## Model × input grid",
        "",
        "Balanced accuracy for every trained model against every input variant."
        " The diagonal is each model on its own preprocessing; the off-diagonal"
        " is the cost of a mismatch.",
        "",
        *table(["", "tested on masked", "tested on raw"], rows),
        "",
    ]
    return lines, {
        "raw_on_raw": cell("raw", "raw"),
        "raw_on_masked": cell("raw", "masked"),
        "masked_on_masked": cell("masked", "masked"),
        "masked_on_raw": cell("masked", "raw"),
    }


def decomposition_section(grid):
    """Split the observed damage into mismatch vs unrecoverable loss."""
    if not grid or any(value is None for value in grid.values()):
        return []

    baseline = grid["raw_on_raw"]
    mismatched = grid["raw_on_masked"]
    retrained = grid["masked_on_masked"]

    mismatch_cost = (baseline - mismatched) * 100
    recovered = (retrained - mismatched) * 100
    residual = (baseline - retrained) * 100

    return [
        "## Where the accuracy goes",
        "",
        *table(
            ["effect", "points", "meaning"],
            [
                [
                    "mismatch cost",
                    f"−{mismatch_cost:.1f}",
                    "raw-trained model fed masked input",
                ],
                [
                    "recovered by retraining",
                    f"+{recovered:.1f}",
                    "training on masked removes the mismatch",
                ],
                [
                    "residual loss",
                    f"−{residual:.1f}",
                    "never recovered — information the mask removed",
                ],
            ],
        ),
        "",
        # Phrased from the numbers rather than asserted: which effect
        # dominates has already flipped once between runs, and a hardcoded
        # reading would have silently become false.
        (
            f"Retraining on masked images recovers {recovered:.1f} of the "
            f"{mismatch_cost:.1f} points lost, leaving {residual:.1f} "
            f"unrecovered — "
            + (
                "so most of the damage is a distribution mismatch that "
                "training can fix, though a real remainder is lost signal."
                if recovered > residual
                else "so most of the damage is not a mismatch at all. "
                "Training on masked images barely helps, which means the "
                "masking is destroying information the classifier needs "
                "rather than merely presenting it unfamiliarly."
            )
        ),
        "",
    ]


def examples_section(limit=2):
    """Embed a couple of masked/raw pairs, if demo_client has produced any.

    Runs where SAM found no coherent mask are excluded: demo_client writes
    those as `*_masked-fallback.png`, because the pipeline passed the original
    image through unchanged. Charting one as "masked vs raw" would imply a
    difference that isn't there -- the two images are the same picture.
    """
    pairs = []
    skipped = len(list(RESULTS.glob("*_masked-fallback.png")))
    for masked in sorted(RESULTS.glob("*_masked.png")):
        stem = masked.name.replace("_masked.png", "")
        raw = RESULTS / f"{stem}_raw.png"
        if not raw.exists():
            continue
        pairs.append((stem, masked.name, raw.name))
        if len(pairs) >= limit:
            break

    if not pairs:
        return []

    lines = ["## Examples", "", "What the two inputs actually look like:", ""]
    for stem, masked_name, raw_name in pairs:
        lines += [
            f"**{stem}**",
            "",
            *table(["masked", "raw"], [[f"![masked]({masked_name})", f"![raw]({raw_name})"]]),
            "",
        ]
    if skipped:
        lines += [
            f"_{skipped} further pair(s) omitted: SAM found no coherent mask, "
            "so the masked and raw inputs are identical._",
            "",
        ]
    return lines


def main():
    prepare, train, evaluate = load("prepare"), load("train"), load("evaluate")
    if not any([prepare, train, evaluate]):
        raise SystemExit(
            "no results found — run train_client.py first "
            "(it writes results/train_<stage>.json)"
        )

    grid_lines, grid = grid_section((evaluate or {}).get("evaluate"))

    lines = [
        "# Masked vs raw: training experiment",
        "",
        f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}_",
        "",
        "Does SAM masking help or hurt classification, and if it hurts, is it "
        "because the classifier never saw masked input? Two models are "
        "fine-tuned on identical images — one masked, one not — and scored "
        "against both input types.",
        "",
        *dataset_section((prepare or {}).get("prepare")),
        *training_section((train or {}).get("train")),
        *grid_lines,
        *decomposition_section(grid),
        *examples_section(),
        "## Caveats",
        "",
        "- Single run, single seed, no error bars.",
        "- Masking blacks out the background at the original framing; it does "
        "not crop, so framing and scale are held constant between arms.",
        "- Lesion pixels keep their colour. Himel et al.'s wording (\"converted "
        "to binary masking\", white = lesion / black = everything else) more "
        "likely means their ViT saw the bare silhouette. Keeping the pixels is "
        "the more generous reading, so these numbers are an upper bound on how "
        "well their stated preprocessing could do.",
        "- Images where SAM found no coherent mask are identical in both arms, "
        "which understates the true contrast.",
        "- 7-class, lesion-grouped split. Not comparable to published binary "
        "HAM10000 numbers, which are typically much higher.",
        "",
    ]

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "training_report.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
