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
VARIANTS = ("masked", "raw", "gtmasked")

# How each arm reads in prose. gtmasked is the control for mask quality:
# expert masks instead of SAM's, so it is the best case masking can achieve.
VARIANT_LABEL = {
    "masked": "SAM-masked",
    "raw": "raw",
    "gtmasked": "ground-truth-masked",
}


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


def dataset_section(prepare, prepare_gt=None):
    if not prepare:
        return ["## Dataset", "", "_No prepare output found._", ""]

    counts = prepare.get("per_class", {})
    total = prepare.get("images", 0)
    masked = prepare.get("masked", 0)
    passed = prepare.get("passed_through", 0)

    lines = [
        "## Dataset",
        "",
        f"- **{total}** images from HAM10000, each written three ways: "
        "SAM-masked, untouched, and ground-truth-masked, so every training "
        "arm sees identical source images.",
        f"- SAM found a coherent mask on **{masked}**; **{passed}** "
        f"({passed / total * 100:.0f}%) had none and were passed through "
        "unchanged — identical to the raw copy, which dilutes the contrast.",
    ]

    if prepare_gt:
        written = prepare_gt.get("written", 0) + prepare_gt.get("already_present", 0)
        missing = prepare_gt.get("no_ground_truth_mask", 0)
        coverage = prepare_gt.get("mean_coverage")
        lines.append(
            f"- Expert masks covered **{written}** images with **{missing}** "
            "missing — no fallbacks at all, against SAM's "
            f"{passed / total * 100:.0f}%. Mean lesion coverage "
            + (f"**{coverage * 100:.1f}%**." if coverage else "unavailable.")
        )

    lines += [
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


def segmentation_section(seg):
    """How good the pipeline's masks actually are, against the published ones."""
    if not seg or seg.get("mean_iou") is None:
        return []

    reported = seg.get("himel_reported_iou")
    rows = [
        ["mean IoU (fallbacks counted as 0)", f"{seg['mean_iou']:.3f}"],
        [
            "mean IoU (fallbacks excluded)",
            f"{seg['mean_iou_excluding_fallbacks']:.3f}"
            if seg.get("mean_iou_excluding_fallbacks") is not None
            else "—",
        ],
        ["mean Dice", f"{seg['mean_dice']:.3f}"],
        [
            "fallback rate",
            f"{seg['fallback_rate'] * 100:.1f}% ({seg['fell_back']}/{seg['n']})"
            if seg.get("fallback_rate") is not None
            else "—",
        ],
    ]
    if reported:
        rows.append([f"Himel et al. reported IoU", f"{reported:.3f}"])

    lines = [
        "## Segmentation quality",
        "",
        "This pipeline prompts SAM zero-shot with a centre point. Himel et al. "
        "report IoU 96.01% from a segmenter *trained* on these same masks, so "
        "this is the gap between prompting and training. Scored on the same "
        f"{seg['n']}-image test split as every accuracy number above.",
        "",
        *table(["metric", "value"], rows),
        "",
        "Both IoU figures are given on purpose. Counting fallbacks as 0 is what "
        "the pipeline actually delivers, since a fallback passes the whole "
        "unmasked image through. Excluding them says how well SAM does when it "
        "commits to a lesion at all. Quoting only the second would flatter it.",
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
                    f"trained on **{VARIANT_LABEL.get(variant, variant)}**",
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
            [f"trained on **{VARIANT_LABEL.get(model_variant, model_variant)}**"]
            + [pct(cell(model_variant, data)) for data in VARIANTS]
        )

    lines = [
        "## Model × input grid",
        "",
        "Balanced accuracy for every trained model against every input variant."
        " The diagonal is each model on its own preprocessing; the off-diagonal"
        " is the cost of a mismatch.",
        "",
        *table(
            [""] + [f"tested on {VARIANT_LABEL.get(v, v)}" for v in VARIANTS], rows
        ),
        "",
    ]
    return lines, {
        "raw_on_raw": cell("raw", "raw"),
        "raw_on_masked": cell("raw", "masked"),
        "masked_on_masked": cell("masked", "masked"),
        "masked_on_raw": cell("masked", "raw"),
        "raw_on_gtmasked": cell("raw", "gtmasked"),
        "gtmasked_on_gtmasked": cell("gtmasked", "gtmasked"),
    }


def decomposition_section(grid):
    """Split the observed damage into mismatch vs unrecoverable loss.

    Done once per masking source. Comparing the two is the point: the SAM and
    ground-truth arms differ only in mask quality and consistency, so the gap
    between their residuals is what bad masks actually cost.
    """
    if not grid or grid.get("raw_on_raw") is None:
        return []

    baseline = grid["raw_on_raw"]
    lines, summaries = [], []

    for key, label in (("masked", "SAM masks"), ("gtmasked", "ground-truth masks")):
        mismatched = grid.get(f"raw_on_{key}")
        retrained = grid.get(f"{key}_on_{key}")
        if mismatched is None or retrained is None:
            continue

        mismatch_cost = (baseline - mismatched) * 100
        recovered = (retrained - mismatched) * 100
        residual = (baseline - retrained) * 100
        summaries.append((label, recovered, mismatch_cost, residual))

        lines += [
            f"### {label}",
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
            # dominates has already flipped between runs, and a hardcoded
            # reading would have silently become false.
            (
                f"Retraining recovers {recovered:.1f} of the "
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

    if not lines:
        return []

    header = ["## Where the accuracy goes", ""]

    # The comparison between the two residuals is the headline: it separates
    # "masking is harmful" from "our segmentation was bad".
    if len(summaries) == 2:
        (_, _, _, sam_residual), (_, _, _, gt_residual) = summaries
        gap = sam_residual - gt_residual
        header += [
            f"Masking costs **{gt_residual:.1f} points** of balanced accuracy "
            f"even with expert masks on every image. SAM masks cost "
            f"**{sam_residual:.1f}**, so roughly **{gap:.1f} points** of the "
            "original result was poor segmentation rather than masking "
            "itself — and the remainder is the cost of masking done as well "
            "as it can be done here.",
            "",
        ]

    return header + lines


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
    prepare_gt = load("prepare_gt")
    seg_iou = load("segmentation_iou")
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
        "Updated by Lynn Trickey 2026-09-15",
        "",
        "Does SAM masking help or hurt classification, and if it hurts, is it "
        "because the classifier never saw masked input, or because the masks "
        "themselves were bad? Three models are fine-tuned on identical "
        "images — SAM-masked, untouched, and masked with expert ground-truth "
        "boundaries — and each is scored against every input type.",
        "",
        *dataset_section(
            (prepare or {}).get("prepare"),
            (prepare_gt or {}).get("prepare_gt"),
        ),
        *segmentation_section((seg_iou or {}).get("segmentation_iou")),
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
        "HAM10000 numbers, which are typically much higher due to 2 class "
        "problem and likely data leakage.",
        "- Himel et al. used 100 epochs and a larger dataset (6,000 training "
        "images), enlarged by augmenting the malignant class with rotated, "
        "flipped and zoomed-in copies. When we tried 30 epochs on an earlier, "
        "larger subset of our data (~2,850 training images), our model "
        "overfit: it scored worse than after 5 epochs. The runs reported here "
        "use 10 epochs on 1,760 training images.",
        "",
    ]

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "training_report.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
