# Evaluation protocol

![Protocol flow](../assets/v2-evidence-flow.svg)

## Objective

This repository is a reproducible **local-data** baseline for PPE object detection. It has no bundled training images, labels, checkpoints, or validated hardware/deployment integration.

The single class contract is fixed in [`config.yaml`](../config.yaml):

| ID | Class |
|---:|---|
| 0 | `person` |
| 1 | `hard_hat` |
| 2 | `no_hard_hat` |

Do not silently remap labels from a different taxonomy. Update the configuration and document the migration before using a different dataset.

## Split discipline

- `train` is used to fit model parameters.
- `validation` (`val` on disk) is used for monitoring and checkpoint selection.
- `test` is held out. It is consumed only by `python -m src.evaluate` after the selected checkpoint is fixed.

`validate_dataset()` rejects sample IDs that occur in more than one split. This check is a necessary guard, not proof that visually near-duplicate images are absent; inspect collection provenance and perform duplicate analysis where appropriate.

## Validation gates

Before training or evaluation, the validation CLI checks:

1. expected `images/<split>` directories, extensions, and readable image files;
2. missing/orphan labels;
3. YOLO row width, numeric values, class IDs, and normalized coordinate bounds;
4. negative images (allowed and counted);
5. overlap of normalized sample IDs across splits.

It records a `split_manifest.json` containing split inventory, per-file SHA-256 values, label counts, and a dataset fingerprint. Dataset files and run outputs stay local by design. Before test evaluation, the current fingerprint must equal the fingerprint recorded at training; this rejects renamed, replaced, or label-edited inputs even when split counts happen to match.

## Artifact chain

A credible experiment should preserve these linked, generated artifacts:

1. resolved configuration and its digest (`checkpoints/resolved_config.json`);
2. content-addressed split manifest (`checkpoints/split_manifest.json`);
3. run metadata and validation history (`checkpoints/run_metadata.json`);
4. selected weights plus SHA-256 (`checkpoints/checkpoint.sha256`);
5. held-out test output linked to the same config, data, and checkpoint digests (`reports/test_metrics.json`);
6. optional dashboard generated from those JSON files (`reports/experiment_dashboard.png` or `.svg`).

See [`experiment-card-template.md`](experiment-card-template.md) for a concise evidence record. Do not publish a metric, plot, or deployment assertion without the matching artifact chain.
