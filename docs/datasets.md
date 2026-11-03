# Data, labels and dataset versions

This document describes the M1 data model: how images are stored, how labels are represented,
how dataset versions are built, and exactly what makes two builds "the same dataset".

## Storage layout

```
projects/<p>/data/raw/                       operator-owned source images (gitignored)
artifacts/<p>/
  assets/<xx>/<sha256>.<ext>                 imported image bytes, read-only, written once
  assets/<xx>/<sha256>.json                  asset sidecar: dimensions, original names, imports
  imports/<import_id>.json                   import event: sources, counts, assets, rejects
  labels/<label_set_id>.json                 label set record: source, scope, counts, issues
  labels/<label_set_id>/annotations.jsonl    canonical annotations, one per line
  datasets/<ds_version>.json                 dataset version record (same content as below)
  datasets/<ds_version>/                     immutable materialised version
    dataset.json  manifest.jsonl  data.yaml
    images/{train,val,test}/<sha256>.<ext>   hardlinks to assets (copies across filesystems)
    labels/{train,val,test}/<sha256>.txt     labels rendered in the project's label_format
```

## Assets (raw data) are immutable

`ai-pipeline data import <project> [<source>]` walks the source (or every
`type: local_images` entry in `project.yaml`), hashes each image and stores it under its
SHA-256. The asset id **is** the content hash, so:

- importing the same bytes again — from any path, under any filename — does not create a
  second copy; only the sidecar's `original_filenames` / `import_ids` lists grow;
- stored files are `chmod 0444`; no pipeline operation ever rewrites them;
- derived data (dataset versions) hardlinks to them, so disk cost is paid once.

Each import is recorded as an event even when nothing new was stored (`images_existing`),
so the history of *when* data arrived is preserved. Non-image files are skipped; unreadable
images are listed under `rejected` with a reason and do not abort the import.

Supported extensions: `.jpg .jpeg .png .bmp .webp`.

## Canonical annotation (object detection)

YOLO txt is an **import/export format**, not the internal representation. Labels are stored
as JSON lines of:

```json
{
  "asset_id":   "<sha256 of the image>",
  "class_name": "person",
  "bbox":       {"x_min": 0.30, "y_min": 0.40, "x_max": 0.70, "y_max": 0.60},
  "confidence": 0.87,
  "source":     {"kind": "imported", "format": "yolo", "ref": "/path/img_001.txt",
                 "label_set_id": "lbl-…"},
  "metadata":   {"line": 1}
}
```

- Coordinates are normalised to `[0, 1]` (`x_min, y_min, x_max, y_max`). Pixel formats convert
  via the asset's recorded width/height.
- Classes are referenced by **name**; indices are assigned at build time from the project's
  class order, so re-ordering classes never requires rewriting labels.
- `source.kind` is one of `human`, `imported`, `model`. Auto-labeling (`label auto`) writes
  `kind: model` with `ref` = weights identifier plus `run_id` and `provider`.
- **One label set per dataset build.** Label sets are never merged automatically. A
  model-generated set can later be reviewed into a new corrected set that records
  `parent_label_set_id` (lineage), and that set is what gets built.
- Values are stored as parsed: out-of-range or empty boxes are kept so validation can report
  them instead of silently clamping.

Conversions live in the task module (`ai_pipeline.tasks.object_detection.yolo`):
`parse_yolo_tree` (YOLO → canonical) and `format_yolo` (canonical → YOLO).

### Label import

`ai-pipeline label import <project> <dir> [--format yolo] [--classes-file names.txt]
[--import <id>]…`

- Label files are matched to assets by **filename stem** (`img_001.txt` ↔ `img_001.jpg`). If a
  stem maps to more than one asset in scope the import fails with `AMBIGUOUS_STEMS`; restrict the
  scope with `--import`.
- Class indices in the files are resolved with `--classes-file` (one name per line, index
  order) or, by default, the project's `dataset.classes` order.
- Lines that cannot be parsed (wrong field count, non-numeric, class index out of range) are
  recorded as `parse_issues` on the label set and surface as `LABEL_PARSE_ERROR` in validation.
- Label files with no matching asset are recorded as `unmatched_label_files`.
- Assets in scope with no label file are negatives (background) — allowed, bounded by
  `max_empty_label_ratio`.

## Validation (task-owned)

`ai-pipeline dataset validate <project>` runs `ObjectDetectionTask.validate` on the current
assets, the chosen label set (default: latest READY) and the computed split. It never writes.

| Code | Severity | Meaning |
|---|---|---|
| `LABEL_PARSE_ERROR` | ERROR | a label line could not be converted |
| `UNKNOWN_CLASS` | ERROR | `class_name` not in `dataset.classes` |
| `BBOX_OUT_OF_BOUNDS` | ERROR | coordinate outside `[0, 1]` beyond `1e-6` tolerance |
| `BBOX_EMPTY` | ERROR | zero or negative width/height |
| `BBOX_INVALID` | ERROR | non-finite coordinate |
| `CONFIDENCE_OUT_OF_RANGE` | ERROR | confidence outside `[0, 1]` |
| `ANNOTATION_UNKNOWN_ASSET` | ERROR | annotation references an asset not in the dataset |
| `ASSET_FILE_MISSING` | ERROR | stored image missing (or hash mismatch with `--verify-assets`) |
| `ASSET_BAD_DIMENSIONS` / `DUPLICATE_ASSET` | ERROR | structural problems |
| `TOO_FEW_IMAGES` | ERROR | fewer than `validation.min_images` |
| `TOO_FEW_INSTANCES` | ERROR | a class has fewer than `validation.min_instances_per_class` |
| `TOO_MANY_EMPTY_IMAGES` | ERROR | empty ratio above `validation.max_empty_label_ratio` |
| `EMPTY_SPLIT` | ERROR | a split with fraction > 0 received 0 images |
| `DUPLICATE_ANNOTATION` | WARNING | identical class+box repeated on one image |
| `BBOX_TINY` | WARNING | normalised area below `1e-5` |
| `EMPTY_IMAGES` | WARNING | some images have no annotations |
| `NO_LABEL_SET` | WARNING | no label set exists; everything is unlabeled |

Warnings never invalidate a dataset unless `dataset.validation.fail_on_warnings: true`.
The report (`valid`, counts, `issues_by_code`, `issues`, `stats`) is returned by
`dataset validate` and persisted in `dataset.json` on build.

## Deterministic split

Implemented in `ai_pipeline.core.dataset.split`:

1. rank key per asset: `sha256("<seed>:<asset_id>")`;
2. sort assets by rank key;
3. split sizes from the configured fractions by **largest remainder** (counts sum to N; ties
   broken in `train, val, test` order);
4. cut the ranked list into `train | val | test`.

No random number generator, no dependence on filenames, paths, import order or filesystem
iteration order — the same assets, seed and fractions give the same assignment on every machine.
Split membership is recorded per sample in `manifest.jsonl`. Adding or removing assets re-ranks
and may move other assets between splits; that is a new dataset version anyway.
Stratified splitting is not implemented in V1.

## Dataset identity (content hash)

`content_hash` (`hash_spec: dataset-hash-v1`) is SHA-256 over a canonical JSON document built
from:

| Included | Why |
|---|---|
| task type | different tasks are different datasets |
| class list, **in order** | indices in rendered labels depend on it |
| label format | rendered files differ |
| split fractions and seed | assignment depends on them |
| samples sorted by `asset_id`: `asset_id`, split name, annotations sorted by content key (`class_name`, bbox rounded to 6 decimals, confidence) | the actual content |

Explicitly **excluded**: absolute/relative paths, original filenames, import ids, label set
ids, timestamps, hostnames, the version number, annotation `source`/`metadata` (identical boxes
from a human and a model are the same content), and validation thresholds (they gate
acceptance, not content).

Consequences:

- `dataset build` is idempotent: if a READY version with the same hash exists it is returned
  with `"created": false`; the immutable version directory is not touched.
- Re-importing the same labels under a new label set id yields the same version. Its
  `provenance` stays that of the first build, but every build request that resolved to the
  version is appended to `build_history` on the **store record**
  (`{at, created, label_set_id, import_ids, annotations_sha256}`), so "another import/build
  produced identical content" is never lost. `dataset.json` inside the version directory keeps
  only the first entry — it is part of the immutable content snapshot.
- Changing a box, a class name, class order, the seed or fractions, or adding/removing an image
  produces a new version. Changing `min_images` does not.

## Dataset versions are immutable

Versions are `ds-v001, ds-v002, …`. A build writes the record as `BUILDING`, materialises the
directory, then flips it to `READY`, writes `dataset.json`, makes every file read-only and sets
the project pointer `current_dataset_version`. If anything fails the record is `FAILED` with the
error; failed ids are never reused and failed versions are never returned as "existing".

There is no code path that writes into an existing version directory. To change anything,
build a new version.

## Provenance recorded in `dataset.json`

| Question | Field |
|---|---|
| which source assets? | `manifest.jsonl` (`asset_id` = content hash), `provenance.import_ids` |
| which annotations? | `provenance.label_set_id`, `provenance.annotations_sha256`, per-sample `n_annotations`, rendered label files |
| where did labels come from? | `provenance.label_source` (`kind`, `format`, `path`, `class_names`) and `source` on every annotation in the label set |
| which configuration/classes? | `config_snapshot` (classes, split, validation), `provenance.config_path` |
| which split per asset? | `manifest.jsonl` `split` |
| what validation was performed? | `validation` (full report incl. warnings and stats) |
| what identity resulted? | `content_hash`, `hash_spec`, `dataset_version` |
