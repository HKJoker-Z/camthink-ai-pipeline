# CLI

`ai-pipeline` is the only operator interface. Every command is non-interactive, takes the
project name explicitly, and can emit a machine-readable envelope with `--output json`.

## Global options

| Option | Purpose |
|---|---|
| `-o, --output human\|json` | output format (default `human`) |
| `--infra <file>` | `infrastructure.yaml` to use (default: search upward from CWD; env `AIP_INFRA_FILE`) |
| `--projects-root <dir>` | override `projects_root` (env `AIP_PROJECTS_ROOT`) |
| `--artifact-root <dir>` | override `artifact_root` (env `AIP_ARTIFACT_ROOT`) |
| `-v` / `-vv` | info / debug logging on stderr |
| `-q` | errors only |
| `--log-json` | JSON-lines logs on stderr |
| `--version` | print version |

Logs always go to **stderr**; stdout carries only the command's output.

## Commands implemented

### Project (M0)

| Command | Effect | Typical exit codes |
|---|---|---|
| `project create <name> [--task object_detection] [--template default]` | scaffold `projects/<name>/project.yaml`, `README.md`, `data/raw/`; initialise `artifacts/<name>/project.json` | 0, 2 (bad name/task), 3 (exists) |
| `project validate <name>` | load and fully validate the project config, including infrastructure references | 0, 2, 3 (not found) |
| `project list` | list projects under the projects root with validity | 0 |
| `status <name>` | pointers, entity counts (incl. assets), latest/active runs, `next_operations` | 0, 3 |

### Data, labels, datasets (M1) — see [datasets.md](datasets.md)

| Command | Effect | Typical exit codes |
|---|---|---|
| `data import <name> [<source>] [--no-recursive]` | import images into the content-addressed asset store; default source = `dataset.sources` of type `local_images` | 0 (also when nothing new), 1 (`NO_IMAGES_FOUND`, `IMPORT_SOURCE_MISSING`), 2 (`NO_LOCAL_IMAGE_SOURCE`) |
| `data list <name>` | import events and total asset count | 0 |
| `label import <name> <path> [--format yolo] [--classes-file f] [--import id]…` | convert label files to a canonical label set (`kind: imported`) | 0, 1 (`NO_ASSETS`, `AMBIGUOUS_STEMS`, `LABEL_SOURCE_MISSING`) |
| `label list <name>` | label sets with counts and parse-issue totals | 0 |
| `dataset validate <name> [--labels id] [--import id]… [--verify-assets]` | task-owned validation report; writes nothing | 0 valid, 1 `DATASET_INVALID` (report still in `result`) |
| `dataset build <name> [--labels id] [--import id]…` | validate, then build an immutable version; idempotent on content (`created: false`) | 0, 1 (`DATASET_INVALID`, `DATASET_BUILD_FAILED`) |
| `dataset list <name>` | versions with state, counts, short hash; `*` marks the current pointer | 0 |
| `dataset show <name> <version> [--manifest]` | full `dataset.json` (provenance, config snapshot, validation) | 0, 3 (`DATASET_NOT_FOUND`) |

Planned command groups (`train`, `evaluate`, `model`, `export`, `target`, `benchmark`,
`deploy`, `rollback`, `health`) are specified in
[design/v1-architecture-proposal.md §13](design/v1-architecture-proposal.md) and are added
milestone by milestone.

## JSON envelope

```json
{
  "schema_version": 1,
  "project": "demo_person_detection",
  "operation": "project.validate",
  "status": "ok",
  "result": { "...": "command-specific" },
  "error": null,
  "next_operations": ["data import demo_person_detection <path>"],
  "timestamp": "2026-11-03T07:33:00Z"
}
```

- `status` is `ok`, `failed`, or `rejected` (policy rejection; not a crash).
- On failure `result` is `{}` and `error` is
  `{ "code", "message", "details", "hint" }`. `code` is stable and machine-matchable
  (e.g. `PROJECT_NOT_FOUND`, `CONFIG_INVALID`, `CONFIG_INFRA_MISMATCH`, `PROJECT_EXISTS`,
  `ENV_VAR_MISSING`, `INFRA_FILE_MISSING`, `ILLEGAL_TRANSITION`, `DATASET_INVALID`,
  `AMBIGUOUS_STEMS`, `INTERNAL_ERROR`).
- `dataset validate` is the one command that returns `status: failed` *with* a populated
  `result` (the full report), so an agent can read the issues without a second call.
- `next_operations` are suggestions, expressed as `ai-pipeline` argument strings.
- Exactly one JSON document is written to stdout, also on failure.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | operation failed (including unexpected internal errors) |
| 2 | usage or configuration error |
| 3 | precondition / state error (not found, already exists, illegal transition) |
| 4 | remote / connectivity error |
| 5 | policy rejection (metrics below configured acceptance) |

Codes 4 and 5 are defined now and used from M2/M3 onward.
