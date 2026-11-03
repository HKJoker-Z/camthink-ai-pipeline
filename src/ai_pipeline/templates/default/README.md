# {{project_name}}

Task type: `{{task_type}}`. Created {{created_at}}.

- `project.yaml` — the only file to edit. Classes, candidates, thresholds, backend/target names.
- `data/raw/` — source images (gitignored). Import with `ai-pipeline data import {{project_name}} <path>`.

Generated state (datasets, runs, models, deployments) lives under the artifact root, not here.
Check progress with `ai-pipeline status {{project_name}}`.
