"""``ai-pipeline evaluate ...`` commands."""

from __future__ import annotations

import typer

from ai_pipeline.cli.context import CliContext, run_command
from ai_pipeline.cli.output import Envelope
from ai_pipeline.cli.run_cmd import _TAIL, _TIMEOUT, _WAIT, run_envelope, status_impl
from ai_pipeline.core.project import Project
from ai_pipeline.core.runs import start_evaluation, wait_run

app = typer.Typer(help="Evaluate registered model versions.", no_args_is_help=True)


@app.command("run")
def run_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    model_version: str = typer.Argument(..., help="Model version, e.g. mv-001."),
    dataset: str | None = typer.Option(
        None, "--dataset", help="Dataset version (default: the model's)."
    ),
    split: str | None = typer.Option(
        None, "--split", help="val | test (default: evaluation.split)."
    ),
    backend: str | None = typer.Option(None, "--backend", help="Compute backend."),
    wait: bool = _WAIT,
    timeout: float = _TIMEOUT,
) -> None:
    """Submit an evaluation run; on completion the model gets metrics and an acceptance verdict."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        run = start_evaluation(
            project, model_version, dataset_version=dataset, split=split, backend=backend
        )
        if wait:
            run = wait_run(project, run["run_id"], timeout_s=timeout)
        env = run_envelope(project, run, "evaluate.run")
        if run["state"] == "COMPLETED":
            model = project.store.read("models", model_version)
            env.result["model_state"] = model["state"]
            env.result["acceptance"] = (model.get("evaluation") or {}).get("acceptance")
            env.human_text = (env.human_text or "") + f"\n  model {model_version}: {model['state']}"
            for c in (env.result["acceptance"] or {}).get("checks", []):
                verdict = "PASS" if c["passed"] else "FAIL"
                env.human_text += f"\n    {verdict} {c['metric']}={c['value']} rule={c['rule']}"
        return env

    run_command(ctx, "evaluate.run", op, project=name)


@app.command("status")
def status_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    run_id: str = typer.Argument(..., help="Run id."),
    wait: bool = _WAIT,
    timeout: float = _TIMEOUT,
    tail: int | None = _TAIL,
) -> None:
    """Alias of `run status` for evaluation runs."""
    status_impl(ctx, "evaluate.status", name, run_id, wait, timeout, tail)
