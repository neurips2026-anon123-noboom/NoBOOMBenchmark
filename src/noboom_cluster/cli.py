"""Automate Ray cluster launch and benchmark submission."""
from __future__ import annotations

import logging
import os
import sys
from typing import List, Optional

import typer

try:
    from .noboom_cli_lib.ray_lifecycle import (
        ClusterRunConfig,
        MLFLOW_UI_LOCAL_PORT,
        RAY_DASHBOARD_LOCAL_PORT,
        UpdateAllowedConfig,
        run_cluster_job,
        update_cluster_allowlist,
    )
    from noboom_benchmark.noboom_lib.core.notifications import ensure_telegram_enrollment_env
    from .noboom_cli_lib.specs import parse_dataset_model_pair
    from .noboom_cli_lib.ray_utils.internal.logging import setup_logging
except ImportError:  # pragma: no cover - compatibility for direct script execution
    from noboom_cluster.noboom_cli_lib.ray_lifecycle import (
        ClusterRunConfig,
        MLFLOW_UI_LOCAL_PORT,
        RAY_DASHBOARD_LOCAL_PORT,
        UpdateAllowedConfig,
        run_cluster_job,
        update_cluster_allowlist,
    )
    from noboom_cluster.noboom_cli_lib.specs import parse_dataset_model_pair
    from noboom_cluster.noboom_cli_lib.ray_utils.internal.logging import setup_logging
    from noboom_benchmark.noboom_lib.core.notifications import ensure_telegram_enrollment_env

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help="NoBoom cluster orchestration CLI.")


def _normalize_legacy_multi_value_options(args: List[str]) -> List[str]:
    normalized: List[str] = []
    index = 0
    multi_value_flags = {"--dataset", "--model"}

    while index < len(args):
        token = args[index]
        if token not in multi_value_flags:
            normalized.append(token)
            index += 1
            continue

        index += 1
        values: List[str] = []
        while index < len(args) and not args[index].startswith("-"):
            values.append(args[index])
            index += 1

        if not values:
            normalized.append(token)
            continue

        for value in values:
            normalized.extend([token, value])

    return normalized


def _execute_run(
    *,
    machine_ip_file: str,
    cluster_config: Optional[str],
    dataset: Optional[List[str]],
    model: Optional[List[str]],
    pair: Optional[List[str]],
    tune: bool,
    experiment_id: Optional[str],
    gpus_per_run: float,
    force_restart: bool,
    exclusive: bool,
    ssh_user: str,
    ssh_key: str,
    root_dir: str,
    ray_temp_dir: str,
    dashboard_port: int,
    mlflow_port: int,
    deployment_mode: str,
    verbose: int,
    max_in_flight: int,
    hpo_seeds: Optional[List[int]],
    notify_channel: Optional[List[str]],
    notify_email_to: Optional[List[str]],
    ram_guard: bool,
    ram_guard_max_used_fraction: float,
    ram_guard_poll_interval_s: float,
    ram_guard_resume_used_fraction: float,
    ram_guard_cooldown_s: float,
) -> int:
    setup_logging(verbose)
    if not pair and (not dataset or not model):
        raise typer.BadParameter(
            "Provide --dataset with --model, one or more --pair values, or both."
        )
    for raw_pair in pair or []:
        try:
            parse_dataset_model_pair(raw_pair)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--pair") from exc
    os.environ["NOBOOM_RAM_GUARD_ENABLED"] = "1" if ram_guard else "0"
    os.environ["NOBOOM_RAM_GUARD_MAX_USED_FRACTION"] = str(ram_guard_max_used_fraction)
    os.environ["NOBOOM_RAM_GUARD_POLL_INTERVAL_S"] = str(ram_guard_poll_interval_s)
    os.environ["NOBOOM_RAM_GUARD_RESUME_USED_FRACTION"] = str(ram_guard_resume_used_fraction)
    os.environ["NOBOOM_RAM_GUARD_COOLDOWN_S"] = str(ram_guard_cooldown_s)
    channels = {channel.strip().lower() for channel in notify_channel or []}
    unknown_channels = channels - {"email", "telegram"}
    if unknown_channels:
        raise typer.BadParameter(
            "Unsupported --notify-channel value(s): " + ", ".join(sorted(unknown_channels))
        )
    if "email" in channels:
        os.environ["NOBOOM_NOTIFY_EMAIL_ENABLED"] = "1"
        recipients = [
            item.strip()
            for raw in notify_email_to or []
            for item in str(raw).split(",")
            if item.strip()
        ]
        if recipients:
            os.environ["NOBOOM_NOTIFY_EMAIL_TO"] = ",".join(recipients)
            os.environ["NOBOOM_NOTIFY_SMTP_TO"] = ",".join(recipients)
    if "telegram" in channels or os.getenv("NOBOOM_NOTIFY_TELEGRAM_ENABLED") == "1":
        os.environ["NOBOOM_NOTIFY_TELEGRAM_ENABLED"] = "1"
        link = ensure_telegram_enrollment_env(os.environ)
        typer.echo(f"Telegram notifications enrollment link: {link}")

    config = ClusterRunConfig(
        machine_ip_file=machine_ip_file,
        cluster_config=cluster_config,
        datasets=dataset,
        models=model,
        pairs=pair,
        tune=tune,
        experiment_id=experiment_id,
        force_restart=force_restart,
        exclusive=exclusive,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        root_dir=root_dir,
        ray_temp_dir=ray_temp_dir,
        dashboard_port=dashboard_port,
        mlflow_port=mlflow_port,
        deployment_mode=deployment_mode,
        verbose=verbose,
        gpus_per_run=gpus_per_run,
        max_in_flight=max_in_flight,
        hpo_seeds=hpo_seeds,
    )
    return run_cluster_job(config)


@app.command("run")
def run_command(
    machine_ip_file: str = typer.Option(..., help="YAML inventory file with a 'nodes' list."),
    cluster_config: Optional[str] = typer.Option(None, help="Optional pre-rendered Ray cluster YAML."),
    dataset: Optional[List[str]] = typer.Option(None, help="Dataset names."),
    model: Optional[List[str]] = typer.Option(None, help="Model names."),
    pair: Optional[List[str]] = typer.Option(
        None,
        "--pair",
        help="Explicit dataset-model pair in DATASET:MODEL format. May be repeated.",
    ),
    tune: bool = typer.Option(False, help="Run tuning mode."),
    experiment_id: Optional[str] = typer.Option(
        None,
        help="Source MLflow experiment ID for single-train parameter/checkpoint lookup.",
    ),
    gpus_per_run: float = typer.Option(0.125, help="GPUs to allocate per single-train run."),
    force_restart: bool = typer.Option(False, help="Force restart of the Ray job server."),
    exclusive: bool = typer.Option(False, help="Stop all running jobs before submitting a new one."),
    ssh_user: str = typer.Option("cloud", help="SSH username for accessing the nodes."),
    ssh_key: str = typer.Option(
        os.path.expanduser("~/.ssh/id_ed25519"),
        help="SSH key with passwordless access to cluster nodes.",
    ),
    root_dir: str = typer.Option("~/noboom", help="Default root directory for running."),
    ray_temp_dir: str = typer.Option("/tmp/ray", help="Temporary directory for Ray logs."),
    dashboard_port: int = typer.Option(RAY_DASHBOARD_LOCAL_PORT, help="Local Ray dashboard port."),
    mlflow_port: int = typer.Option(MLFLOW_UI_LOCAL_PORT, help="Local MLflow UI port."),
    deployment_mode: str = typer.Option("docker", help="Deployment mode: docker or native."),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase logging verbosity."),
    max_in_flight: int = typer.Option(4, help="Maximum number of pair jobs to keep in flight."),
    hpo_seed: Optional[List[int]] = typer.Option(
        None,
        "--hpo-seed",
        help="Seed to use during HPO. Repeat to provide multiple seeds.",
    ),
    notify_channel: Optional[List[str]] = typer.Option(
        None,
        "--notify-channel",
        help="Notification channel: email or telegram. May be repeated.",
    ),
    notify_email_to: Optional[List[str]] = typer.Option(
        None,
        "--notify-email-to",
        help="Email recipient. May be repeated or comma-separated.",
    ),
    ram_guard: bool = typer.Option(True, "--ram-guard/--no-ram-guard"),
    ram_guard_max_used_fraction: float = typer.Option(0.95),
    ram_guard_poll_interval_s: float = typer.Option(10.0),
    ram_guard_resume_used_fraction: float = typer.Option(0.90),
    ram_guard_cooldown_s: float = typer.Option(120.0),
) -> int:
    exit_code = _execute_run(
        machine_ip_file=machine_ip_file,
        cluster_config=cluster_config,
        dataset=dataset,
        model=model,
        pair=pair,
        tune=tune,
        experiment_id=experiment_id,
        gpus_per_run=gpus_per_run,
        force_restart=force_restart,
        exclusive=exclusive,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        root_dir=root_dir,
        ray_temp_dir=ray_temp_dir,
        dashboard_port=dashboard_port,
        mlflow_port=mlflow_port,
        deployment_mode=deployment_mode,
        verbose=verbose,
        max_in_flight=max_in_flight,
        hpo_seeds=hpo_seed,
        notify_channel=notify_channel,
        notify_email_to=notify_email_to,
        ram_guard=ram_guard,
        ram_guard_max_used_fraction=ram_guard_max_used_fraction,
        ram_guard_poll_interval_s=ram_guard_poll_interval_s,
        ram_guard_resume_used_fraction=ram_guard_resume_used_fraction,
        ram_guard_cooldown_s=ram_guard_cooldown_s,
    )
    return exit_code


@app.command("telegram-link")
def telegram_link_command() -> int:
    link = ensure_telegram_enrollment_env(os.environ)
    typer.echo(link)
    return 0


@app.command("update-allowed")
def update_allowed_command(
    machine_ip_file: str = typer.Option(..., help="YAML inventory file with a 'nodes' list."),
    ssh_user: str = typer.Option("cloud", help="SSH username for accessing the nodes."),
    ssh_key: str = typer.Option(
        os.path.expanduser("~/.ssh/id_ed25519"),
        help="SSH key with passwordless access to cluster nodes.",
    ),
    dashboard_port: int = typer.Option(RAY_DASHBOARD_LOCAL_PORT, help="Local Ray dashboard port."),
    mlflow_port: int = typer.Option(MLFLOW_UI_LOCAL_PORT, help="Local MLflow UI port."),
    root_dir: str = typer.Option("~/noboom", help="Default root directory for running."),
    ray_temp_dir: str = typer.Option("/tmp/ray", help="Temporary directory for Ray logs."),
    deployment_mode: str = typer.Option("docker", help="Deployment mode: docker or native."),
) -> int:
    response = update_cluster_allowlist(
        UpdateAllowedConfig(
            machine_ip_file=machine_ip_file,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            dashboard_port=dashboard_port,
            mlflow_port=mlflow_port,
            root_dir=root_dir,
            ray_temp_dir=ray_temp_dir,
            deployment_mode=deployment_mode,
        )
    )
    typer.echo(response.to_json())
    return 0 if response.accepted else 1


def main(argv: Optional[List[str]] = None, prog_name: Optional[str] = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    command_name: Optional[str] = None
    if args and args[0] in {"update-allowed", "telegram-link"}:
        command_name = "update-allowed"
        if args[0] == "telegram-link":
            command_name = "telegram-link"
        args = args[1:]
    elif not args or args[0] == "run" or args[0].startswith("-"):
        command_name = "run"
        if args and args[0] == "run":
            args = args[1:]

    args = _normalize_legacy_multi_value_options(args)
    if command_name is not None:
        args = [command_name] + args

    try:
        result = app(
            args=args,
            prog_name=prog_name or "noboom-run-cluster",
            standalone_mode=False,
        )
        return int(result) if isinstance(result, int) else 0
    except typer.Exit as exc:
        return int(exc.exit_code)
    except Exception:  # noqa: BLE001
        logger.exception("Cluster job execution failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
