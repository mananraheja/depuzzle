from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from depuzzle import __version__
from depuzzle.backends.ollama import OllamaBackend
from depuzzle.exporters import save_trace
from depuzzle.loaders import load_trace
from depuzzle.metrics import Metrics
from depuzzle.models import Device, ExecutionConfig, Lifecycle, ProfileConfig
from depuzzle.profiler import Profiler

console = Console()

app = typer.Typer(
    name="depuzzle",
    help="Inference profiler for local LLMs.",
)

profile_app = typer.Typer(help="Profile LLM inference requests.")


def format_change(key, value1, value2):
    if not isinstance(value1, (int, float)) or not isinstance(value2, (int, float)):
        return "N/A"

    if value1 == 0:
        return "-"

    change = ((value2 - value1) / value1) * 100

    if key in ["latency_seconds", "ttft_seconds"]:
        if change > 0:
            return f"[red]{change:+.2f}% (slower)[/red]"
        return f"[green]{change:+.2f}% (faster)[/green]"

    if key in ["tokens_per_second", "generation_tokens_per_second",
               "prompt_tokens_per_second"]:
        if change > 0:
            return f"[green]{change:+.2f}% (better)[/green]"
        return f"[red]{change:+.2f}% (worse)[/red]"

    return f"{change:+.2f}%"


def add_metric_comparison(table, summary1, summary2):
    for key in [
        "model",
        "tokens",
        "latency_seconds",
        "ttft_seconds",
        "tokens_per_second",
    ]:
        value1 = summary1[key]
        value2 = summary2[key]
        change = format_change(key, value1, value2)
        table.add_row(key, str(value1), str(value2), change)


def add_runtime_comparison(table, summary1, summary2):
    runtime1 = summary1.get("runtime")
    runtime2 = summary2.get("runtime")

    if not runtime1 and not runtime2:
        return

    table.add_section()
    table.add_row("[bold]runtime[/bold]", "", "", "")

    for key in ["backend", "processor", "context_length"]:
        value1 = runtime1.get(key, "N/A") if runtime1 else "N/A"
        value2 = runtime2.get(key, "N/A") if runtime2 else "N/A"

        if value1 == value2:
            change = "same"
        elif value1 == "N/A" or value2 == "N/A":
            change = "N/A"
        else:
            change = "changed"

        table.add_row(f"  {key}", str(value1), str(value2), change)


def add_runtime_stats_comparison(table, summary1, summary2):
    stats1 = summary1.get("ollama")
    stats2 = summary2.get("ollama")

    if not stats1 and not stats2:
        return

    table.add_section()
    table.add_row("[bold]backend measurements[/bold]", "", "", "")

    keys = [
        "load_duration_seconds",
        "prompt_eval_duration_seconds",
        "prompt_eval_count",
        "prompt_tokens_per_second",
        "generation_duration_seconds",
        "generation_count",
        "generation_tokens_per_second",
    ]

    for key in keys:
        value1 = stats1.get(key) if stats1 else None
        value2 = stats2.get(key) if stats2 else None

        if value1 is None or value2 is None:
            change = "N/A"
        elif isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            change = format_change(key, value1, value2)
        else:
            change = "same" if value1 == value2 else "changed"

        table.add_row(
            f"  {key}",
            "N/A" if value1 is None else str(value1),
            "N/A" if value2 is None else str(value2),
            change,
        )


def print_comparison(summary1, summary2):
    table = Table(
        title="Comparison",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold",
    )

    table.add_column("Metric", style="white", width=28)
    table.add_column("Run 1", width=20)
    table.add_column("Run 2", width=20)
    table.add_column("Change", width=25)

    add_metric_comparison(table, summary1, summary2)
    add_runtime_comparison(table, summary1, summary2)
    add_runtime_stats_comparison(table, summary1, summary2)
    console.print(table)


def print_summary(trace):
    """Print inference metrics."""
    summary = Metrics(trace).summary()
    typer.echo("\n--- Trace Summary ---")

    for key, value in summary.items():
        if isinstance(value, dict):
            typer.echo(f"{key}:")
            for sub_key, sub_value in value.items():
                typer.echo(f"  {sub_key}: {sub_value}")
        else:
            typer.echo(f"{key}: {value}")


def trace_filename(
    model: str,
    device: Device,
    lifecycle: Lifecycle,
    run_number: int,
    gpu_layers: int | None = None,
) -> str:
    model_name = model.replace(":", "_").replace("/", "_")

    if device == Device.HYBRID:
        device_name = f"hybrid_{gpu_layers}"
    else:
        device_name = device.value

    return f"{model_name}_{device_name}_{lifecycle.value}_run_{run_number}.json"


@profile_app.command()
def run(
    model: str = typer.Option("qwen2.5:3b", help="LLM model to use."),
    prompt: str = typer.Option(
        "Explain virtual memory in one paragraph.",
        help="Prompt to send to the model.",
    ),
    output: str = typer.Option(None, help="Save trace to JSON file."),
    output_dir: str = typer.Option("traces", help="Directory to save traces."),
    runs: int = typer.Option(1, min=1, help="Number of times to run the inference."),
    lifecycle: Lifecycle = typer.Option(
        Lifecycle.HOT, help="Run lifecycle: cold, warmup, or hot."
    ),
    device: Device = typer.Option(
        Device.CPU, help="Execution device: cpu, gpu, or hybrid."
    ),
    gpu_layers: int = typer.Option(
        None, help="Number of GPU layers to use for hybrid execution."
    ),
):
    """Profile a single LLM inference request."""
    backend = OllamaBackend(model=model)

    if runs < 1:
        raise typer.BadParameter("runs must be at least 1")
    if output and runs > 1:
        raise typer.BadParameter("--output can only be used with --runs 1")

    config = ProfileConfig(
        lifecycle=lifecycle,
        execution=ExecutionConfig(device=device, gpu_layers=gpu_layers),
    )
    profiler = Profiler(backend, config)

    if not output:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    for run_number in range(1, runs + 1):
        trace = profiler.run(prompt)
        typer.echo(f"\n--- Run {run_number}/{runs} ---")
        print_summary(trace)

        if output:
            trace_path = Path(output)
        else:
            filename = trace_filename(
                model=model,
                device=device,
                lifecycle=lifecycle,
                run_number=run_number,
                gpu_layers=gpu_layers,
            )
            trace_path = output_path / filename

        save_trace(trace, str(trace_path))
        typer.echo(f"Saved: {trace_path}")


@app.command()
def version():
    typer.echo(f"depuzzle {__version__}")


@app.command()
def compare(run1: str, run2: str):
    """Compare two inference traces."""
    typer.echo(f"Comparing {run1}")
    typer.echo(f"against {run2}")

    try:
        trace1 = load_trace(run1)
        trace2 = load_trace(run2)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    print_comparison(Metrics(trace1).summary(), Metrics(trace2).summary())


app.add_typer(profile_app, name="profile")

if __name__ == "__main__":
    app()
