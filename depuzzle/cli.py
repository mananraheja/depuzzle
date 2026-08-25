import typer
from rich import box
from rich.console import Console
from rich.table import Table

from depuzzle import __version__
from depuzzle.backends.ollama import OllamaBackend
from depuzzle.exporters import save_trace
from depuzzle.loaders import load_trace
from depuzzle.metrics import Metrics
from depuzzle.profiler import Profiler

console = Console()

app = typer.Typer(
    name="depuzzle",
    help="Inference profiler for local LLMs.",
)


profile_app = typer.Typer(help="Profile LLM inference requests.")


def format_change(key, value1, value2):
    if not isinstance(value1, (int, float)):
        return "-"

    if value1 == 0:
        return "-"

    change = ((value2 - value1) / value1) * 100

    if key in [
        "latency_seconds",
        "ttft_seconds",
    ]:
        if change > 0:
            return f"[red]{change:+.2f}% (slower)[/red]"
        else:
            return f"[green]{change:+.2f}% (faster)[/green]"

    if key == "tokens_per_second":
        if change > 0:
            return f"[green]{change:+.2f}% (better)[/green]"
        else:
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

        table.add_row(
            key,
            str(value1),
            str(value2),
            change,
        )


def add_runtime_comparison(table, summary1, summary2):
    runtime1 = summary1.get("runtime")
    runtime2 = summary2.get("runtime")

    print(runtime1)
    print(runtime2)

    if not runtime1 and not runtime2:
        return

    table.add_section()

    table.add_row(
        "[bold]runtime[/bold]",
        "",
        "",
        "",
    )

    for key in [
        "backend",
        "processor",
        "context_length",
    ]:
        value1 = runtime1.get(key, "N/A") if runtime1 else "N/A"
        value2 = runtime2.get(key, "N/A") if runtime2 else "N/A"

        if value1 == value2:
            change = "same"
        elif value1 == "N/A" or value2 == "N/A":
            change = "N/A"
        else:
            change = "changed"

        table.add_row(
            f"  {key}",
            str(value1),
            str(value2),
            change,
        )


def print_comparison(summary1, summary2):
    table = Table(
        title="Comparison",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold",
    )

    table.add_column("Metric", style="white", width=20)
    table.add_column("Run 1", width=20)
    table.add_column("Run 2", width=20)
    table.add_column("Change", width=25)

    add_metric_comparison(table, summary1, summary2)
    add_runtime_comparison(table, summary1, summary2)

    console.print(table)


def print_summary(trace):
    """
    Print inference metrics.
    """

    metrics = Metrics(trace)

    summary = metrics.summary()

    typer.echo("\n--- Trace Summary ---")

    for key, value in summary.items():
        if isinstance(value, dict):
            typer.echo(f"{key}:")
            for sub_key, sub_value in value.items():
                typer.echo(f"  {sub_key}: {sub_value}")

        else:
            typer.echo(f"{key}: {value}")


@profile_app.command()
def run(
    model: str = typer.Option(
        "qwen2.5:3b",
        help="LLM model to use.",
    ),
    prompt: str = typer.Option(
        "Explain virtual memory in one paragraph.",
        help="Prompt to send to the model.",
    ),
    output: str = typer.Option(
        None,
        help="Save trace to JSON file.",
    ),
    runs: int = typer.Option(
        1,
        min=1,
        help="Number of times to run the inference.",
    ),
):
    """
    Profile a single LLM inference request.
    """

    backend = OllamaBackend(model=model)

    if runs < 1:
        raise typer.BadParameter("runs must be at least 1")

    profiler = Profiler(backend)

    traces = [profiler.run(prompt) for _ in range(runs)]

    for i, trace in enumerate(traces, start=1):
        typer.echo(f"\n--- Run {i}/{len(traces)} ---")
        print_summary(trace)

    if output:
        for i, trace in enumerate(traces, start=1):
            save_trace(trace, f"{output}_{i}")

        typer.echo(f"\nTraces saved to {output}")


@app.command()
def version():
    typer.echo(f"depuzzle {__version__}")


@app.command()
def compare(
    run1: str,
    run2: str,
):
    """
    Compare two inference traces.
    """

    typer.echo(f"Comparing {run1}")
    typer.echo(f"against {run2}")

    try:
        trace1 = load_trace(run1)
        trace2 = load_trace(run2)

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    metrics1 = Metrics(trace1)
    metrics2 = Metrics(trace2)

    summary1 = metrics1.summary()
    summary2 = metrics2.summary()

    print_comparison(
        summary1,
        summary2,
    )


app.add_typer(
    profile_app,
    name="profile",
)


if __name__ == "__main__":
    app()
