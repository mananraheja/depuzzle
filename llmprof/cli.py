import typer

from llmprof import __version__
from llmprof.backends.ollama import OllamaBackend
from llmprof.exporters import save_trace
from llmprof.loaders import load_trace
from llmprof.metrics import Metrics
from llmprof.profiler import Profiler

app = typer.Typer(
    name="llmprof",
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
            return f"{change:+.2f}% (slower)"
        else:
            return f"{change:+.2f}% (faster)"

    if key == "tokens_per_second":
        if change > 0:
            return f"{change:+.2f}% (better)"
        else:
            return f"{change:+.2f}% (worse)"

    return f"{change:+.2f}%"


def print_comparison(summary1, summary2):

    typer.echo("\n--- Comparison ---")

    typer.echo(f"{'Metric':<25}{'Run 1':<20}{'Run 2':<20}{'Change':<25}")
    typer.echo("-" * 90)

    for key in [
        "model",
        "tokens",
        "latency_seconds",
        "ttft_seconds",
        "tokens_per_second",
    ]:
        change = format_change(
            key,
            summary1[key],
            summary2[key],
        )

    typer.echo(
        f"{key:<25}" f"{summary1[key]!s:<20}" f"{summary2[key]!s:<20}" f"{change:<25}"
    )


def print_summary(trace):
    """
    Print inference metrics.
    """

    metrics = Metrics(trace)

    summary = metrics.summary()

    typer.echo("\n--- Trace Summary ---")

    for key, value in summary.items():
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
):
    """
    Profile a single LLM inference request.
    """

    backend = OllamaBackend(model=model)

    profiler = Profiler(backend)

    trace = profiler.run(prompt)

    print_summary(trace)

    if output:
        save_trace(trace, output)

        typer.echo(f"\nTrace saved to {output}")


@app.command()
def version():
    typer.echo(f"llmprof {__version__}")


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
