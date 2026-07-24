import typer

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


def print_comparison(summary1, summary2):

    typer.echo("\n--- Comparison ---")

    typer.echo(f"{'Metric':<25}{'Run 1':<20}{'Run 2':<20}")

    typer.echo("-" * 65)

    for key in [
        "model",
        "tokens",
        "latency_seconds",
        "ttft_seconds",
        "tokens_per_second",
    ]:
        typer.echo(f"{key:<25}{summary1[key]!s:<20}{summary2[key]!s:<20}")


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
