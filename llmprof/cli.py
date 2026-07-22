import typer

from llmprof.backends.ollama import OllamaBackend
from llmprof.profiler import Profiler
from llmprof.exporters import save_trace
from llmprof.metrics import Metrics


app = typer.Typer(
    name="llmprof",
    help="Inference profiler for local LLMs.",
)


profile_app = typer.Typer(
    help="Profile LLM inference requests."
)


def print_summary(trace):
    """
    Print inference metrics.
    """

    metrics = Metrics(trace)

    summary = metrics.summary()

    typer.echo("\n--- Trace Summary ---")

    for key, value in summary.items():
        typer.echo(
            f"{key}: {value}"
        )


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

    backend = OllamaBackend(
        model=model
    )

    profiler = Profiler(
        backend
    )

    trace = profiler.run(
        prompt
    )

    print_summary(
        trace
    )

    if output:
        save_trace(
            trace,
            output
        )

        typer.echo(
            f"\nTrace saved to {output}"
        )


app.add_typer(
    profile_app,
    name="profile",
)


if __name__ == "__main__":
    app()