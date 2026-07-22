import typer

app = typer.Typer(
    name="llmprof",
    help="Inference profiler for local LLMs.",
)


profile_app = typer.Typer(
    help="Profile LLM inference requests."
)


@profile_app.command()
def run():
    """
    Profile a single LLM inference request.
    """
    typer.echo("LLMProf profiler started")


app.add_typer(
    profile_app,
    name="profile"
)


if __name__ == "__main__":
    app()