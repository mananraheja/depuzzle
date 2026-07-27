import subprocess

import pytest

from llmprof.backends.ollama import OllamaBackend


def ollama_model_running() -> bool:
    result = subprocess.run(
        ["ollama", "ps"],
        capture_output=True,
        text=True,
    )

    lines = result.stdout.strip().splitlines()

    return len(lines) >= 2


@pytest.mark.integration
def test_backend_info():

    if not ollama_model_running():
        pytest.skip("No running Ollama model available")

    backend = OllamaBackend("llama3.2:3b")

    info = backend.get_info()

    assert info.backend == "ollama"
    assert info.processor != ""
    assert info.context_length > 0
