import shutil
import subprocess
import time

import pytest

from depuzzle.backends.ollama import OllamaBackend


def wait_for_model_state(
    model: str,
    running: bool,
    timeout: float = 5.0,
) -> bool:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if ollama_model_running(model) == running:
            return True

        time.sleep(0.1)

    return False


def ollama_available() -> bool:
    return shutil.which("ollama") is not None


def ollama_model_running(model: str) -> bool:
    result = subprocess.run(
        ["ollama", "ps"],
        capture_output=True,
        text=True,
        check=True,
    )

    return model in result.stdout


@pytest.mark.integration
def test_backend_info():

    if not ollama_available():
        pytest.skip("Ollama binary not installed")

    backend = OllamaBackend("llama3.2:3b")

    if not ollama_model_running("llama3.2:3b"):
        pytest.skip("No running Ollama model available")

    info = backend.get_info()

    assert info.backend == "ollama"
    assert info.processor != ""
    assert info.context_length > 0


@pytest.mark.integration
def test_backend_prepare_and_unload():
    if not ollama_available():
        pytest.skip("Ollama binary not installed")

    backend = OllamaBackend("llama3.2:3b")

    backend = OllamaBackend("llama3.2:3b")

    backend.prepare()

    assert wait_for_model_state(
        "llama3.2:3b",
        running=True,
    )

    backend.unload()

    assert wait_for_model_state(
        "llama3.2:3b",
        running=False,
    )
