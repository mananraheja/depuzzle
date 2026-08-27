import json
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


def test_generate_captures_runtime_stats(monkeypatch):
    backend = OllamaBackend("llama3.2:3b")

    lines = [
        json.dumps(
            {
                "message": {
                    "content": "Hello",
                },
                "done": False,
            }
        ),
        json.dumps(
            {
                "message": {
                    "content": " world",
                },
                "done": True,
                "load_duration": 100_000_000,
                "prompt_eval_duration": 200_000_000,
                "prompt_eval_count": 10,
                "eval_duration": 300_000_000,
                "eval_count": 20,
            }
        ),
    ]

    class FakeResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_lines(self):
            return lines

    def fake_stream(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(
        "depuzzle.backends.ollama.httpx.stream",
        fake_stream,
    )

    tokens = list(backend.generate("Hello"))

    assert tokens == ["Hello", " world"]

    assert backend.last_runtime_stats is not None
    assert backend.last_runtime_stats.load_duration == 100_000_000
    assert backend.last_runtime_stats.prompt_eval_duration == 200_000_000
    assert backend.last_runtime_stats.prompt_eval_count == 10
    assert backend.last_runtime_stats.eval_duration == 300_000_000
    assert backend.last_runtime_stats.eval_count == 20
