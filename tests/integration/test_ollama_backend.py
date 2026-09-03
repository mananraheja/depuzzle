import json
import shutil
import subprocess
import time

import pytest

from depuzzle.backends.ollama import OllamaBackend
from depuzzle.models import Device, ExecutionConfig


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


def test_get_model_info(monkeypatch):
    backend = OllamaBackend("llama3.2:3b")

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "model_info": {
                    "general.architecture": "llama",
                    "llama.block_count": 28,
                }
            }

    def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(
        "depuzzle.backends.ollama.httpx.post",
        fake_post,
    )

    info = backend.get_model_info()

    assert info == {
        "general.architecture": "llama",
        "llama.block_count": 28,
    }


def test_get_layer_count(monkeypatch):
    backend = OllamaBackend("llama3.2:3b")

    def fake_get_model_info():
        return {
            "general.architecture": "llama",
            "llama.block_count": 28,
        }

    monkeypatch.setattr(
        backend,
        "get_model_info",
        fake_get_model_info,
    )

    assert backend.get_layer_count() == 28


def test_get_layer_count_missing_architecture(monkeypatch):
    backend = OllamaBackend("llama3.2:3b")

    monkeypatch.setattr(
        backend,
        "get_model_info",
        lambda: {},
    )

    with pytest.raises(
        RuntimeError,
        match="Model architecture not found",
    ):
        backend.get_layer_count()


def test_get_layer_count_missing_block_count(monkeypatch):
    backend = OllamaBackend("llama3.2:3b")

    monkeypatch.setattr(
        backend,
        "get_model_info",
        lambda: {
            "general.architecture": "llama",
        },
    )

    with pytest.raises(
        RuntimeError,
        match="Layer count not found",
    ):
        backend.get_layer_count()


def test_gpu_device_requests_all_model_layers(monkeypatch):
    backend = OllamaBackend("llama3.2:3b")

    monkeypatch.setattr(
        backend,
        "get_layer_count",
        lambda: 28,
    )

    captured = {}

    class FakeResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_lines(self):
            return [
                json.dumps(
                    {
                        "message": {"content": "Hello"},
                        "done": True,
                    }
                )
            ]

    def fake_stream(*args, **kwargs):
        captured["json"] = kwargs["json"]
        return FakeResponse()

    monkeypatch.setattr(
        "depuzzle.backends.ollama.httpx.stream",
        fake_stream,
    )

    execution_config = ExecutionConfig(
        device=Device.GPU,
    )

    list(
        backend.generate(
            "Hello",
            execution_config=execution_config,
        )
    )

    assert captured["json"]["options"]["num_gpu"] == 28


def test_hybrid_device_requests_configured_gpu_layers(monkeypatch):
    backend = OllamaBackend("llama3.2:3b")

    monkeypatch.setattr(
        backend,
        "get_layer_count",
        lambda: 28,
    )

    captured = {}

    class FakeResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_lines(self):
            return [
                json.dumps(
                    {
                        "message": {"content": "Hello"},
                        "done": True,
                    }
                )
            ]

    def fake_stream(*args, **kwargs):
        captured["json"] = kwargs["json"]
        return FakeResponse()

    monkeypatch.setattr(
        "depuzzle.backends.ollama.httpx.stream",
        fake_stream,
    )

    execution_config = ExecutionConfig(
        device=Device.HYBRID,
        gpu_layers=14,
    )

    list(
        backend.generate(
            "Hello",
            execution_config=execution_config,
        )
    )

    assert captured["json"]["options"]["num_gpu"] == 14


def test_hybrid_device_requires_gpu_layers(monkeypatch):
    backend = OllamaBackend("llama3.2:3b")

    monkeypatch.setattr(
        backend,
        "get_layer_count",
        lambda: 28,
    )

    with pytest.raises(
        ValueError,
        match="gpu_layers must be specified for hybrid execution",
    ):
        list(
            backend.generate(
                "Hello",
                execution_config=ExecutionConfig(
                    device=Device.HYBRID,
                ),
            )
        )


def test_hybrid_device_rejects_zero_gpu_layers(monkeypatch):
    backend = OllamaBackend("llama3.2:3b")

    monkeypatch.setattr(
        backend,
        "get_layer_count",
        lambda: 28,
    )

    with pytest.raises(ValueError, match="gpu_layers must be between 1 and 27"):
        list(
            backend.generate(
                "Hello",
                execution_config=ExecutionConfig(
                    device=Device.HYBRID,
                    gpu_layers=0,
                ),
            )
        )


def test_hybrid_device_rejects_all_model_layers(monkeypatch):
    backend = OllamaBackend("llama3.2:3b")

    monkeypatch.setattr(
        backend,
        "get_layer_count",
        lambda: 28,
    )

    with pytest.raises(
        ValueError,
        match="gpu_layers must be between 1 and 27",
    ):
        list(
            backend.generate(
                "Hello",
                execution_config=ExecutionConfig(
                    device=Device.HYBRID,
                    gpu_layers=28,
                ),
            )
        )


def test_gpu_device_uses_dynamic_model_layer_count(monkeypatch):
    backend = OllamaBackend("some-model")

    monkeypatch.setattr(
        backend,
        "get_layer_count",
        lambda: 42,
    )

    captured = {}

    class FakeResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_lines(self):
            return [
                json.dumps(
                    {
                        "message": {"content": "Hello"},
                        "done": True,
                    }
                )
            ]

    def fake_stream(*args, **kwargs):
        captured["json"] = kwargs["json"]
        return FakeResponse()

    monkeypatch.setattr(
        "depuzzle.backends.ollama.httpx.stream",
        fake_stream,
    )

    execution_config = ExecutionConfig(
        device=Device.GPU,
    )

    list(
        backend.generate(
            "Hello",
            execution_config=execution_config,
        )
    )

    assert captured["json"]["options"]["num_gpu"] == 42


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


def test_cpu_device_requests_zero_gpus(monkeypatch):
    backend = OllamaBackend("llama3.2:3b")

    captured = {}

    class FakeResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_lines(self):
            return [
                json.dumps(
                    {
                        "message": {"content": "Hello"},
                        "done": True,
                    }
                )
            ]

    def fake_stream(*args, **kwargs):
        captured["json"] = kwargs["json"]
        return FakeResponse()

    monkeypatch.setattr(
        "depuzzle.backends.ollama.httpx.stream",
        fake_stream,
    )

    execution_config = ExecutionConfig(
        device=Device.CPU,
    )

    list(
        backend.generate(
            "Hello",
            execution_config=execution_config,
        )
    )

    assert captured["json"]["options"]["num_gpu"] == 0
