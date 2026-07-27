from llmprof.models import BackendInfo


def test_backend_info_creation():
    info = BackendInfo(
        backend="ollama",
        processor="72%/28% CPU/GPU",
        context_length=131072,
    )

    assert info.backend == "ollama"
    assert info.processor == "72%/28% CPU/GPU"
    assert info.context_length == 131072
