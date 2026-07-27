from llmprof.backends.ollama import OllamaBackend


def test_parse_ollama_ps():

    output = """
NAME            ID              SIZE     PROCESSOR          CONTEXT     UNTIL
llama3.2:3b     a80c4f17acd5    23 GB    72%/28% CPU/GPU    131072      2 minutes from \
    now
"""

    backend = OllamaBackend("llama3.2:3b")

    info = backend._parse_ollama_ps(output)

    assert info.backend == "ollama"
    assert info.processor == "72%/28% CPU/GPU"
    assert info.context_length == 131072
