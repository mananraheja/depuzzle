import json
import subprocess

import httpx

from depuzzle.backends.base import BaseBackend
from depuzzle.models import BackendInfo, Device, ExecutionConfig, RuntimeStats


class OllamaBackend(BaseBackend):
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.last_runtime_stats: RuntimeStats | None = None

    def _request_model(self, keep_alive: int) -> None:
        url = f"{self.host}/api/generate"

        payload = {
            "model": self.model,
            "prompt": "",
            "keep_alive": keep_alive,
            "stream": False,
        }

        response = httpx.post(
            url,
            json=payload,
            timeout=None,
        )

        response.raise_for_status()

    def prepare(self) -> None:
        """Ensure the model remains loaded."""
        self._request_model(keep_alive=-1)

    def unload(self) -> None:
        """Unload the model from memory."""
        self._request_model(keep_alive=0)

    def _parse_ollama_ps(self, output: str) -> BackendInfo:

        lines = output.strip().splitlines()

        if len(lines) < 2:
            raise RuntimeError("No running Ollama models found")

        row = lines[1]

        columns = row.split()
        print(columns)

        processor = f"{columns[4]} {columns[5]}"
        context = int(columns[6])

        return BackendInfo(
            backend="ollama",
            processor=processor,
            context_length=context,
        )

    def get_info(self) -> BackendInfo:
        result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            check=True,
        )

        output = result.stdout

        return self._parse_ollama_ps(output)

    def get_model_info(self) -> dict:
        """Return model metadata from Ollama."""
        url = f"{self.host}/api/show"

        response = httpx.post(
            url,
            json={"model": self.model},
            timeout=None,
        )
        response.raise_for_status()

        data = response.json()
        return data.get("model_info", {})

    def get_layer_count(self) -> int:
        """Return the number of transformer blocks in the model."""
        model_info = self.get_model_info()

        architecture = model_info.get("general.architecture")
        if not architecture:
            raise RuntimeError("Model architecture not found")

        key = f"{architecture}.block_count"
        layer_count = model_info.get(key)

        if layer_count is None:
            raise RuntimeError(f"Layer count not found in Ollama model metadata: {key}")

        return int(layer_count)

    def generate(
        self,
        prompt: str,
        execution_config: ExecutionConfig | None = None,
        keep_alive: str | int | None = None,
    ):

        self.last_runtime_stats = None

        url = f"{self.host}/api/chat"

        options = {}

        if execution_config is not None:
            if execution_config.device == Device.CPU:
                options["num_gpu"] = 0

            elif execution_config.device == Device.GPU:
                options["num_gpu"] = self.get_layer_count()

            elif execution_config.device == Device.HYBRID:
                print(execution_config.gpu_layers)
                if execution_config.gpu_layers is None:
                    raise ValueError(
                        "gpu_layers must be specified for hybrid execution"
                    )

                layer_count = self.get_layer_count()

                if not 0 < execution_config.gpu_layers < layer_count:
                    raise ValueError(
                        f"gpu_layers must be between 1 and {layer_count - 1} for \
                            hybrid execution"
                    )

                options["num_gpu"] = execution_config.gpu_layers

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": True,
            "options": options,
        }

        with httpx.stream(
            "POST",
            url,
            json=payload,
            timeout=None,
        ) as response:
            if response.status_code != 200:
                print(response.text)

            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)

                    if "message" in data:
                        yield data["message"]["content"]

                    if data.get("done"):
                        self.last_runtime_stats = RuntimeStats(
                            load_duration=data.get("load_duration"),
                            prompt_eval_duration=data.get("prompt_eval_duration"),
                            prompt_eval_count=data.get("prompt_eval_count"),
                            eval_duration=data.get("eval_duration"),
                            eval_count=data.get("eval_count"),
                        )
