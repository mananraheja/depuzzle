import json
import subprocess

import httpx

from depuzzle.backends.base import BaseBackend
from depuzzle.models import BackendInfo, RuntimeStats


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

    def generate(
        self,
        prompt: str,
        keep_alive: str | int | None = None,
    ):

        self.last_runtime_stats = None

        url = f"{self.host}/api/chat"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": True,
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
