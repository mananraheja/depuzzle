import json
import subprocess

import httpx

from depuzzle.backends.base import BaseBackend
from depuzzle.models import BackendInfo


class OllamaBackend(BaseBackend):

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.host = host.rstrip("/")

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
