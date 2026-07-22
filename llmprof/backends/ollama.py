import json
import httpx


class OllamaBackend:

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.host = host.rstrip("/")


    def generate(self, prompt: str):

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