import time
from datetime import datetime

from llmprof.models import (
    TokenEvent,
    InferenceTrace,
)


class Profiler:

    def __init__(self, backend):
        self.backend = backend

    def run(
        self,
        prompt: str,
    ) -> InferenceTrace:

        events: list[TokenEvent] = []

        start_time = datetime.now()
        start_counter = time.perf_counter()

        first_token_latency = None

        for token in self.backend.generate(prompt):

            current_time = datetime.now()

            if first_token_latency is None:
                first_token_latency = time.perf_counter() - start_counter

            event = TokenEvent(
                token=token,
                timestamp=current_time,
            )

            events.append(event)

        end_time = datetime.now()

        total_latency = time.perf_counter() - start_counter

        return InferenceTrace(
            model=self.backend.model,
            prompt=prompt,
            start_time=start_time,
            end_time=end_time,
            tokens=events,
            total_latency=total_latency,
            time_to_first_token=first_token_latency,
        )
