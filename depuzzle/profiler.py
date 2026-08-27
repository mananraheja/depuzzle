import time
from datetime import UTC, datetime

from depuzzle.models import (
    InferenceTrace,
    Lifecycle,
    ProfileConfig,
    TokenEvent,
)


class Profiler:

    def __init__(self, backend, config: ProfileConfig):
        self.backend = backend
        self.config = config

    def _warmup(self, prompt: str) -> None:
        """Run a warmup inference to prepare the model for profiling."""
        for _ in self.backend.generate(prompt):
            pass

    def run(
        self,
        prompt: str,
    ) -> InferenceTrace:

        if self.config.lifecycle == Lifecycle.COLD:
            self.backend.unload()

        elif self.config.lifecycle == Lifecycle.WARMUP:
            self.backend.prepare()
            self._warmup(prompt)

        events: list[TokenEvent] = []

        start_time = datetime.now(UTC)
        start_counter = time.perf_counter()

        first_token_latency = None

        for token in self.backend.generate(
            prompt, execution_config=self.config.execution
        ):

            current_time = datetime.now(UTC)

            if first_token_latency is None:
                first_token_latency = time.perf_counter() - start_counter

            event = TokenEvent(
                token=token,
                timestamp=current_time,
            )

            events.append(event)

        end_time = datetime.now(UTC)

        total_latency = time.perf_counter() - start_counter

        return InferenceTrace(
            model=self.backend.model,
            prompt=prompt,
            start_time=start_time,
            end_time=end_time,
            tokens=events,
            total_latency=total_latency,
            time_to_first_token=first_token_latency,
            lifecycle=self.config.lifecycle,
            execution=self.config.execution,
            runtime_stats=self.backend.last_runtime_stats,
            backend_info=self.backend.get_info(),
        )
