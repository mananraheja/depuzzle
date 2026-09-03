class Metrics:
    def __init__(self, trace):
        self.trace = trace

    def token_count(self):
        return len(self.trace.tokens)

    def latency_seconds(self):
        return self.trace.total_latency

    def ttft_seconds(self):
        return self.trace.time_to_first_token

    def tokens_per_second(self):
        latency = self.latency_seconds()

        if latency == 0:
            return 0

        return self.token_count() / latency

    def runtime_details(self):
        return self.trace.backend_info

    def runtime_stats(self):
        return getattr(self.trace, "runtime_stats", None)

    def load_duration_seconds(self):
        stats = self.runtime_stats()

        if stats is None or stats.load_duration is None:
            return None

        return stats.load_duration / 1_000_000_000

    def prompt_eval_duration_seconds(self):
        stats = self.runtime_stats()

        if stats is None or stats.prompt_eval_duration is None:
            return None

        return stats.prompt_eval_duration / 1_000_000_000

    def prompt_token_count(self):
        stats = self.runtime_stats()

        if stats is None:
            return None

        return stats.prompt_eval_count

    def generation_duration_seconds(self):
        stats = self.runtime_stats()

        if stats is None or stats.eval_duration is None:
            return None

        return stats.eval_duration / 1_000_000_000

    def generation_token_count(self):
        stats = self.runtime_stats()

        if stats is None:
            return None

        return stats.eval_count

    def generation_tokens_per_second(self):
        stats = self.runtime_stats()

        if (
            stats is None
            or stats.eval_count is None
            or stats.eval_duration is None
            or stats.eval_duration == 0
        ):
            return None

        return stats.eval_count / (stats.eval_duration / 1_000_000_000)

    def prompt_tokens_per_second(self):
        stats = self.runtime_stats()

        if (
            stats is None
            or stats.prompt_eval_count is None
            or stats.prompt_eval_duration is None
            or stats.prompt_eval_duration == 0
        ):
            return None

        return stats.prompt_eval_count / (stats.prompt_eval_duration / 1_000_000_000)

    def summary(self):
        summary = {
            "model": self.trace.model,
            "tokens": self.token_count(),
            "latency_seconds": round(self.latency_seconds(), 3),
            "ttft_seconds": (
                round(self.ttft_seconds(), 3)
                if self.ttft_seconds() is not None
                else None
            ),
            "tokens_per_second": round(self.tokens_per_second(), 2),
        }

        stats = self.runtime_stats()

        if stats is not None:
            summary["ollama"] = {
                "load_duration_seconds": (
                    round(self.load_duration_seconds(), 3)
                    if self.load_duration_seconds() is not None
                    else None
                ),
                "prompt_eval_duration_seconds": (
                    round(self.prompt_eval_duration_seconds(), 3)
                    if self.prompt_eval_duration_seconds() is not None
                    else None
                ),
                "prompt_eval_count": self.prompt_token_count(),
                "generation_duration_seconds": (
                    round(self.generation_duration_seconds(), 3)
                    if self.generation_duration_seconds() is not None
                    else None
                ),
                "generation_count": self.generation_token_count(),
                "generation_tokens_per_second": (
                    round(self.generation_tokens_per_second(), 2)
                    if self.generation_tokens_per_second() is not None
                    else None
                ),
                "prompt_tokens_per_second": (
                    round(self.prompt_tokens_per_second(), 2)
                    if self.prompt_tokens_per_second() is not None
                    else None
                ),
            }

        runtime = self.runtime_details()

        if runtime is not None:
            summary["runtime"] = {
                "backend": runtime.backend,
                "processor": runtime.processor,
                "context_length": runtime.context_length,
            }

        return summary
