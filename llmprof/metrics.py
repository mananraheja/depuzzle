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

        if self.trace.backend_info is not None:
            return self.trace.backend_info

    def summary(self):
        summary = {
            "model": self.trace.model,
            "tokens": self.token_count(),
            "latency_seconds": round(self.latency_seconds(), 3),
            "ttft_seconds": round(self.ttft_seconds(), 3),
            "tokens_per_second": round(self.tokens_per_second(), 2),
        }

        if self.trace.backend_info:
            summary["runtime"] = {
                "backend": self.trace.backend_info.backend,
                "processor": self.trace.backend_info.processor,
                "context_length": self.trace.backend_info.context_length,
            }

        return summary
