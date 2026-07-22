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

    def summary(self):

        return {
            "model": self.trace.model,
            "tokens": self.token_count(),
            "latency_seconds": self.latency_seconds(),
            "ttft_seconds": self.ttft_seconds(),
            "tokens_per_second": self.tokens_per_second(),
        }
