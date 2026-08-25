from depuzzle.profiler import Profiler


def test_profiler_creates_trace(fake_backend, profile_config):

    profiler = Profiler(fake_backend, profile_config)

    trace = profiler.run("Say hello")

    assert trace.model == "fake-model"

    assert trace.prompt == "Say hello"

    assert len(trace.tokens) == 2


def test_profiler_records_latency(fake_backend, profile_config):

    profiler = Profiler(fake_backend, profile_config)

    trace = profiler.run("Test")

    assert trace.total_latency >= 0


def test_first_token_latency_exists(fake_backend, profile_config):

    profiler = Profiler(fake_backend, profile_config)

    trace = profiler.run("Test")

    assert trace.time_to_first_token is not None
