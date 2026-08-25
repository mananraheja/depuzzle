from depuzzle.models import Device, ExecutionConfig, Lifecycle, ProfileConfig
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


def test_hot_lifecycle_does_not_prepare_or_unload(fake_backend):
    config = ProfileConfig(
        lifecycle=Lifecycle.HOT,
        execution=ExecutionConfig(device=Device.CPU),
    )

    profiler = Profiler(fake_backend, config)

    profiler.run("Test")

    assert fake_backend.prepare_calls == 0
    assert fake_backend.unload_calls == 0


def test_warmup_lifecycle_prepares_backend(fake_backend):
    config = ProfileConfig(
        lifecycle=Lifecycle.WARMUP,
        execution=ExecutionConfig(device=Device.CPU),
    )

    profiler = Profiler(fake_backend, config)

    profiler.run("Test")

    assert fake_backend.prepare_calls == 1
    assert fake_backend.unload_calls == 0


def test_cold_lifecycle_unloads_backend(fake_backend):
    config = ProfileConfig(
        lifecycle=Lifecycle.COLD,
        execution=ExecutionConfig(device=Device.CPU),
    )

    profiler = Profiler(fake_backend, config)

    profiler.run("Test")

    assert fake_backend.unload_calls == 1
    assert fake_backend.prepare_calls == 0


def test_warmup_is_not_included_in_trace(fake_backend):
    config = ProfileConfig(
        lifecycle=Lifecycle.WARMUP,
        execution=ExecutionConfig(device=Device.CPU),
    )

    profiler = Profiler(fake_backend, config)

    trace = profiler.run("Test")

    assert fake_backend.generate_calls == 2
    assert len(trace.tokens) == 2


def test_trace_records_lifecycle(fake_backend):
    config = ProfileConfig(
        lifecycle=Lifecycle.WARMUP,
        execution=ExecutionConfig(device=Device.CPU),
    )

    profiler = Profiler(fake_backend, config)

    trace = profiler.run("Test")

    assert trace.lifecycle == Lifecycle.WARMUP
    assert trace.execution.device == Device.CPU
