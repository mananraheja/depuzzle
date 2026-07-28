# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-07-28

### Added

- Added `BackendInfo` model to capture inference runtime metadata.
- Added runtime information to `InferenceTrace`.
- Added Ollama runtime detection using `ollama ps`.
- Added processor split (CPU/GPU) to runtime metadata.
- Added context length to runtime metadata.
- Added runtime information to saved trace JSON.
- Added runtime information to profile summary output.
- Added runtime comparison in `llmprof compare`.
- Added integration tests for the Ollama backend.

### Changed

- Separated unit tests from integration tests.
- Updated CI to skip integration tests by default.
- Improved trace schema to support runtime metadata.

### Fixed

- Improved handling of traces without runtime metadata.
- Improved parsing of `ollama ps` output.

## [0.1.0] - 2026-07-24

### Added

- Initial release of llmprof.
- CLI for profiling local LLM inference.
- Support for profiling Ollama models.
- Token-level inference trace collection.
- JSON trace export.
- Trace loading utilities.
- Inference metrics including:
  - token count
  - total latency
  - time to first token (TTFT)
  - tokens per second
- Trace comparison command.
- Rich terminal output for summaries and comparisons.
- Unit test suite.
- GitHub Actions continuous integration.
- Project documentation and installation instructions.

### Changed

- Established the initial project architecture around:
  - Backends
  - Profiler
  - Trace models
  - Metrics
  - CLI

### Fixed

- Initial public release.