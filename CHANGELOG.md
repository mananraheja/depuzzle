# Changelog

All notable changes to this project will be documented in this file.

## [0.2.2] - 2026-08-13

### Added

- Separate workflow for Release and Release-Test.
- Testing release and release-test workflows.

## [0.2.1] - 2026-08-12

### Fixed

- TestPyPI package upload and successful clean install
- PyPI package upload and successful clean install with pip install depuzzle.

## [0.2.0] - 2026-07-28

### Added

- Added runtime metadata collection for inference runs.
- Added backend runtime information to trace summaries:
  - Backend name
  - Processor utilization (CPU/GPU split)
  - Context length
- Added runtime metadata to JSON trace outputs.
- Added runtime comparison support for `depuzzle compare`.
- Added shared test fixtures using `conftest.py`.
- Added integration test coverage for Ollama backend functionality.

### Changed

- Updated trace summaries to display runtime information separately from inference metrics.
- Improved comparison output by separating runtime configuration from performance metrics.
- Refactored tests to use reusable fake backends and fake traces.
- Improved handling of missing runtime information.

### Fixed

- Fixed trace summary output formatting.
- Fixed comparison handling when runtime metadata is unavailable.
- Fixed CI failures caused by integration tests requiring Ollama.

### Testing

- Added unit tests for runtime metadata.
- Added tests for runtime comparison.
- Added Ollama integration test markers.
- Improved GitHub Actions CI workflow.

## [0.1.0] - 2026-07-24

### Added

- Initial release of depuzzle.
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