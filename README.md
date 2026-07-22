# llmprof

[![CI](https://github.com/mananraheja/llmprof/actions/workflows/ci.yml/badge.svg)](https://github.com/mananraheja/llmprof/actions)

A lightweight Python tool for analyzing and comparing LLM inference performance traces.

`llmprof` helps developers measure and compare inference characteristics such as latency, time-to-first-token, and token generation performance.

---

## Features

### v1.0

- Run profiler for Ollama hosted models on MacBook
- Load LLM inference traces from JSON files
- Validate trace format and schema
- Calculate inference performance metrics
- Compare two inference runs
- Identify performance regressions or improvements
- Automated testing with GitHub Actions

---

## Installation

Clone the repository:

```bash
git clone https://github.com/mananraheja/llmprof.git
cd llmprof
```

Create a virtual environment:

python -m venv .venv
source .venv/bin/activate

Install:

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

---

## Quick Start

Run profiler:

```bash
llmprof profile run \
--model <model_name> \
--prompt "<sample_prompt>" \
--output <path_to_output_file.json>
```

Compare two inference runs:

```bash
llmprof compare run1.json run2.json
```

Example:

```bash
Metric                  Run 1       Run 2       Change

Total Latency           12.5s       10.2s       -18%
Time To First Token     0.8s        0.6s        -25%
Tokens Generated        256         256          -
Tokens/sec              20.4        25.1        +23%
```

---

## Trace Format

`llmprof` expects traces in JSON format:
```json
{
  "model": "llama3.2:3b",
  "prompt": "Hello world",
  "start_time": "2026-07-22T10:00:00",
  "end_time": "2026-07-22T10:00:12",
  "tokens": [
    {
      "token": "Hello",
      "timestamp": "2026-07-22T10:00:01"
    }
  ],
  "total_latency": 12.0,
  "time_to_first_token": 1.0
}
```

---

## Architecture

`llmprof` consists of two main workflows:

### 1. Profile Run

The profiling workflow executes an inference run and captures performance data.

```bash
      User
        |
        v
llmprof profile run
        |
        v
    Profiler
        |
        v
Inference Backend
        |
        v
  Trace Recorder
        |
        v
   trace.json
```


The generated trace contains:
- model information
- prompt metadata
- token generation timestamps
- latency measurements
- time-to-first-token

### 2. Analyze and Compare

The analysis workflow processes captured traces.

```bash
    trace.json
        |
        v
  Trace Loader
        |
        v
InferenceTrace Model
        |
        v
  Metrics Engine
        |
        v
  Comparison Report
```

Core components:

- **Profiler**: Captures inference execution events and generates trace files.
- **Loader**: Parses and validates trace JSON files.
- **Models**: Provides structured representations of inference runs.
- **Metrics**: Calculates performance statistics.
- **Compare**: Compares multiple inference runs.

---

## Development

Run tests:

```bash
pytest
```

Format code:

```bash
black .
```

Lint:

```bash
ruff check .
```

Type checking:

```bash
mypy llmprof
```

---

## Roadmap

### v1.0
[x] Profiler run
[x] JSON trace loading
[x] Trace validation
[x] Latency metrics
[x] Run comparison
[x] CI pipeline

### Future
- Live inference profiling
- GPU utilization tracking
- Memory profiling
- Support for inference backends:
- llama.cpp
- MLX
- vLLM
- TensorRT-LLM

---

## LICENSE

MIT
