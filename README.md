# DePuzzle

[![CI](https://github.com/mananraheja/depuzzle/actions/workflows/ci.yml/badge.svg)](https://github.com/mananraheja/depuzzle/actions)

A lightweight Python tool for profiling, analyzing, and comparing LLM inference performance.

`depuzzle` helps developers profile local LLM inference runs, capture token-level traces, and compare performance across models and configurations.

---

## Usage

### Profile an LLM inference run

depuzzle can profile local LLM inference requests and capture token-level timing information.

Example:

```bash
depuzzle profile run \
  --model qwen2.5:3b \
  --prompt "Explain virtual memory in one paragraph." \
  --output run1.json
```

Example output:

```bash
--- Trace Summary ---

model: qwen2.5:3b
tokens: 633
latency_seconds: 31.13s
ttft_seconds: 4.71s
tokens_per_second: 20.33

runtime:
  backend: ollama
  processor: 72%/28% CPU/GPU
  context_length: 131072

Trace saved to run1.json
```

### Compare inference runs

depuzzle can compare two inference traces to evaluate model performance differences.

Example:

```bash
depuzzle compare sample_traces/run1.json sample_traces/run2.json
```

Example output:

```bash
--- Comparison ---

Metric                   Run 1               Run 2               Change
------------------------------------------------------------------------------------------
model                    qwen2.5:3b          llama3.2:3b         -
tokens                   633                 555                 -12.32%
latency_seconds          31.13               28.41               -8.74% (faster)
ttft_seconds             4.71                6.98                +48.07% (slower)
tokens_per_second        20.33               19.53               -3.93% (worse)


runtime
    backend             ollama               ollama               same
    processor           72%/28% CPU/GPU      100% CPU             changed
    context_length      131072               131072               same
```

---

## Features

- Profile local LLM inference runs
- Capture token-level inference traces
- Measure:
  - Total latency
  - Time to First Token (TTFT)
  - Tokens per second
- Record runtime information
  - Backend
  - CPU/GPU processor split
  - Context length
- Export traces as JSON
- Compare inference runs
- Automated testing with GitHub Actions

---

## Supported Backends

Current:

- Ollama

Planned:

- MLX
- llama.cpp
- vLLM
- TensorRT-LLM

---

## Installation

Now available directly on PyPI. Install latest version via:

```bash
pip install depuzzle
```

or specific release version via:

```bash
pip install depuzzle==<version>
```

Clone the repository:

```bash
git clone https://github.com/mananraheja/depuzzle.git
cd depuzzle
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

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
depuzzle profile run \
--model <model_name> \
--prompt "<sample_prompt>" \
--output <path_to_output_file.json>
```

Compare two inference runs:

```bash
depuzzle compare run1.json run2.json
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

`depuzzle` expects traces in JSON format:
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
    "time_to_first_token": 1.0,
    "backend_info": {
    "backend": "ollama",
    "processor": "72%/28% CPU/GPU",
    "context_length": 131072
  }
}
```

---

## Architecture

`depuzzle` consists of two main workflows:

### 1. Profile Run

The profiling workflow executes an inference run and captures performance data.

```bash
      User
        |
        v
depuzzle profile run
        |
        v
    Profiler
        |
        v
Backend Adapter
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
- backend information
- processor utilization
- context length

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
pytest -m "not integration"
```

Run integration tests (requires Ollama and downloaded local models):
```bash
pytest -m integration
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
mypy depuzzle
```

---

## Roadmap

### v0.1.0
- [x] Profile local LLM inference runs
- [x] Export traces to JSON
- [x] Calculate latency and throughput metrics
- [x] Compare inference runs
- [x] GitHub Actions CI

### v0.2.2
- [x] Backend runtime metadata
- [x] Runtime information in trace summaries
- [x] Runtime comparison between inference runs
- [x] PyPI project upload and clean pip install

### Future

#### Additional Backends
- [ ] MLX
- [ ] llama.cpp
- [ ] vLLM
- [ ] TensorRT-LLM
- [ ] Hugging Face Transformers

#### Profiling
- [ ] Memory usage
- [ ] GPU utilization
- [ ] Live inference profiling
- [ ] Batch benchmarking
- [ ] Multi-run statistical summaries

#### Visualization
- [ ] Interactive timeline visualization
- [ ] Trace diff visualization
- [ ] HTML report generation

#### AI Agent Integration
- [ ] MCP server
- [ ] Additional MCP tools
- [ ] Benchmark orchestration through MCP

#### Long-term
- [ ] Layer-by-layer profiling
- [ ] Distributed inference profiling
- [ ] Multi-node benchmarking
- [ ] Plugin system for custom backends

---

## LICENSE

MIT
