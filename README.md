# DePuzzle

[![CI](https://github.com/mananraheja/depuzzle/actions/workflows/ci.yml/badge.svg)](https://github.com/mananraheja/depuzzle/actions)

A lightweight Python tool for profiling, analyzing, and comparing LLM inference performance.

`depuzzle` helps developers profile local LLM inference runs, capture token-level traces, and compare performance across models, lifecycle states, and execution configurations.

---

## Usage

### Profile an inference run

```bash
depuzzle profile run \
  --model llama3.2:3b \
  --prompt "Explain virtual memory in one paragraph."
```

By default, depuzzle runs a hot inference using CPU execution and saves the trace under traces/.

You can control the profiling lifecycle and execution device:

```bash
depuzzle profile run \
  --model llama3.2:3b \
  --lifecycle cold \
  --device gpu
```

For hybrid CPU/GPU execution, specify the number of layers to offload:

```bash
depuzzle profile run \
  --model llama3.2:3b \
  --device hybrid \
  --gpu-layers 14
```

Run multiple independent measurements:

```bash
depuzzle profile run \
  --model llama3.2:3b \
  --runs 5
```

Each run is saved as a separate trace artifact, allowing the results to be analyzed later.

### Compare inference runs

```bash
depuzzle compare traces/run1.json traces/run2.json
```

This compares key inference metrics between two profiling runs, including latency, time to first token, tokens per second, and runtime statistics.

### Lifecycle configuration

depuzzle supports three profiling lifecycle states:

- cold — unload the model before the measured inference.
- warmup — prepare the model and perform an unmeasured warmup inference before measurement.
- hot — measure inference with the model already loaded.

Example:

```bash
depuzzle profile run \
  --model llama3.2:3b \
  --lifecycle cold
```

### Execution configuration

Execution can be configured for CPU, GPU, or hybrid CPU/GPU placement.

```bash
# CPU
depuzzle profile run \
  --model llama3.2:3b \
  --device cpu

# GPU
depuzzle profile run \
  --model llama3.2:3b \
  --device gpu

# Hybrid CPU/GPU
depuzzle profile run \
  --model llama3.2:3b \
  --device hybrid \
  --gpu-layers 14
```

For hybrid execution, --gpu-layers specifies how many model layers are requested for GPU offloading.

### Multiple runs

Use --runs to collect multiple independent inference traces:

```bash
depuzzle profile run \
  --model llama3.2:3b \
  --runs 5
```

Each run is stored as a separate JSON trace under the output directory:

```bash
traces/
├── llama3.2_3b_cpu_hot_run_1.json
├── llama3.2_3b_cpu_hot_run_2.json
├── llama3.2_3b_cpu_hot_run_3.json
├── llama3.2_3b_cpu_hot_run_4.json
└── llama3.2_3b_cpu_hot_run_5.json
```

This allows repeated measurements to be analyzed independently and provides the foundation for benchmark aggregation and statistical analysis.

NOTE: Execution placement is currently represented as part of the profiling configuration and trace model. Backend-specific placement control is being implemented incrementally.

---

## Features

- Profile local LLM inference runs
- Capture token-level inference traces
- Configure inference run lifecycle
- Profile cold, warmup, and hot runs
- Configure execution placement
- Record lifecycle and execution configuration in traces
- Control backend model preparation and unloading
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
  "lifecycle": "hot",
  "execution": {
    "device": "cpu",
    "gpu_layers": null
  },
  "backend_info": {
    "backend": "ollama",
    "processor": "100% CPU",
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
               ProfileConfig
              /             \
             /               \
       Lifecycle          Execution
             \               /
              \             /
                   Profiler
                      |
                      v
                Backend Adapter
                      |
              +-------+-------+
              |               |
          prepare()        unload()
              |
              v
           Inference
              |
              v
         InferenceTrace
              |
              v
           trace.json
```

The profiling configuration consists of:

- **Lifecycle** — controls whether the run is cold, warmup, or hot.
- **Execution** — describes the intended execution device and backend-specific execution configuration.
- **Profiler** — coordinates the configured profiling run.
- **Backend Adapter** — handles backend-specific model preparation, inference, and unloading.
- **InferenceTrace** — records the configuration and measurements associated with the run.

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

### v0.1.0 - Basic MVP supporting profile run and compare
- [x] Profile local LLM inference runs
- [x] Export traces to JSON
- [x] Calculate latency and throughput metrics
- [x] Compare inference runs
- [x] GitHub Actions CI

### v0.2.4 - Added Backend runtime info and release configs
- [x] Backend runtime metadata
- [x] Runtime information in trace summaries
- [x] Runtime comparison between inference runs
- [x] PyPI project upload and clean pip install

### v0.3.0 — Lifecycle-Aware Profiling

- [x] Run lifecycle
  - [x] Cold
  - [x] Warmup
  - [x] Hot
- [x] Execution configuration
  - [x] CPU configuration
  - [x] GPU configuration abstraction
  - [x] Hybrid configuration abstraction
- [x] Lifecycle-aware profiling
- [x] Lifecycle metadata in traces
- [x] Execution metadata in traces
- [x] Backend `prepare()` / `unload()` interface
- [x] Ollama lifecycle integration
- [x] Lifecycle test coverage
- [x] CPU vs GPU comparison
- [x] Cold vs hot comparison

### v0.3.x — Backend Runtime Instrumentation

- [ ] Capture backend runtime statistics
- [ ] Model load time
- [ ] Prefill latency
- [ ] Decode latency
- [ ] Decode tokens/sec
- [ ] Execution scaling comparisons

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
