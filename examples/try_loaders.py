from depuzzle.loaders import load_trace

trace = load_trace("run1.json")

print(trace.model)
print(len(trace.tokens))
print(trace.total_latency)
