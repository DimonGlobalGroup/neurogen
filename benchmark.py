#!/usr/bin/env python3
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interface import Neurogen

def bench(name, src, inputs, n=100):
    times = []
    for _ in range(n):
        ng = Neurogen().load(src)
        t0 = time.perf_counter()
        ng.run(inputs)
        times.append(time.perf_counter() - t0)
    avg = sum(times)/len(times)*1000
    print(f"{name:20s}  avg {avg:7.3f} ms  over {n} runs")

if __name__ == "__main__":
    print("NEUROGEN v0.2 micro-benchmarks")
    bench("math_ops", open("examples/math_ops.ng").read(), {"x":1.0,"y":2.0})
    bench("vector_ops", open("examples/vector_ops.ng").read(), {"data":[1,-1,2,-2,3]})
    bench("pipeline", open("examples/pipeline.ng").read(), {"raw":[0.1,0.2,0.3,0.4]})
