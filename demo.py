#!/usr/bin/env python3
"""NEUROGEN v0.2 end-to-end demonstration."""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interface import Neurogen

def main():
    print("=" * 70)
    print("NEUROGEN v0.2 — Demonstration")
    print("=" * 70)

    demos = [
        ("math_ops.ng", {"x": 2.5, "y": 1.5}),
        ("vector_ops.ng", {"data": [1.0, -0.5, 2.0, -1.5, 0.3]}),
        ("logic.ng", {"a": True, "b": False}),
        ("pipeline.ng", {"raw": [0.2, -0.4, 0.9, 0.1, -0.2]}),
        ("motif_reuse.ng", {"signal": 1.0}),
    ]

    for fname, inputs in demos:
        print(f"\n--- {fname} ---")
        src = open(f"examples/{fname}").read()
        ng = Neurogen().load(src)
        result = ng.run(inputs)
        info = ng.inspect()
        print(f"  nodes={info['nodes']}  result={result}")

    print("\n" + "=" * 70)
    print("All demonstrations executed successfully.")
    print("=" * 70)

if __name__ == "__main__":
    main()
