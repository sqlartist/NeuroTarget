"""Generate the simulated cohort into ../data  (python generate_data.py)"""
import argparse
import os
from pathlib import Path

from neurosim.cohort import generate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--ch", type=int, default=240)
    ap.add_argument("--stn", type=int, default=360)
    ap.add_argument("--gpi", type=int, default=160)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    counts = generate(Path(a.out), a.ch, a.stn, a.gpi, seed=a.seed, workers=a.workers)
    print("Generated:", counts, "->", a.out)
