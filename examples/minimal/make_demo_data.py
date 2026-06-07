#!/usr/bin/env python
"""Generate the HomoeoGWAS minimal demo dataset (thin wrapper).

The generator lives in the installed package (``homoeogwas.demo_data``) so that
``homoeogwas demo`` works from a pip/conda install too. This script just calls it
for users browsing the repository.

Usage:  python make_demo_data.py [out_dir]
        homoeogwas demo -o out_dir        # equivalent + runs the fit
"""
import sys
from pathlib import Path

from homoeogwas.demo_data import make_demo

if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "demo_data"
    cfg = make_demo(d)
    print(f"[demo] wrote dataset under {Path(d).resolve()}")
    print(f"[demo] run:  homoeogwas fit -c {cfg} -o {Path(d) / 'demo_out'}")
