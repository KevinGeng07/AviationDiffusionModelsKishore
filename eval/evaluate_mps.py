"""
Same as eval/evaluate.py, but runs the models on the Apple GPU (MPS) of an M-series Mac.

Usage (from the repo root) — same arguments as evaluate.py:
    python eval/evaluate_mps.py                                  # every folder in FOLDERS (evaluate.py)
    python eval/evaluate_mps.py finished_ckpt/6_5_SWiDiT --max_batches 20
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import main

if __name__ == "__main__":
    if not torch.backends.mps.is_available():
        sys.exit("MPS is not available (needs an Apple-silicon Mac and a PyTorch build with MPS). "
                 "Use eval/evaluate.py to run on CPU/CUDA instead.")
    # num_workers 0 by default: each macOS worker process would copy the whole dataset
    if not any(a.startswith("--num_workers") for a in sys.argv):
        sys.argv += ["--num_workers", "0"]
    main(torch.device("mps"))
