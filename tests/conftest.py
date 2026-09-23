"""
Pin OpenMP to one thread before anything imports torch: two OpenMP runtimes
are loaded in this environment and PyTorch's threaded reductions crash the
process on tensors above 32,768 elements. The pipeline CLI does the same.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
