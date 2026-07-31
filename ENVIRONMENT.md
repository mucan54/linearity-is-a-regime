# Environment used for the reported runs

All numbers in the paper were produced with the versions below. `requirements.txt`
pins the package set; this file records the exact interpreter and hardware so the
GPU-dependent runs (the depth, dual-arm, common-mode and phase studies) are
reproducible bit-for-bit where CUDA determinism allows.

| component | version |
|---|---|
| Python | 3.12 |
| PyTorch | (record `python3 -c "import torch;print(torch.__version__, torch.version.cuda)"`) |
| NumPy | (record `python3 -c "import numpy;print(numpy.__version__)"`) |
| SciPy | (record `python3 -c "import scipy;print(scipy.__version__)"`) |
| Matplotlib | (record `python3 -c "import matplotlib;print(matplotlib.__version__)"`) |
| GPU | NVIDIA GB10 (Grace Blackwell), 128 GB unified memory |

TF32 matmul is enabled (`torch.backends.cuda.matmul.allow_tf32 = True`) in the
training scripts. Seeds are fixed per run (0, 1, 2); the reported figures are
means over seeds with the standard deviation stored alongside in the JSON files.

The CPU-only scripts (`sim_ppa_breakeven.py`, `sim_roofline.py`,
`sim_microring_activation.py`, `sim_softmax_cascade.py`, `sim_energy_budget.py`,
`analysis_heldout.py`, and the plotting scripts) are deterministic and do not
require a GPU.
