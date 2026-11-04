# Spectral Recovery Learning (SRL) for Measurement-Free Quantum Error Correction

Code repository for the paper `Spectral Recovery Learning for Measurement-Free Quantum Error Correction`

For optimal performance, it is strongly recommended to run the jobs on high-performance GPU servers

## Installation

Requires Python >= 3.12. Results were produced with the exact versions pinned in `pyproject.toml` / `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Choose one:

```bash
# GPU (NVIDIA, CUDA 13)
pip install -e ".[gpu]"

# CPU only
pip install -e .
```

To also run the IBM hardware scripts in `hardware/`, add the `hardware` extra, e.g. `pip install -e ".[gpu,hardware]"`.

Alternatively, `pip install -r requirements.txt` installs the exact pinned set (GPU build plus hardware dependencies).

Main dependencies: `jax`, `optax`, `pennylane`, `numpy`, `scipy`, `matplotlib` (plus `qiskit`, `qiskit-aer`, `qiskit-ibm-runtime` for hardware runs).