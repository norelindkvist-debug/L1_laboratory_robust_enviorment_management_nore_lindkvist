## Environment verification

This project uses uv for dependency management.

A verification script (check_env.py) is included to validate that the
Python environment is correctly set up.

The script performs the following checks:

- Verifies that PyTorch is installed and prints the installed version
- Detects available hardware acceleration (CUDA / ROCm / MPS / CPU)
- Selects the best available device automatically
- Creates tensors on the selected device
- Executes a matrix multiplication and reduction to confirm that
  tensor computations work correctly on the selected backend

### To run the script you need to go into the terminal and type: 

- uv run python check_env.py

### Installed packages

- PyTorch
- Scikit-learn
- Pandas
- Jupyter