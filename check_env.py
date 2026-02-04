import torch

if torch.cuda.is_available():
    device = "cuda"
    backend = "ROCm" if torch.version.hip else "CUDA"
elif torch.backends.mps.is_available():
    device = "mps"
    backend = "MPS"
else:
    device = "cpu"
    backend = "CPU"

print(f"Using device: {device} ({backend} backend)")
print("PyTorch version:", torch.__version__)

a = torch.randn(1000, 1000, device=device)
b = torch.randn(1000, 1000, device=device)

c = a @ b
result = c.mean()

print("Tensor computation successful")
print("Result:", result.item())
print("Tensor device:", c.device)