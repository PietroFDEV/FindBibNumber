import torch

# Check if CUDA is available
print(torch.cuda.is_available())

# Print CUDA version
print(torch.version.cuda)

# Print PyTorch version
print(torch.__version__)