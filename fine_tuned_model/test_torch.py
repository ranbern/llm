# test_torch.py
import torch

print(f"PyTorch version: {torch.__version__}")
# Check if MPS (Metal Performance Shaders) is available for Apple Silicon
print(f"Is MPS available: {torch.backends.mps.is_available()}")
print(f"Is MPS built: {torch.backends.mps.is_built()}")
print(f"Using device: {'mps' if torch.backends.mps.is_available() else 'cpu'}")

# Perform a simple tensor operation on the detected device
try:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x = torch.randn(5, 5, device=device) # Create tensor on device
    y = torch.randn(5, 5, device=device) # Create tensor on device
    z = torch.matmul(x, y)
    # Move result back to CPU to print if needed, avoids potential device print issues
    print("PyTorch simple operation successful.")
    # print(z.cpu()) # Optional: uncomment to see the result
except Exception as e:
    print(f"Error during PyTorch operation: {e}")

print("PyTorch test finished.")