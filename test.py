import torch
from DDPG.DDPG_agent import Agent

def find_1d_input_length(model_path, device="cpu", batch_size=1, min_len=1, max_len=5000):
    """
    Finds the input vector length for a 1D PyTorch model.

    Args:
        model_path (str): Path to .pt or .pth model file
        device (str): 'cpu' or 'cuda'
        batch_size (int): Batch size to test
        min_len (int): Minimum input length to try
        max_len (int): Maximum input length to try

    Returns:
        tuple or None: (batch_size, length) if found
    """
    # Load model
    model = torch.load(model_path, map_location=device)
    if hasattr(model, 'eval'):
        model.eval()
    else:
        raise ValueError("The loaded object is not a PyTorch model.")

    # Try different vector lengths
    for length in range(min_len, max_len + 1):
        try:
            x = torch.randn(batch_size, length).to(device)
            with torch.no_grad():
                model(x)
            print(f"✅ Model works with input shape: ({batch_size}, {length})")
            return (batch_size, length)
        except Exception:
            continue

    print("❌ Could not determine input length.")
    return None

# Example usage
path = r'C:\Users\harsh\OneDrive\Desktop\truck_trailer_DDPG\DDPG\tmp\ddpg\actor_ddpg'
shape = find_1d_input_length(path)
print("Detected input shape:", shape)
