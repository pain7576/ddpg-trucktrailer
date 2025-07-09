import random
import numpy as np
import torch as T

def set_seed(seed):
    """
    Set seed for all random number generators to ensure reproducibility
    """
    # Python random module
    random.seed(seed)

    # NumPy random module
    np.random.seed(seed)

    # PyTorch random module
    T.manual_seed(seed)

    # PyTorch CUDA random module (if using GPU)
    if T.cuda.is_available():
        T.cuda.manual_seed(seed)
        T.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # Make PyTorch deterministic (may impact performance)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False

    print(f"🎲 Seed set to: {seed}")
    print("   - Python random module seeded")
    print("   - NumPy random module seeded")
    print("   - PyTorch random module seeded")
    if T.cuda.is_available():
        print("   - PyTorch CUDA random module seeded")
        print("   - CUDNN deterministic mode enabled")