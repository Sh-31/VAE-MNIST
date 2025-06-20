import os
import torch

def load_checkpoint(checkpoint_path, device="cpu"):
        """Load model checkpoint."""
        if not os.path.exists(checkpoint_path):
            return
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        return checkpoint