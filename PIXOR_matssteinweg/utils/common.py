import torch
import random
import numpy as np


def seed_everything(seed=123):
    """
    Seeds basic parameters for reproductibility of results.

    Args:
        seed (int, optional): Number of the seed. Defaults to 123.

    Returns:
        None
    """
    # Set seed for python
    random.seed(seed)

    # Set seed for python with numpy
    np.random.seed(seed)

    # Set seed for pytorch
    torch.manual_seed(seed)
    torch.Generator().manual_seed(seed)

    # Set seed for CUDA
    torch.cuda.manual_seed(seed)

    # Set seed for CUDA with new generator
    torch.cuda.manual_seed_all(seed)

    # Set seed for all devices (GPU and CPU)
    torch.backends.cudnn.deterministic = True

    # Set seed for all devices (GPU and CPU) - faster but not deterministic
    torch.backends.cudnn.benchmark = True
