import numpy as np
import torch


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of contiguous sequences and their next-token targets from a 1D token array.

    Args:
        dataset (np.ndarray): 1D numpy array (or memmap) of integer token IDs, shape (n,)
        batch_size (int): Number of sequences to sample
        context_length (int): Sequence length per example
        device (str): Torch device string (e.g., 'cpu', 'cuda:0', 'mps')

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            inputs:  (batch_size, context_length)
            targets: (batch_size, context_length)
            Both are torch.long on the requested device.
    """
    n = int(dataset.shape[0])
    assert n >= context_length + 1, "Dataset too small for the requested context_length"

    # Start indices such that targets have valid next token
    # Valid starts: 0 .. n - context_length - 1 (inclusive)
    starts = np.random.randint(0, n - context_length, size=batch_size)

    x_batch = np.empty((batch_size, context_length), dtype=np.int64)
    y_batch = np.empty((batch_size, context_length), dtype=np.int64)
    for i, s in enumerate(starts):
        seq = dataset[s : s + context_length + 1]
        x_batch[i] = seq[:-1]
        y_batch[i] = seq[1:]

    inputs = torch.from_numpy(x_batch).to(device=device, dtype=torch.long)
    targets = torch.from_numpy(y_batch).to(device=device, dtype=torch.long)
    return inputs, targets
