import torch
from typing import BinaryIO, IO
import os


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    obj = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    torch.save(obj, out)


def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    obj = torch.load(src)
    model.load_state_dict(obj["model_state"])
    optimizer.load_state_dict(obj["optimizer_state"])
    return int(obj["iteration"])
