import torch
from torch.utils.data import Dataset, TensorDataset, StackDataset
import numpy as np


def makeTorchDataset(features:list[np.ndarray], genmatching:np.ndarray, device="cpu"):
    return StackDataset(features=TensorDataset(torch.tensor(np.stack(features, axis=1), device=device)),
        genmatching=TensorDataset(torch.tensor(genmatching, device=device, dtype=torch.float32)))

