import h5py
import torch
from torch.utils.data import Dataset


class H5BehaviorCloningDataset(Dataset):
    def __init__(self, h5_path: str, indices=None):
        self.h5_path = h5_path
        self._h5 = h5py.File(h5_path, "r")
        self.obs = self._h5["obs"]
        self.act = self._h5["act"]
        n = len(self.obs)
        self.indices = list(range(n)) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return torch.from_numpy(self.obs[i]), torch.from_numpy(self.act[i])

    def close(self):
        if self._h5:
            self._h5.close()
            self._h5 = None
