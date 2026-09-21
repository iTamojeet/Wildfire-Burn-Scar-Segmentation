"""
04_training/dataset.py

CHANGED: manifest paths are relative to where tile_dataset.py was run
from (03_tiling/), not where train.py runs from (04_training/).
Resolving each entry's path relative to the manifest file's own
directory fixes this regardless of which directory a script is later
run from.
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset

REFLECTANCE_SCALE = 10000.0
NOT_APPLICABLE_CODE = 4
IGNORE_INDEX = 255


class BurnSeverityDataset(Dataset):
    def __init__(self, manifest_path):
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        # NEW: resolve paths relative to the manifest's directory
        manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
        for entry in self.manifest:
            entry["image_path"] = self._resolve(entry["image_path"], manifest_dir)
            entry["label_path"] = self._resolve(entry["label_path"], manifest_dir)

    @staticmethod
    def _resolve(stored_path, manifest_dir):
        # stored_path looks like "data/patches/train/train_00001_img.npy"
        # (relative to 03_tiling/). The actual file lives in the same
        # directory as the manifest itself, so just take the basename
        # and join it with manifest_dir.
        filename = os.path.basename(stored_path)
        return os.path.join(manifest_dir, filename)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        image = np.load(entry["image_path"]).astype(np.float32) / REFLECTANCE_SCALE
        label = np.load(entry["label_path"]).astype(np.int64)

        label[label == NOT_APPLICABLE_CODE] = IGNORE_INDEX

        return torch.from_numpy(image), torch.from_numpy(label)