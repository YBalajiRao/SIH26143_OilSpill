import os
import glob
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

class GulfSARPatchDataset(Dataset):
    """
    RAM-cached patch dataset for Gulf of Mexico Sentinel-1 SAR scenes.
    Ensures identical float32 dB scaling across all transform modes.
    """
    def __init__(self, csv_path, img_dir, mask_dir, transform=None, db_min=-35.0, db_max=5.0):
        self.transform = transform
        self.db_min = db_min
        self.db_max = db_max
        self.df = pd.read_csv(csv_path)

        self.scene_images = {}
        self.scene_masks = {}

        img_files = glob.glob(os.path.join(img_dir, "*.tif"))
        if not img_files:
            raise FileNotFoundError(f"No .tif images found in: {img_dir}")

        for img_path in img_files:
            fname = os.path.basename(img_path)
            mask_path = os.path.join(mask_dir, fname)

            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

            if img is None or mask is None:
                continue

            img = np.nan_to_num(img, nan=db_min, posinf=db_max, neginf=db_min).astype(np.float32)
            self.scene_images[fname] = img
            self.scene_masks[fname] = (mask > 0.5).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = os.path.basename(str(row["paths"]).replace("\\", "/"))
        coord_str = str(row["coordinates"]).strip('"').strip("'")
        y, x = (int(v.strip()) for v in coord_str.split(","))

        full_img = self.scene_images[fname]
        full_mask = self.scene_masks[fname]
        H, W = full_img.shape[:2]

        y = max(0, min(y, H - 256))
        x = max(0, min(x, W - 256))

        img = full_img[y : y + 256, x : x + 256].copy()
        mask = full_mask[y : y + 256, x : x + 256].copy()

        img = np.clip(img, self.db_min, self.db_max)
        img = (img - self.db_min) / (self.db_max - self.db_min)
        img = np.ascontiguousarray(img[..., None])

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img_t = augmented["image"]
            mask_t = augmented["mask"]
        else:
            img_t = torch.from_numpy(img).permute(2, 0, 1).float()
            mask_t = torch.from_numpy(mask)

        if isinstance(mask_t, np.ndarray):
            mask_t = torch.from_numpy(mask_t)
        if mask_t.dim() == 2:
            mask_t = mask_t.unsqueeze(0)

        return img_t.float(), mask_t.float()
