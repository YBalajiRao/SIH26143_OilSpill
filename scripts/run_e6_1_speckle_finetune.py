import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.oil_spill_dataset import GulfSARPatchDataset
from src.datasets.transforms import get_train_transforms
from src.segmentation.proposed_model import PhysioGraphSpillPerception
from src.segmentation.loss_functions import ComboBCEDiceLoss

def finetune_robustness(epochs=3, lr=5e-5):
    root = r"D:\SIH26143_OilSpill"
    ckpt_in = os.path.join(root, "models", "checkpoints", "E5_2_proposed_best.pth")
    ckpt_out = os.path.join(root, "models", "checkpoints", "E6_1_robust_finetuned.pth")
    
    csv_path = os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "train", "dataframe_train_dataset_256_90.csv")
    img_dir = os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "train", "images")
    mask_dir = os.path.join(root, "data", "raw", "gulf_mexico", "extracted", "train", "masks")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PhysioGraphSpillPerception(in_channels=1, out_classes=1).to(device)
    model.load_state_dict(torch.load(ckpt_in, map_location=device)["model_state_dict"], strict=False)
    
    train_tf = get_train_transforms(speckle_var=0.20)
    train_ds = GulfSARPatchDataset(csv_path, img_dir, mask_dir, transform=train_tf)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)

    criterion = ComboBCEDiceLoss(bce_weight=0.5, dice_weight=0.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print("    Fine-Tuning E5.2 with Heavy Speckle Augmentation (var=0.20)...")
    model.train()
    for epoch in range(1, epochs + 1):
        loss_sum = 0
        for imgs, masks in tqdm(train_loader, desc=f"    Fine-Tune Epoch {epoch}/{epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(f"    Epoch {epoch}/{epochs} Mean Loss: {loss_sum/len(train_loader):.4f}")

    torch.save({"model_state_dict": model.state_dict()}, ckpt_out)
    print(f"    [✓] Robust Checkpoint Saved -> {ckpt_out}\n")

if __name__ == "__main__":
    finetune_robustness()
