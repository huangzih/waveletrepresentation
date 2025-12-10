#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import gc
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pywt
import timm
import torch
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

NUM_CLASSES = 5
BATCH_SIZE = 32
N_EPOCHS = 100
IMAGE_SIZE = 224
LEARNING_RATE = 1e-3
PATIENCE = 2
MAX_LR_CHANGES = 2
NUM_WORKERS = 2
BACKBONE_NAME = "vit_huge_patch14_224"
MODEL_ROOT = "./models/wavelet"
CSV_SEP = ";"


class WaveletDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        dimension: int,
        seq_len: int,
        label: bool = True,
        image_size: int = IMAGE_SIZE,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.label = label
        self.dimension = dimension
        self.seq_len = seq_len
        self.image_size = image_size
        if self.dimension not in (1, 2, 3):
            raise ValueError(f"dimension must be 1, 2 or 3, got {self.dimension}.")
        all_cols = ["pos_x", "pos_y", "pos_z"]
        self.dim_cols = all_cols[: self.dimension]
        for col in self.dim_cols:
            if col not in self.df.columns:
                raise KeyError(
                    f"dimension={self.dimension} but column '{col}' not found in DataFrame."
                )
        if self.label and "label" not in self.df.columns:
            raise KeyError("label=True but column 'label' not found in DataFrame.")
        self.scales = np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
             2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            dtype=np.float32,
        )
        self.wavelet_list = [
            "morl",
            "mexh",
            "cgau1",
            "cmor1.5-1.0",
            "shan1.5-1.0",
            "fbsp3-0.5-0.5",
        ]
        self.num_wavelets = len(self.wavelet_list)
        self.num_channels = self.num_wavelets * self.dimension

    @staticmethod
    def _string_to_array(s: str, seq_len: int) -> np.ndarray:
        values = np.array([float(v) for v in s.split(",")], dtype=np.float32)
        if len(values) != seq_len:
            raise ValueError(
                f"Sequence length mismatch: expected {seq_len}, got {len(values)}."
            )
        return values

    def _cwt_to_image(self, signal: np.ndarray, wavelet_name: str) -> torch.Tensor:
        coeffs, _ = pywt.cwt(signal, self.scales, wavelet=wavelet_name)
        spec = np.abs(coeffs)
        spec_resized = cv2.resize(spec, (self.image_size, self.image_size))
        spec_resized = spec_resized.astype(np.float32)
        return torch.from_numpy(spec_resized).unsqueeze(0)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        channels = []
        for col in self.dim_cols:
            arr = self._string_to_array(row[col], self.seq_len)
            for w_name in self.wavelet_list:
                channels.append(self._cwt_to_image(arr, w_name))
        wavelet_spec = torch.cat(channels, dim=0)
        if self.label:
            target = int(row["label"])
        else:
            target = 0
        return wavelet_spec.float(), target

    def __len__(self) -> int:
        return len(self.df)


class WaveletViTModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        backbone_name: str = BACKBONE_NAME,
        pretrained: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=0.0,
            drop_path_rate=0.0,
        )
        old_proj = backbone.patch_embed.proj
        if in_channels != old_proj.in_channels:
            new_proj = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                padding=old_proj.padding,
                bias=(old_proj.bias is not None),
            )
            nn.init.xavier_uniform_(new_proj.weight)
            if new_proj.bias is not None:
                nn.init.zeros_(new_proj.bias)
            backbone.patch_embed.proj = new_proj
        self.backbone = backbone
        embed_dim = self.backbone.num_features
        self.head = nn.Linear(embed_dim, out_dim)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        out = self.head(feat)
        return out


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    model_dir: str,
) -> float:
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
    for _, (inputs, targets) in enumerate(loop):
        inputs = inputs.to(device)
        targets_t = torch.as_tensor(targets, device=device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets_t)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    avg_loss = running_loss / max(1, len(train_loader))
    ckpt_path = os.path.join(model_dir, f"epoch{epoch}.pth")
    torch.save(model.state_dict(), ckpt_path)
    return avg_loss


def evaluate(
    model: nn.Module,
    valid_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    model_dir: str,
) -> Tuple[float, float]:
    model.eval()
    valid_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs = inputs.to(device)
            targets_t = torch.as_tensor(targets, device=device, dtype=torch.long)
            outputs = model(inputs)
            loss = criterion(outputs, targets_t)
            valid_loss += loss.item()
            preds = outputs.argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets_t.cpu().numpy())
    valid_loss /= max(1, len(valid_loader))
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    valid_f1 = f1_score(all_targets, all_preds, average="micro")
    log_line = f"Epoch: {epoch}\tValid Loss: {valid_loss:.4f}\tValid F1 (micro): {valid_f1:.4f}"
    print(log_line)
    log_path = os.path.join(model_dir, "log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")
    return valid_loss, valid_f1


def main(seq_length: int, dimension: int, train_path: str, valid_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_dir = os.path.join(MODEL_ROOT, f"len{seq_length}_dim{dimension}_vit_cls")
    os.makedirs(model_dir, exist_ok=True)
    train_df = pd.read_csv(train_path, sep=CSV_SEP)
    valid_df = pd.read_csv(valid_path, sep=CSV_SEP)
    print(f"There are {len(train_df)} samples in the training set.")
    print(f"There are {len(valid_df)} samples in the validation set.")
    train_dataset = WaveletDataset(train_df, dimension=dimension, seq_len=seq_length, label=True)
    valid_dataset = WaveletDataset(valid_df, dimension=dimension, seq_len=seq_length, label=True)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    model = WaveletViTModel(
        in_channels=train_dataset.num_channels,
        out_dim=NUM_CLASSES,
        backbone_name=BACKBONE_NAME,
        pretrained=True,
        device=device,
    )
    criterion = nn.CrossEntropyLoss()
    lr = LEARNING_RATE
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_valid_loss = float("inf")
    valid_losses = []
    f1_scores = []
    lr_changes = 0
    lr_reset_epoch = 0
    for epoch in range(N_EPOCHS):
        torch.cuda.empty_cache()
        gc.collect()
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            model_dir=model_dir,
        )
        valid_loss, valid_f1 = evaluate(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            model_dir=model_dir,
        )
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}, F1={valid_f1:.4f}"
        )
        valid_losses.append(valid_loss)
        f1_scores.append(valid_f1)
        if valid_loss < best_valid_loss - 1e-6:
            best_valid_loss = valid_loss
        elif (
            PATIENCE
            and epoch - lr_reset_epoch >= PATIENCE
            and min(valid_losses[-PATIENCE:]) > best_valid_loss
        ):
            lr_changes += 1
            if lr_changes > MAX_LR_CHANGES:
                print("Early stopping triggered.")
                break
            lr /= 5.0
            print(f"Learning rate updated to {lr:.6f}")
            lr_reset_epoch = epoch
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train wavelet-based ViT model on trajectory data (classification)."
    )
    parser.add_argument("length", type=int, help="Trajectory length (number of time steps).")
    parser.add_argument(
        "dimension",
        type=int,
        choices=[1, 2, 3],
        help="Trajectory dimension: 1 (pos_x), 2 (pos_x + pos_y), or 3 (pos_x + pos_y + pos_z).",
    )
    parser.add_argument("train_path", type=str, help="Path to training CSV file.")
    parser.add_argument("valid_path", type=str, help="Path to validation CSV file.")
    args = parser.parse_args()
    main(args.length, args.dimension, args.train_path, args.valid_path)
