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
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

OUTPUT_DIM = 1
BATCH_SIZE = 32
N_EPOCHS = 100
IMAGE_SIZE = 256
LEARNING_RATE = 1e-3
PATIENCE = 2
MAX_LR_CHANGES = 2
NUM_WORKERS = 2
BACKBONE_NAME = "efficientnet_b5"
MODEL_ROOT = "./models/wavelet_reg"
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
            raise ValueError("dimension must be 1, 2 or 3, got {}.".format(self.dimension))

        all_cols = ["pos_x", "pos_y", "pos_z"]
        self.dim_cols = all_cols[: self.dimension]

        for col in self.dim_cols:
            if col not in self.df.columns:
                raise KeyError(
                    "dimension={} but column '{}' not found in DataFrame.".format(
                        self.dimension, col
                    )
                )

        if self.label and "label" not in self.df.columns:
            raise KeyError("label=True but column 'label' not found in DataFrame.")

        self.scales = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

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
                "Sequence length mismatch: expected {}, got {}.".format(seq_len, len(values))
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
            target = float(row["label"])
        else:
            target = 0.0

        return wavelet_spec.float(), target

    def __len__(self) -> int:
        return len(self.df)


class WaveletExtractor(nn.Module):
    def __init__(self, in_channels: int, model_name: str = BACKBONE_NAME, pretrained: bool = True):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=pretrained)
        if not hasattr(backbone, "conv_stem"):
            raise RuntimeError("Backbone {} has no attribute 'conv_stem'.".format(model_name))

        old_conv = backbone.conv_stem
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        backbone.conv_stem = new_conv
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature_extractor(x)
        b, c, h, w = out.shape
        wavelet_features = out.view(b, c, -1)
        return wavelet_features


class WaveletModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        image_size: int = IMAGE_SIZE,
        backbone_name: str = BACKBONE_NAME,
        pretrained: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cpu")

        self.extractor = WaveletExtractor(in_channels, model_name=backbone_name, pretrained=pretrained)
        self.extractor.to(device)
        self.extractor.eval()

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size, device=device)
            features = self.extractor(dummy)
            _, feature_dim, _ = features.shape

        self.regressor = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(feature_dim, out_dim),
        )
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extractor(x)
        features = features.permute(0, 2, 1)
        last_step = features[:, -1, :]
        preds = self.regressor(last_step)
        return preds


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
    loop = tqdm(train_loader, desc="Epoch {} [train]".format(epoch))

    for _, (inputs, targets) in enumerate(loop):
        inputs = inputs.to(device)
        targets_t = torch.as_tensor(targets, device=device, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze(-1)
        loss = criterion(outputs, targets_t)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / max(1, len(train_loader))

    ckpt_path = os.path.join(model_dir, "epoch{}.pth".format(epoch))
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
    mae_total = 0.0
    n_batches = 0

    mae_fn = nn.L1Loss()

    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs = inputs.to(device)
            targets_t = torch.as_tensor(targets, device=device, dtype=torch.float32)

            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, targets_t)
            mae = mae_fn(outputs, targets_t)

            valid_loss += loss.item()
            mae_total += mae.item()
            n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0

    valid_loss /= n_batches
    mae_avg = mae_total / n_batches

    log_line = "Epoch: {}\tValid Loss (MSE): {:.4f}\tValid MAE: {:.4f}".format(
        epoch, valid_loss, mae_avg
    )
    print(log_line)

    log_path = os.path.join(model_dir, "log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")

    return valid_loss, mae_avg


def main(seq_length: int, dimension: int, train_path: str, valid_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    model_dir = os.path.join(MODEL_ROOT, "len{}_dim{}".format(seq_length, dimension))
    os.makedirs(model_dir, exist_ok=True)

    train_df = pd.read_csv(train_path, sep=CSV_SEP)
    valid_df = pd.read_csv(valid_path, sep=CSV_SEP)

    print("There are {} samples in the training set.".format(len(train_df)))
    print("There are {} samples in the validation set.".format(len(valid_df)))

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

    model = WaveletModel(
        in_channels=train_dataset.num_channels,
        out_dim=OUTPUT_DIM,
        image_size=IMAGE_SIZE,
        backbone_name=BACKBONE_NAME,
        pretrained=True,
        device=device,
    )

    criterion = nn.MSELoss()
    lr = LEARNING_RATE
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_valid_loss = float("inf")
    valid_losses = []
    mae_scores = []
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

        valid_loss, valid_mae = evaluate(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            model_dir=model_dir,
        )

        print(
            "Epoch {}: train_loss(MSE)={:.4f}, valid_loss(MSE)={:.4f}, MAE={:.4f}".format(
                epoch, train_loss, valid_loss, valid_mae
            )
        )

        valid_losses.append(valid_loss)
        mae_scores.append(valid_mae)

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
            print("Learning rate updated to {:.6f}".format(lr))
            lr_reset_epoch = epoch
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train wavelet-based EfficientNet model on trajectory data (regression)."
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
