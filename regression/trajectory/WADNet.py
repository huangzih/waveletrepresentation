import argparse
import os
import gc
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import LSTM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

BATCH_SIZE = 256
N_EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 2
MAX_LR_CHANGES = 2
MODEL_ROOT = "./models/wad"
CSV_SEP = ";"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AnDiDataset(Dataset):
    def __init__(self, df, seq_length, dimension=2, label=True):
        self.df = df.copy()
        self.seq_length = seq_length
        self.dimension = dimension
        self.label = label
        all_cols = ["pos_x", "pos_y", "pos_z"]
        self.dim_cols = all_cols[: self.dimension]
        for col in self.dim_cols:
            if col not in self.df.columns:
                raise KeyError(f"column '{col}' not found in DataFrame")

    def __getitem__(self, index):
        row = self.df.iloc[index]
        channels = []
        for col in self.dim_cols:
            values = [float(v) for v in row[col].split(",")]
            if len(values) != self.seq_length:
                raise ValueError(
                    f"sequence length mismatch in column {col}: expected {self.seq_length}, got {len(values)}"
                )
            channels.append(torch.tensor(values, dtype=torch.float32))
        data_seq = torch.stack(channels, dim=0)
        if self.label:
            target = float(row["label"])
        else:
            target = 0.0
        return data_seq, target

    def __len__(self):
        return len(self.df)


class Wave_LSTM_Layer(nn.Module):
    def __init__(self, filters, kernel_size, dilation_depth, input_dim, hidden_dim, layer_dim):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_depth = dilation_depth
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.dilations = [2 ** i for i in range(dilation_depth)]
        self.conv1d_tanh = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=filters,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    padding=dilation,
                    dilation=dilation,
                )
                for dilation in self.dilations
            ]
        )
        self.conv1d_sigm = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=filters,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    padding=dilation,
                    dilation=dilation,
                )
                for dilation in self.dilations
            ]
        )
        self.conv1d_0 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=1,
        )
        self.conv1d_1 = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=1,
            padding=0,
        )
        self.post = nn.Sequential(nn.BatchNorm1d(filters), nn.Dropout(0.1))
        self.lstm = LSTM(filters, hidden_dim, layer_dim, batch_first=True)

    def forward(self, x):
        x = self.conv1d_0(x)
        res_x = x
        for i in range(self.dilation_depth):
            tahn_out = torch.tanh(self.conv1d_tanh[i](x))
            sigm_out = torch.sigmoid(self.conv1d_sigm[i](x))
            x = tahn_out * sigm_out
            x = self.conv1d_1(x)
            res_x = res_x + x
        x = self.post(res_x)
        x = x.permute(0, 2, 1)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=device).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=device).requires_grad_()
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        return out.permute(0, 2, 1)


class AnDiModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim):
        super().__init__()
        self.wave_lstm_1 = Wave_LSTM_Layer(32, 3, 16, input_dim, hidden_dim, layer_dim)
        self.fc = nn.Sequential(nn.Dropout(p=0.1), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        x = self.wave_lstm_1(x)
        x = self.fc(x.permute(0, 2, 1)[:, -1, :]).squeeze(-1)
        return x


def train_one_epoch(model, loader, criterion, optimizer, epoch, model_dir):
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, desc=f"Epoch {epoch} [train]")
    for seq_batch, label_batch in loop:
        seq_batch = seq_batch.to(device)
        label_batch = label_batch.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        output = model(seq_batch)
        loss = criterion(output, label_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    avg_loss = running_loss / max(1, len(loader))
    ckpt_path = os.path.join(model_dir, f"epoch{epoch}.pth")
    torch.save(model.state_dict(), ckpt_path)
    return avg_loss


def evaluate(model, loader, criterion, optimizer, epoch, model_dir):
    model.eval()
    valid_loss = 0.0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for seq_batch, label_batch in loader:
            all_targets.append(label_batch.numpy().copy())
            seq_batch = seq_batch.to(device)
            label_batch = label_batch.to(device, dtype=torch.float32)
            output = model(seq_batch)
            loss = criterion(output, label_batch)
            valid_loss += loss.item()
            all_predictions.append(output.cpu().numpy())
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    valid_loss /= max(1, len(loader))
    mae = float(np.mean(np.abs(all_predictions - all_targets)))
    log_line = "Epoch: {}\tLR: {:.6f}\tValid Loss (MSE): {:.4f}\tValid MAE: {:.4f}".format(
        epoch, optimizer.state_dict()["param_groups"][0]["lr"], valid_loss, mae
    )
    print(log_line)
    log_path = os.path.join(model_dir, "log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")
    return valid_loss, mae


def main(seq_length, dimension, train_path, valid_path):
    print("The validation fold is 0")
    model_dir = os.path.join(MODEL_ROOT, f"len{seq_length}_dim{dimension}_reg")
    os.makedirs(model_dir, exist_ok=True)
    train_df = pd.read_csv(train_path, sep=CSV_SEP)
    valid_df = pd.read_csv(valid_path, sep=CSV_SEP)
    print("There are {} samples in the training set.".format(len(train_df)))
    print("There are {} samples in the validation set.".format(len(valid_df)))
    train_dataset = AnDiDataset(train_df, seq_length=seq_length, dimension=dimension, label=True)
    valid_dataset = AnDiDataset(valid_df, seq_length=seq_length, dimension=dimension, label=True)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=pin_memory)
    print(device)
    model = AnDiModel(input_dim=dimension, hidden_dim=64, layer_dim=3).to(device)
    criterion = nn.MSELoss()
    lr = LEARNING_RATE
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n_epochs = N_EPOCHS
    init_epoch = 0
    max_lr_changes = MAX_LR_CHANGES
    valid_losses = []
    lr_reset_epoch = init_epoch
    patience = PATIENCE
    lr_changes = 0
    best_valid_loss = float("inf")
    for epoch in range(init_epoch, n_epochs):
        torch.cuda.empty_cache()
        gc.collect()
        train_one_epoch(model, train_loader, criterion, optimizer, epoch, model_dir)
        valid_loss, mae = evaluate(model, valid_loader, criterion, optimizer, epoch, model_dir)
        valid_losses.append(valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
        elif patience and epoch - lr_reset_epoch > patience and min(valid_losses[-patience:]) > best_valid_loss:
            lr_changes += 1
            if lr_changes > max_lr_changes:
                break
            lr /= 5.0
            print(f"lr updated to {lr}")
            lr_reset_epoch = epoch
            optimizer.param_groups[0]["lr"] = lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("length", type=int)
    parser.add_argument("dimension", type=int, choices=[1, 2, 3])
    parser.add_argument("train_path", type=str)
    parser.add_argument("valid_path", type=str)
    args = parser.parse_args()
    main(args.length, args.dimension, args.train_path, args.valid_path)
