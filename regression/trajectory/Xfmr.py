import argparse
import os
import gc

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

CSV_SEP = ";"
MODEL_ROOT = "./models/trans_reg"
BATCH_SIZE = 256


class AnDiDataset(Dataset):
    def __init__(self, df, seq_length, dimension, label=True):
        self.df = df.copy()
        self.seq_length = seq_length
        self.dimension = dimension
        self.label = label
        self.dim_cols = ["pos_x", "pos_y", "pos_z"][:dimension]

    def __getitem__(self, index):
        channels = []
        for col in self.dim_cols:
            values = [float(v) for v in self.df[col].iloc[index].split(",")]
            if len(values) != self.seq_length:
                raise ValueError(
                    f"Sequence length mismatch in column {col}: expected {self.seq_length}, got {len(values)}"
                )
            channels.append(values)
        data_seq = torch.tensor(channels, dtype=torch.float32)
        if self.label:
            target = float(self.df["label"].iloc[index])
        else:
            target = 0.0
        return data_seq, target

    def __len__(self):
        return len(self.df)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        scores = self.attn(x)
        weights = torch.nn.functional.softmax(scores, dim=1)
        pooled = torch.sum(x * weights, dim=1)
        return pooled


class TransformerTimeSeriesRegressor(nn.Module):
    def __init__(
        self,
        input_dim=2,
        cnn_channels=64,
        num_cnn_layers=3,
        transformer_layers=4,
        nhead=4,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.cnn_layers = nn.Sequential()
        in_channels = input_dim
        for i in range(num_cnn_layers):
            self.cnn_layers.add_module(
                f"conv_{i}", nn.Conv1d(in_channels, cnn_channels, kernel_size=3, stride=1, padding=1)
            )
            self.cnn_layers.add_module(f"bn_{i}", nn.BatchNorm1d(cnn_channels))
            self.cnn_layers.add_module(f"relu_{i}", nn.ReLU())
            self.cnn_layers.add_module(f"dropout_{i}", nn.Dropout(dropout))
            in_channels = cnn_channels
        self.pos_encoder = PositionalEncoding(d_model=cnn_channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.attn_pooling = SelfAttentionPooling(cnn_channels)
        self.regressor = nn.Sequential(
            nn.Linear(cnn_channels, cnn_channels // 2),
            nn.BatchNorm1d(cnn_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cnn_channels // 2, 1),
        )

    def forward(self, x):
        cnn_out = self.cnn_layers(x)
        trans_in = cnn_out.transpose(1, 2)
        pos_encoded = self.pos_encoder(trans_in)
        transformer_out = self.transformer_encoder(pos_encoded)
        pooled = self.attn_pooling(transformer_out)
        output = self.regressor(pooled).squeeze(-1)
        return output


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, history=None):
    model.train()
    t = tqdm(loader)
    for batch_idx, (seq_batch, label_batch) in enumerate(t):
        seq_batch = seq_batch.to(device)
        label_batch = label_batch.to(device).float()
        optimizer.zero_grad()
        output = model(seq_batch)
        loss = criterion(output, label_batch)
        t.set_description(f"train_loss (l={loss:.4f})")
        if history is not None:
            history.loc[epoch + batch_idx / len(loader), "train_loss"] = loss.detach().cpu().numpy()
        loss.backward()
        optimizer.step()


def evaluate(model, loader, criterion, device, epoch, history=None, model_path=None):
    model.eval()
    valid_loss = 0.0
    all_predictions, all_targets = [], []
    with torch.no_grad():
        for batch_idx, (seq_batch, label_batch) in enumerate(loader):
            seq_batch = seq_batch.to(device)
            label_batch = label_batch.to(device).float()
            output = model(seq_batch)
            loss = criterion(output, label_batch)
            valid_loss += loss.item()
            all_predictions.append(output.detach().cpu().numpy())
            all_targets.append(label_batch.detach().cpu().numpy())
    y_pred = np.concatenate(all_predictions).reshape(-1)
    y_true = np.concatenate(all_targets).reshape(-1)
    valid_loss = valid_loss / (batch_idx + 1)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    if history is not None:
        history.loc[epoch, "valid_loss"] = valid_loss
        history.loc[epoch, "valid_mae"] = mae
    status = f"Epoch: {epoch}\tValid Loss: {valid_loss:.4f}\tValid MAE: {mae:.4f}"
    print(status)
    if model_path is not None:
        with open(os.path.join(model_path, "log.txt"), "a+", encoding="utf-8") as f:
            f.write(status + "\n")
    return valid_loss, mae


def train_transformer(seq_length, dimension, train_path, valid_path):
    df_train = pd.read_csv(train_path, sep=CSV_SEP)
    df_valid = pd.read_csv(valid_path, sep=CSV_SEP)
    print(f"There are {len(df_train)} samples in the training set.")
    print(f"There are {len(df_valid)} samples in the validation set.")


    train_dataset = AnDiDataset(df_train, seq_length=seq_length, dimension=dimension, label=True)
    valid_dataset = AnDiDataset(df_valid, seq_length=seq_length, dimension=dimension, label=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model_dir = os.path.join(MODEL_ROOT, f"len{seq_length}_dim{dimension}")
    os.makedirs(model_dir, exist_ok=True)

    model = TransformerTimeSeriesRegressor(input_dim=dimension).to(device)
    criterion = nn.MSELoss()
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history_train = pd.DataFrame()
    history_valid = pd.DataFrame()

    n_epochs = 100
    init_epoch = 0
    max_lr_changes = 2
    valid_losses = []
    lr_reset_epoch = init_epoch
    patience = 2
    lr_changes = 0
    best_valid_loss = float("inf")

    for epoch in range(init_epoch, n_epochs):
        torch.cuda.empty_cache()
        gc.collect()
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, history_train)
        valid_loss, valid_mae = evaluate(model, valid_loader, criterion, device, epoch, history_valid, model_dir)
        torch.save(model.state_dict(), os.path.join(model_dir, f"epoch{epoch}.pth"))
        valid_losses.append(valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
        elif patience and epoch - lr_reset_epoch > patience and min(valid_losses[-patience:]) > best_valid_loss:
            lr_changes += 1
            if lr_changes > max_lr_changes:
                break
            lr /= 5
            print(f"lr updated to {lr}")
            lr_reset_epoch = epoch
            optimizer.param_groups[0]["lr"] = lr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("length", type=int)
    parser.add_argument("dimension", type=int, choices=[1, 2, 3])
    parser.add_argument("train_path", type=str)
    parser.add_argument("valid_path", type=str)
    args = parser.parse_args()
    train_transformer(args.length, args.dimension, args.train_path, args.valid_path)


if __name__ == "__main__":
    main()
