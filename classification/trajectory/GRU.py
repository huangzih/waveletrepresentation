import argparse
import os
import gc

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

CSV_SEP = ";"
MODEL_ROOT = "./models_rnn_gru"
BATCH_SIZE = 256
NUM_CLASSES = 5
HIDDEN_DIM = 64
LAYER_DIM = 3


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
            target = int(self.df["label"].iloc[index])
        else:
            target = 0
        return data_seq, target

    def __len__(self):
        return len(self.df)


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        batch_size = x.size(0)
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


def train_gru(seq_length, dimension, train_path, valid_path):
    df_train = pd.read_csv(train_path, sep=CSV_SEP)
    df_valid = pd.read_csv(valid_path, sep=CSV_SEP)
    print(f"There are {len(df_train)} samples in the training set.")
    print(f"There are {len(df_valid)} samples in the validation set.")

    train_dataset = AnDiDataset(df_train, seq_length, dimension, label=True)
    valid_dataset = AnDiDataset(df_valid, seq_length, dimension, label=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model_dir = os.path.join(MODEL_ROOT, f"len{seq_length}_dim{dimension}")
    os.makedirs(model_dir, exist_ok=True)

    model = GRUModel(dimension, HIDDEN_DIM, LAYER_DIM, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history_train = pd.DataFrame()
    history_valid = pd.DataFrame()

    n_epochs = 100
    init_epoch = 0
    max_lr_changes = 1
    valid_losses = []
    lr_reset_epoch = init_epoch
    patience = 2
    lr_changes = 0
    best_valid_loss = float("inf")

    for epoch in range(init_epoch, n_epochs):
        torch.cuda.empty_cache()
        gc.collect()

        model.train()
        t = tqdm(train_loader)
        for batch_idx, (seq_batch, label_batch) in enumerate(t):
            seq_batch = seq_batch.to(device).float()
            label_batch = label_batch.to(device).long()

            optimizer.zero_grad()
            output = model(seq_batch)
            loss = criterion(output, label_batch)
            t.set_description(f"train_loss (l={loss:.4f})")

            history_train.loc[epoch + batch_idx / len(train_loader), "train_loss"] = (
                loss.detach().cpu().numpy()
            )

            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), os.path.join(model_dir, f"epoch{epoch}.pth"))

        model.eval()
        valid_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (seq_batch, label_batch) in enumerate(valid_loader):
                all_targets.append(label_batch.numpy().copy())
                seq_batch = seq_batch.to(device).float()
                label_batch = label_batch.to(device).long()

                output = model(seq_batch)
                loss = criterion(output, label_batch)
                valid_loss += loss.item()
                preds = output.argmax(dim=1).cpu().numpy()
                all_predictions.append(preds)

        valid_loss = valid_loss / (batch_idx + 1)
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        valid_f1 = f1_score(all_targets, all_predictions, average="micro")

        history_valid.loc[epoch, "valid_loss"] = valid_loss

        status = (
            f"Epoch: {epoch}\tLR: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}"
            f"\tValid Loss: {valid_loss:.4f}\tValid F1: {valid_f1:.4f}"
        )
        print(status)
        with open(os.path.join(model_dir, "log.txt"), "a+", encoding="utf-8") as f:
            f.write(status + "\n")

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
    train_gru(args.length, args.dimension, args.train_path, args.valid_path)


if __name__ == "__main__":
    main()
