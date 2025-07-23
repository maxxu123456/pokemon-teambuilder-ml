import json
import math
from pathlib import Path
import random
from collections import defaultdict
import os

# for mac users
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


data_dir = Path("data/run5-100pages-allgens")
input_json_path = data_dir / "data.json"
model_path = data_dir / "model.pth"

d_model = 256  # embed dim
nhead = 8  # attention heads
num_encoder_layers = 4
num_decoder_layers = 4
dim_feedforward = 1024
dropout = 0.1

epochs = 2
patience = 5
lr = 1e-3
batch_size = 32
train_frac = 0.8
val_frac = 0.1
test_frac = 0.1
seed = 42

random.seed(seed)
torch.manual_seed(seed)


def split_data(json_path, train_frac, val_frac, test_frac, seed):
    with open(json_path, "r") as f:
        data = json.load(f)
    data = [
        d
        for d in data
        if len(d.get("winning_team", [])) == 6 and len(d.get("losing_team", [])) == 6
    ]
    train_val, test = train_test_split(data, test_size=test_frac, random_state=seed)
    val_size = val_frac / (train_frac + val_frac)
    train, val = train_test_split(train_val, test_size=val_size, random_state=seed)
    return train, val, test


def build_vocab(data, min_freq=1):
    species_freq = defaultdict(int)
    for ex in data:
        for team in ("winning_team", "losing_team"):
            for p_name in ex[team]:
                species_freq[p_name] += 1

    species_itos = ["<pad>", "<unk>", "<bos>", "<eos>"]
    for p, c in sorted(species_freq.items()):
        if c >= min_freq:
            species_itos.append(p)
    species_stoi = {tok: i for i, tok in enumerate(species_itos)}

    return species_stoi, species_itos


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class PokemonTeamDataset(Dataset):

    def __init__(self, data, species_stoi, max_len=6):
        self.data = data
        self.species_stoi = species_stoi
        self.max_len = max_len
        self.pad_idx = species_stoi["<pad>"]
        self.unk_idx = species_stoi["<unk>"]
        self.bos_idx = species_stoi["<bos>"]
        self.eos_idx = species_stoi["<eos>"]

    def __len__(self):
        return len(self.data)

    def tokenize_team(self, team):
        species_ids = [self.species_stoi.get(p, self.unk_idx) for p in team]
        padding_needed = self.max_len - len(species_ids)
        species_ids += [self.pad_idx] * padding_needed
        return torch.tensor(species_ids, dtype=torch.long)

    def __getitem__(self, i):
        ex = self.data[i]

        src_species = self.tokenize_team(ex["losing_team"])

        tgt_team = ex["winning_team"]
        dec_input_species = [self.bos_idx] + [
            self.species_stoi.get(p, self.unk_idx) for p in tgt_team[:-1]
        ]
        tgt_output_species = [
            self.species_stoi.get(p, self.unk_idx) for p in tgt_team
        ] + [self.eos_idx]

        dec_padding = (self.max_len + 1) - len(dec_input_species)
        tgt_padding = (self.max_len + 1) - len(tgt_output_species)

        dec_input_species += [self.pad_idx] * dec_padding
        tgt_output_species += [self.pad_idx] * tgt_padding

        return {
            "src_species": src_species,
            "dec_input_species": torch.tensor(dec_input_species, dtype=torch.long),
            "tgt_output_species": torch.tensor(tgt_output_species, dtype=torch.long),
        }


class ConditionalTeamGenerator(nn.Module):

    def __init__(
        self,
        num_species,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout,
    ):
        super().__init__()
        self.d_model = d_model

        self.species_embedding = nn.Embedding(num_species, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.generator = nn.Linear(d_model, num_species)

    def _create_padding_mask(self, tensor, pad_idx):
        return tensor == pad_idx

    def forward(self, src_species, tgt_species, src_pad_idx, tgt_pad_idx):
        src_padding_mask = self._create_padding_mask(src_species, src_pad_idx)
        tgt_padding_mask = self._create_padding_mask(tgt_species, tgt_pad_idx)
        tgt_mask = self.transformer.generate_square_subsequent_mask(
            tgt_species.size(1)
        ).to(tgt_species.device)

        src_emb = self.species_embedding(src_species) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb.transpose(0, 1)).transpose(0, 1)

        tgt_emb = self.species_embedding(tgt_species) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb.transpose(0, 1)).transpose(0, 1)

        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.generator(output)


def train_epoch(model, loader, optimizer, scheduler, criterion, device, pad_idx):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        src_species = batch["src_species"].to(device)
        dec_input = batch["dec_input_species"].to(device)
        tgt_output = batch["tgt_output_species"].to(device)

        logits = model(src_species, dec_input, pad_idx, pad_idx)

        optimizer.zero_grad()
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, pad_idx):
    model.eval()
    total_loss = 0
    for batch in tqdm(loader, desc="Evaluating"):
        src_species = batch["src_species"].to(device)
        dec_input = batch["dec_input_species"].to(device)
        tgt_output = batch["tgt_output_species"].to(device)

        logits = model(src_species, dec_input, pad_idx, pad_idx)

        loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.view(-1))
        total_loss += loss.item()

    return total_loss / len(loader)


if __name__ == "__main__":
    train_data, val_data, test_data = split_data(
        input_json_path, train_frac, val_frac, test_frac, seed
    )
    print(
        f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test"
    )

    species_stoi, species_itos = build_vocab(train_data)
    pad_idx = species_stoi["<pad>"]

    train_ds = PokemonTeamDataset(train_data, species_stoi)
    val_ds = PokemonTeamDataset(val_data, species_stoi)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    model = ConditionalTeamGenerator(
        num_species=len(species_stoi),
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * len(train_loader)
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for ep in range(1, epochs + 1):
        t_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, pad_idx
        )
        v_loss = eval_epoch(model, val_loader, criterion, device, pad_idx)

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        print(f"Epoch {ep:02d} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss

            checkpoint = {
                "model_state": model.state_dict(),
                "species_stoi": species_stoi,
                "species_itos": species_itos,
                "hyperparameters": {
                    "d_model": d_model,
                    "nhead": nhead,
                    "num_encoder_layers": num_encoder_layers,
                    "num_decoder_layers": num_decoder_layers,
                    "dim_feedforward": dim_feedforward,
                    "dropout": dropout,
                },
            }
            torch.save(checkpoint, model_path)

            print(f"Validation loss improved. Model saved to {model_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"No improvement for {patience} epochs. Stopping early.")
                break

    epochs_range = list(range(1, len(train_losses) + 1))
    plt.figure()
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")
    plt.savefig("training_curves_simple.png")
    plt.close()

    print(
        f"\nDone! Best model saved to {model_path}. Curves saved to training_curves_simple.png."
    )
