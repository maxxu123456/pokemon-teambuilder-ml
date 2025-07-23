import argparse
import math
from pathlib import Path
import os

# for apple silicon computers
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn


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


@torch.no_grad()
def generate_team(model, opponent_team, vocabs, device, max_len=6):
    model.eval()
    species_stoi, species_itos = vocabs
    pad_idx = species_stoi["<pad>"]
    bos_idx = species_stoi["<bos>"]
    eos_idx = species_stoi["<eos>"]
    unk_idx = species_stoi["<unk>"]

    src_species = [species_stoi.get(p, unk_idx) for p in opponent_team]
    src_tensor = torch.tensor(src_species).unsqueeze(0).to(device)

    generated_ids = [bos_idx]

    for _ in range(max_len):
        tgt_tensor = torch.tensor(generated_ids).unsqueeze(0).to(device)
        logits = model(src_tensor, tgt_tensor, pad_idx, pad_idx)
        next_token_id = logits[0, -1, :].argmax().item()

        if next_token_id == eos_idx:
            break

        generated_ids.append(next_token_id)

    return [species_itos[idx] for idx in generated_ids[1:]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a Pokémon counter-team using a trained model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="data/run5-100pages-allgens/model.pth",
        help="Path to the trained model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--team",
        type=str,
        nargs="+",
        required=True,
        help="The opponent's team, separated by spaces (e.g., --team Snorlax Tauros Chansey).",
    )
    args = parser.parse_args()

    if len(args.team) > 6:
        print(f"Warning: Input team has {len(args.team)} members. Using the first 6.")
        args.team = args.team[:6]

    model_file = Path(args.model_path)
    if not model_file.is_file():
        print(f"Error: Model file not found at '{args.model_path}'")
        exit()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)

    species_stoi = checkpoint["species_stoi"]
    species_itos = checkpoint["species_itos"]

    hp = checkpoint.get(
        "hyperparameters",
        {
            "d_model": 256,
            "nhead": 8,
            "num_encoder_layers": 4,
            "num_decoder_layers": 4,
            "dim_feedforward": 1024,
            "dropout": 0.1,
        },
    )

    model = ConditionalTeamGenerator(
        num_species=len(species_stoi),
        d_model=hp["d_model"],
        nhead=hp["nhead"],
        num_encoder_layers=hp["num_encoder_layers"],
        num_decoder_layers=hp["num_decoder_layers"],
        dim_feedforward=hp["dim_feedforward"],
        dropout=hp["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("\nGenerating counter-team...")
    vocabs = (species_stoi, species_itos)
    generated_team = generate_team(model, args.team, vocabs, device)

    print("\n" + "=" * 40)
    print(f"Opponent Team: {', '.join(args.team)}")
    print(f"Generated Counter-Team: {', '.join(generated_team)}")
    print("=" * 40 + "\n")
