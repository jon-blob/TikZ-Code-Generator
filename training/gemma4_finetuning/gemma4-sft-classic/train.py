from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import unsloth

import torch
from transformers import set_seed

from training_config import SFTConfig, configure_runtime
from data import DaTikZDataset
from model_loader import load_model
from trainer import train_sft


def main() -> None:
    configure_runtime()
    cfg = SFTConfig()
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")

    model, processor = load_model(cfg)
    train_dataset = DaTikZDataset(cfg, cfg.train_manifest, "train")
    val_dataset = DaTikZDataset(cfg, cfg.val_manifest, "val")
    train_sft(cfg, model, processor, train_dataset, val_dataset)


if __name__ == "__main__":
    main()