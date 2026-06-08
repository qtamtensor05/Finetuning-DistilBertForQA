from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src" / "model"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from config_model import Config
from experiments.phobert.modeling import PhoBertLoraQA
from evalmodel import evaluate_loss, plot_loss_curves
from src.data_loader import build_qa_datasets


CONFIG_PATH = ROOT / "experiments" / "phobert" / "config.yaml"


def save_history(history, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def save_checkpoint(checkpoint_dir, model, optimizer, tokenizer, config, epoch, train_loss, val_loss, best_val_loss, history):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "history": history,
            "config": config.__dict__,
        },
        checkpoint_dir / "training_state.pt",
    )
    tokenizer.save_pretrained(checkpoint_dir)
    config.to_yaml(checkpoint_dir / "config.yaml")
    save_history(history, checkpoint_dir)


class PhoBertTrainer:
    def __init__(self, profile_name):
        self.profile_name = profile_name
        self.config = Config.from_yaml(path=CONFIG_PATH, profile=profile_name)
        self.device = self._resolve_device()
        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.optimizer = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.best_val_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": []}

    @property
    def output_dir(self):
        return Path(self.config.output_dir)

    def _resolve_device(self):
        if getattr(self.config, "force_cpu", False):
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def setup_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError(
                f"{self.config.model_name} tokenizer does not provide fast offsets. "
                "QA span training needs return_offsets_mapping."
            )
        return self.tokenizer

    def setup_dataloaders(self):
        if self.tokenizer is None:
            self.setup_tokenizer()

        datasets = build_qa_datasets(self.tokenizer, self.config)
        self.train_loader = DataLoader(datasets["train"], batch_size=self.config.batch_size, shuffle=True)
        self.val_loader = DataLoader(datasets["validation"], batch_size=self.config.batch_size, shuffle=False)
        return self.train_loader, self.val_loader

    def setup_model(self):
        self.model = PhoBertLoraQA(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.model.train()
        return self.model

    def setup(self):
        self.setup_tokenizer()
        self.setup_dataloaders()
        self.setup_model()

    def train_one_epoch(self, epoch_index):
        total_loss = 0.0
        progress = tqdm(self.train_loader, desc=f"{self.profile_name} epoch {epoch_index + 1}/{self.config.epochs}")

        for batch in progress:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            start_positions = batch["start_positions"].to(self.device)
            end_positions = batch["end_positions"].to(self.device)

            self.optimizer.zero_grad()
            start_logits, end_logits = self.model(input_ids, attention_mask)
            loss = (
                self.loss_fn(start_logits, start_positions)
                + self.loss_fn(end_logits, end_positions)
            ) / 2
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader)

    def evaluate(self):
        return evaluate_loss(self.model, self.val_loader, self.loss_fn, self.device)

    def save_best_if_needed(self, epoch_number, train_loss, val_loss):
        if val_loss >= self.best_val_loss:
            return

        self.best_val_loss = val_loss
        if getattr(self.config, "save_best_model", True):
            best_dir = self.output_dir / "best_model"
            save_checkpoint(
                best_dir,
                self.model,
                self.optimizer,
                self.tokenizer,
                self.config,
                epoch_number,
                train_loss,
                val_loss,
                self.best_val_loss,
                self.history,
            )
            print(f"Saved best model: {best_dir}")

    def train(self):
        if self.model is None:
            self.setup()

        print(f"Training PhoBERT profile: {self.profile_name}")
        print(f"Model: {self.config.model_name}")
        print(f"Train file: {self.config.train_file}")
        print(f"Validation file: {self.config.validation_file}")
        print(f"Output dir: {self.output_dir}")

        for epoch in range(self.config.epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.evaluate()
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            save_history(self.history, self.output_dir)
            print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            self.save_best_if_needed(epoch + 1, train_loss, val_loss)

        plot_loss_curves(self.history["train_loss"], self.history["val_loss"], self.output_dir)
        plot_loss_curves(self.history["train_loss"], self.history["val_loss"], self.output_dir / "best_model")
        return {
            "profile": self.profile_name,
            "model_name": self.config.model_name,
            "output_dir": str(self.output_dir),
            "best_model_dir": str(self.output_dir / "best_model"),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    profiles = Config.available_profiles(path=CONFIG_PATH)
    parser.add_argument("--profile", default=None, choices=profiles)
    parser.add_argument("--profiles", nargs="+", default=None, choices=profiles)
    return parser.parse_args()


def resolve_profiles(args):
    if args.profile and args.profiles:
        raise ValueError("Chi dung mot trong hai tuy chon: --profile hoac --profiles.")
    if args.profile:
        return [args.profile]
    if args.profiles:
        return args.profiles
    return Config.default_pipeline_profiles(path=CONFIG_PATH)


def main():
    args = parse_args()
    records = []
    for profile in resolve_profiles(args):
        trainer = PhoBertTrainer(profile)
        records.append(trainer.train())

    if records:
        out_dir = Path(records[-1]["output_dir"])
        with open(out_dir / "phobert_training_runs.json", "w", encoding="utf-8") as f:
            json.dump({"runs": records}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
