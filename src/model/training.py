import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from transformers import AutoTokenizer
import torch
import torch.nn as nn
from loadmodel import CustomLoraDistilBertQA

from config_model import Config
from src.data_loader import build_qa_datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from evalmodel import evaluate_loss


def save_checkpoint(
    checkpoint_dir,
    model,
    optimizer,
    tokenizer,
    config,
    epoch,
    train_loss,
    val_loss,
    best_val_loss,
):
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
            "config": config.__dict__,
        },
        checkpoint_dir / "training_state.pt",
    )
    tokenizer.save_pretrained(checkpoint_dir)
    config.to_yaml(checkpoint_dir / "config.yaml")


#Lựa chọn dữ liệu train
config = Config.from_yaml(profile="train_en")
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

datasets = build_qa_datasets(tokenizer, config)
#Gọi data train
train_data = datasets["train"]
val_data = datasets["validation"]

val_loader = DataLoader(
    val_data,
    batch_size=config.batch_size,
    shuffle=False,
)

train_loader = DataLoader(
    train_data,
    batch_size=config.batch_size,
    shuffle=True,
)

print(train_data)
print(train_data[0].keys())
print("input_ids shape:", train_data[0]["input_ids"].shape)
print("attention_mask shape:", train_data[0]["attention_mask"].shape)
print("start_positions:", train_data[0]["start_positions"])
print("end_positions:", train_data[0]["end_positions"])




device = "cuda" if torch.cuda.is_available() else "cpu"

model = CustomLoraDistilBertQA().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
loss_fn = nn.CrossEntropyLoss()

model.train()
output_dir = Path(config.output_dir)
best_val_loss = float("inf")

for epoch in range(config.epochs):
    total_loss = 0.0
    progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")

    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        optimizer.zero_grad()

        start_logits, end_logits = model(input_ids, attention_mask)

        start_loss = loss_fn(start_logits, start_positions)
        end_loss = loss_fn(end_logits, end_positions)
        loss = (start_loss + end_loss) / 2

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = total_loss / len(train_loader)
    val_loss = evaluate_loss(model, val_loader, loss_fn, device)
    print(
        f"Epoch {epoch + 1}: "
        f"train_loss={train_loss:.4f}, "
        f"val_loss={val_loss:.4f}"
    )

    epoch_number = epoch + 1
    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss

    if getattr(config, "save_best_model", True) and is_best:
        save_checkpoint(
            output_dir / "best_model",
            model,
            optimizer,
            tokenizer,
            config,
            epoch_number,
            train_loss,
            val_loss,
            best_val_loss,
        )
        print(f"Saved best model: {output_dir / 'best_model'}")
