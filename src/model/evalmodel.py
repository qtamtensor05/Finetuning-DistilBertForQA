import torch

def evaluate_loss(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            start_logits, end_logits = model(input_ids, attention_mask)

            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

            total_loss += loss.item()

    model.train()
    return total_loss / len(val_loader)