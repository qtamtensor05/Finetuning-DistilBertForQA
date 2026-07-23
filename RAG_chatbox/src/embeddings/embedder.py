from __future__ import annotations

import numpy as np
import torch


class DistilBertEmbedder:
    """Mean-pool embeddings from the fine-tuned QA encoder."""

    def __init__(self, tokenizer, encoder, device, max_length: int = 256, batch_size: int = 16):
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.encoder.eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            tokenized = self.tokenizer(
                batch,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                )
            pooled = _mean_pool(outputs.last_hidden_state, tokenized["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(vectors)


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts
