from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ..utils.paths import REPO_ROOT

PROJECT_SRC = REPO_ROOT / "src"
PROJECT_MODEL_SRC = PROJECT_SRC / "model"
for import_path in (PROJECT_SRC, PROJECT_MODEL_SRC):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

from config_model import Config
from loadmodel import CustomDistilBertQA
from vietnamese import has_vietnamese, normalize_text, segment_texts


DEFAULT_MODEL_DIR = REPO_ROOT / "outputs" / "checkpoints_vi" / "best_model"


def load_qa_model(model_dir: str | Path = DEFAULT_MODEL_DIR):
    model_dir = Path(model_dir)
    config = Config.from_yaml(model_dir / "config.yaml")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    force_cpu = bool(getattr(config, "force_cpu", False))
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    model = CustomDistilBertQA(config)
    checkpoint = torch.load(
        model_dir / "training_state.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, tokenizer, config, device


def prepare_text(question: str, context: str, config):
    question = normalize_text(question)
    context = normalize_text(context)

    examples = {
        getattr(config, "question_column", "question"): [question],
        getattr(config, "context_column", "context"): [context],
    }
    if getattr(config, "use_vietnamese_segmentation", False) and has_vietnamese(examples):
        question = segment_texts([question])[0]
        context = segment_texts([context])[0]

    return question, context


def predict_answer(
    question: str,
    context: str,
    model,
    tokenizer,
    config,
    device,
    max_answer_tokens: int = 30,
) -> dict:
    question, model_context = prepare_text(question, context, config)

    tokenized = tokenizer(
        question,
        model_context,
        max_length=getattr(config, "max_length", 384),
        stride=getattr(config, "doc_stride", 128),
        padding=getattr(config, "padding", "max_length"),
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long, device=device)
    attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long, device=device)

    with torch.no_grad():
        start_logits, end_logits = model(input_ids, attention_mask)

    start_logits = start_logits.cpu()
    end_logits = end_logits.cpu()

    best_answer = ""
    best_score = float("-inf")

    for feature_idx, offsets in enumerate(tokenized["offset_mapping"]):
        sequence_ids = tokenized.sequence_ids(feature_idx)
        valid_context_tokens = [
            idx
            for idx, sequence_id in enumerate(sequence_ids)
            if sequence_id == 1 and offsets[idx] is not None
        ]

        for start_idx in valid_context_tokens:
            max_end_idx = min(start_idx + max_answer_tokens, len(offsets))
            for end_idx in range(start_idx, max_end_idx):
                if sequence_ids[end_idx] != 1 or offsets[end_idx] is None:
                    continue

                start_char, _ = offsets[start_idx]
                _, end_char = offsets[end_idx]
                if end_char <= start_char:
                    continue

                score = start_logits[feature_idx, start_idx].item() + end_logits[feature_idx, end_idx].item()
                if score > best_score:
                    best_score = score
                    best_answer = model_context[start_char:end_char]

    return {
        "answer": best_answer.replace("_", " ").strip(),
        "score": best_score,
    }


class ExtractiveQAReader:
    def __init__(self, model, tokenizer, config, device, max_answer_tokens: int = 30):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.max_answer_tokens = max_answer_tokens

    @classmethod
    def from_model_dir(cls, model_dir: str | Path = DEFAULT_MODEL_DIR, max_answer_tokens: int = 30):
        model, tokenizer, config, device = load_qa_model(model_dir)
        return cls(model, tokenizer, config, device, max_answer_tokens=max_answer_tokens)

    @property
    def encoder(self):
        return self.model.distilbert

    def answer(self, question: str, context: str) -> dict:
        return predict_answer(
            question=question,
            context=context,
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            device=self.device,
            max_answer_tokens=self.max_answer_tokens,
        )
