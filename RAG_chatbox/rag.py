import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs" / "checkpoints_vi" / "best_model"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "model"))

from config_model import Config
from loadmodel import CustomDistilBertQA
from vietnamese import has_vietnamese, normalize_text, segment_texts


def load_qa_model(model_dir):
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


def prepare_text(question, context, config):
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


def predict_answer(question, context, model, tokenizer, config, device, max_answer_tokens=30):
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




model, tokenizer, config, device = load_qa_model(MODEL_DIR)

context = """
Công ty X được thành lập vào năm 2010 bởi ông Nguyễn Văn A.
Sản phẩm chủ lực của công ty X là phần mềm quản lý nhân sự HR-Pro.
Trụ sở chính của công ty X nằm tại Quận 1, Thành phố Hồ Chí Minh.
"""

print(f"Bot Trích xuất Đáp án Device: {device}")
print("-" * 50)

while True:
    user_question = input("Bạn hỏi: ")
    if user_question.lower() in ["thoát", "exit", "quit"]:
        print("Bot: Tạm biệt!")
        break

    result = predict_answer(user_question, context, model, tokenizer, config, device)

    print(f"Đáp án: {result['answer']}")
    print(f"   [Điểm logits: {result['score']:.4f}]")
    print("-" * 50)
