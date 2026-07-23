from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .src.pipeline import RAGPipeline
except ImportError:
    from src.pipeline import RAGPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieval + extractive QA chatbot using the fine-tuned DistilBERT reader."
    )
    parser.add_argument(
        "--config",
        default=Path(__file__).resolve().parent / "config.yaml",
        type=Path,
        help="Path to RAG config.yaml.",
    )
    parser.add_argument(
        "--data",
        nargs="+",
        type=Path,
        help="File or folder paths to index before starting chat.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the vector index from --data instead of loading an existing index.",
    )
    parser.add_argument(
        "--question",
        help="Ask one question and exit.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved context snippets with the answer.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = RAGPipeline.from_config(args.config)

    if args.data:
        pipeline.build_index(args.data, save=True)
    elif not args.rebuild:
        pipeline.load_index()

    if args.question:
        result = pipeline.answer(args.question)
        print(f"Đáp án: {result['answer']}")
        print(f"Điểm logits: {result['score']:.4f}")
        if args.show_context:
            for idx, item in enumerate(result["contexts"], start=1):
                print(f"\n[{idx}] {item['source']} | similarity={item['similarity']:.4f}")
                print(item["text"])
        return

    print(f"Bot RAG trích xuất đáp án | Device: {pipeline.device}")
    print("Gõ 'thoát', 'exit' hoặc 'quit' để dừng.")
    print("-" * 50)
    while True:
        question = input("Bạn hỏi: ").strip()
        if question.lower() in {"thoát", "exit", "quit"}:
            print("Bot: Tạm biệt!")
            break
        if not question:
            continue

        result = pipeline.answer(question)
        print(f"Đáp án: {result['answer']}")
        print(f"   [Điểm logits: {result['score']:.4f}]")
        if args.show_context:
            for idx, item in enumerate(result["contexts"], start=1):
                print(f"   Context {idx}: {item['source']} | similarity={item['similarity']:.4f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
