class NoLLMClient:
    def generate(self, *_args, **_kwargs):
        raise NotImplementedError("This RAG pipeline uses the fine-tuned extractive QA model, not an LLM.")
