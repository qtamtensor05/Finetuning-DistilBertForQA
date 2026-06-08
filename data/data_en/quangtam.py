from datasets import load_dataset
dataset = load_dataset("taidng/UIT-ViQuAD2.0")
print(len(dataset["train"]))
print(len(dataset["validation"]))
print(len(dataset["test"]))