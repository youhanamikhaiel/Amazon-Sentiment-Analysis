import evaluate
import numpy as np

# Define evaluation metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def preprocess_function(examples):
    text = [(title if title else "") + " " + review for title, review in zip(examples["review_title"], examples["review_text"])]
    # Convert to 0-based indexing (0 to 4)
    labels = [label - 1 for label in examples["class_index"]]
    return {"text": text, "labels": labels}


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    }