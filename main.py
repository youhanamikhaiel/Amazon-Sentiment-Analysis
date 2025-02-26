import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

from utils import preprocess_function, tokenize_function, compute_metrics

# Check if mps is available
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

dataset = load_dataset("yassiracharki/Amazon_Reviews_for_Sentiment_Analysis_fine_grained_5_classes")


# Load tokenizer and model
model_checkpoint = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=5)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["query", "value"]
)
model = get_peft_model(model, lora_config)
model.to(device)

# Count total parameters and trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
trainable_ratio = (trainable_params / total_params) * 100

print(f"\nTotal Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,} ({trainable_ratio:.2f}%)")
print(f"Non-Trainable Parameters: {total_params - trainable_params:,}\n")

# Tokenize dataset
dataset = dataset.map(preprocess_function, batched=True, remove_columns=["class_index", "review_title", "review_text"])

train_dataset = dataset["train"].shuffle(seed=42).select(range(1000000))  # Subset for faster training
test_dataset = dataset["test"].shuffle(seed=42).select(range(50000))

# Tokenize dataset
train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_MiniLM_amazon",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=128,
    gradient_accumulation_steps=10,
    num_train_epochs=3,
    learning_rate=8e-4,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
    push_to_hub=False,
    bf16=True if device == "mps" else False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()

# Save the fine-tuned model and tokenizer
trainer.save_model("./finetuned_MiniLM_amazon")
tokenizer.save_pretrained("./finetuned_MiniLM_amazon")

print("Fine-tuning complete! Model saved to './finetuned_MiniLM_amazon'.")
