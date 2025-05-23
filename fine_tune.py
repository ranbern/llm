import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset


def train_model():
    print("Starting model training...")
    # Step 1: Load the transcriptions
    with open("transcripts.json", "r") as f:
        transcriptions = json.load(f)

    # Combine all transcriptions into a single list of text samples
    text_data = [text for text in transcriptions.values()]
    model_name = "distilgpt2"

    # Load tokenizer and properly configure pad token
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    dataset = Dataset.from_dict({"text": text_data})

    def tokenize_function(examples):
        # Explicitly create attention masks during tokenization
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_attention_mask=True  # Explicitly request attention masks
        )
        # For causal LM, labels are same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Load the model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Step 4: Fine-tune the model
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        # Add additional settings for better training stability
        save_strategy="epoch",
        learning_rate=5e-5,
        warmup_steps=500,
        weight_decay=0.01,
    )

    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    # Step 5: Save the fine-tuned model
    print("Saving model...")
    trainer.save_model("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("Training complete. Model saved to ./fine_tuned_model")


if __name__ == "__main__":
    train_model()