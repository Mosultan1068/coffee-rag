"""
LoRA fine-tuning script — coffee knowledge

Fine-tunes distilgpt2 (a small, CPU-friendly model) on the coffee
Q&A training data using LoRA, so only a small set of extra weights
get trained rather than the whole model.

This is a proof-of-concept fine-tune: the goal is to demonstrate the
mechanism and produce a model that can be compared against the RAG
system's answers, not to produce a highly fluent, production-quality
model. distilgpt2 is a small, older model — expect rougher answers
than Claude/GPT produce, and that's an expected, honest trade-off
given the CPU-only, small-model, small-dataset constraints.
"""

import json
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL = "distilgpt2"
TRAINING_FILE = "coffee_training_data.jsonl"
OUTPUT_DIR = "./coffee_lora_model"


def load_training_examples(path: str) -> list[str]:
    """
    Reads the JSONL file (same format used for the OpenAI attempt) and
    converts each example into a single plain-text string, since
    distilgpt2 is a simple causal language model, not a chat model
    with built-in message-role formatting.
    """
    examples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record["messages"]
            user_msg = next(m["content"] for m in messages if m["role"] == "user")
            assistant_msg = next(m["content"] for m in messages if m["role"] == "assistant")
            text = f"Question: {user_msg}\nAnswer: {assistant_msg}"
            examples.append(text)
    return examples


def main():
    print("Loading training data...")
    texts = load_training_examples(TRAINING_FILE)
    print(f"Loaded {len(texts)} training examples.")

    print(f"\nLoading base model: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token  # distilgpt2 has no pad token by default
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    # Configure LoRA: only a small set of extra weights get trained,
    # the base model's original weights stay frozen.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                 # rank of the LoRA update matrices — small and fast
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn"],  # the attention layers distilgpt2 exposes for LoRA
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # shows how few parameters are actually being trained

    # Tokenize the training text
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_dataset = dataset.map(tokenize, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,        # small dataset, more passes over it helps it stick
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="no",         # we save manually at the end instead
        report_to=[],               # disable external logging integrations
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("\nStarting training (this will take a few minutes on CPU)...")
    trainer.train()

    print(f"\nSaving fine-tuned model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()