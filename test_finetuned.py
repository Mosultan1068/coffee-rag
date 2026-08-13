"""
Inference script — load the fine-tuned (LoRA) coffee model and
generate an answer, so it can be compared against the RAG system's
answer to the same question.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "distilgpt2"
ADAPTER_PATH = "./coffee_lora_model"


def load_finetuned_model():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    return model, tokenizer


def generate_answer(question: str, model, tokenizer, max_new_tokens: int = 60) -> str:
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Return just the generated answer portion, after the prompt
    return full_text[len(prompt):].strip()


if __name__ == "__main__":
    print("Loading fine-tuned model...")
    model, tokenizer = load_finetuned_model()

    test_questions = [
        "What happens during light roasting?",
        "How does dark roasting affect flavor?",
        "What's the difference between Arabica and Robusta beans?",
    ]

    for question in test_questions:
        print(f"\nQuestion: {question}")
        answer = generate_answer(question, model, tokenizer)
        print(f"Fine-tuned model answer: {answer}")