"""QLoRA fine-tuning for the Python code bug explainer."""

import argparse
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer


def format_example(example: dict) -> str:
    return (
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        "### Code:\n"
        f"```python\n{example['input']}\n```\n\n"
        "### Response:\n"
        f"{example['output']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/debugging.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/code-debugger"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}. Run prepare_dataset.py first.")

    dataset = load_dataset("json", data_files=str(args.data), split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map="auto",
    )
    model.config.use_cache = False

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=format_example,
        max_seq_length=args.max_seq_length,
        packing=True,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
        args=TrainingArguments(
            output_dir=str(args.output),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"Saved adapter and tokenizer to {args.output}")


if __name__ == "__main__":
    main()
