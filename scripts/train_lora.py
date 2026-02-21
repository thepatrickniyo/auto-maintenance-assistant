"""
LoRA fine-tuning script for Car Maintenance LLM Assistant.
Run from project root: python scripts/train_lora.py
"""

import sys
from pathlib import Path
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))

from _project_root import PROJECT_ROOT

import json
import torch
from typing import Optional

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer

# Defaults (paths relative to project root)
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_DIR = str(PROJECT_ROOT / "data" / "training")
OUTPUT_DIR = str(PROJECT_ROOT / "car-maintenance-llm")
MAX_SEQ_LENGTH = 512
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2
WARMUP_RATIO = 0.03
SAVE_STRATEGY = "epoch"
LOGGING_STEPS = 5


def load_alpaca_dataset(data_dir: str) -> Dataset:
    """Load train.json (Alpaca format)."""
    train_path = Path(data_dir) / "train.json"
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}. Run: python scripts/prepare_training_data.py")
    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def formatting_prompt(example: dict) -> str:
    """Format one example for SFT (Alpaca style)."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")
    if inp:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


def main(
    model_id: str = MODEL_ID,
    data_dir: str = DATA_DIR,
    output_dir: str = OUTPUT_DIR,
    max_seq_length: int = MAX_SEQ_LENGTH,
    lora_r: int = LORA_R,
    lora_alpha: int = LORA_ALPHA,
    lora_dropout: float = LORA_DROPOUT,
    batch_size: int = BATCH_SIZE,
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION,
    learning_rate: float = LEARNING_RATE,
    num_epochs: int = NUM_EPOCHS,
    use_4bit: bool = True,
    save_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU found. Training will be slow.")

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_alpaca_dataset(data_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_strategy=SAVE_STRATEGY,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
    )

    def format_dataset(examples):
        texts = []
        for i in range(len(examples["instruction"])):
            ex = {
                "instruction": examples["instruction"][i],
                "input": examples.get("input", [""] * len(examples["instruction"]))[i],
                "output": examples["output"][i],
            }
            texts.append(formatting_prompt(ex))
        return {"text": texts}

    dataset = dataset.map(format_dataset, batched=True, remove_columns=dataset.column_names, desc="Formatting")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=False,
    )

    print("Starting training...")
    train_result = trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    if save_to_hub and hub_model_id:
        trainer.push_to_hub(hub_model_id)

    print(f"Model and tokenizer saved to {output_dir}")
    return train_result.metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--lora_r", type=int, default=LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRADIENT_ACCUMULATION)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--save_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    args = parser.parse_args()

    main(
        model_id=args.model_id,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        use_4bit=not args.no_4bit,
        save_to_hub=args.save_to_hub,
        hub_model_id=args.hub_model_id,
    )
