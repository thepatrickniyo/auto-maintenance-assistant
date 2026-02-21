"""
Evaluate Car Maintenance LLM: BLEU, ROUGE, perplexity.
Run from project root: python scripts/evaluate.py --peft_model car-maintenance-llm
"""

import sys
from pathlib import Path
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))

from _project_root import PROJECT_ROOT

import json
import math
from typing import List, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

try:
    import nltk
    nltk.download("punkt", quiet=True)
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    HAS_BLEU = True
except Exception:
    HAS_BLEU = False

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except Exception:
    HAS_ROUGE = False


def load_test_data(data_dir: str = None) -> List[Dict]:
    """Load test.json or val.json (Alpaca format)."""
    if data_dir is None:
        data_dir = str(PROJECT_ROOT / "data" / "training")
    path = Path(data_dir) / "test.json"
    if not path.exists():
        path = Path(data_dir) / "val.json"
    if not path.exists():
        raise FileNotFoundError(f"No test.json or val.json in {data_dir}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_and_tokenizer(base_model_id: str, peft_model_path: Optional[str] = None, device: str = None):
    """Load base or base+adapter; return model, tokenizer."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        peft_model_path if peft_model_path else base_model_id,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    if peft_model_path:
        model = PeftModel.from_pretrained(model, peft_model_path)
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, instruction: str, input_text: str = "", max_new_tokens: int = 256, do_sample: bool = False, temperature: float = 0.1) -> str:
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    if "### Response:" in full:
        response = full.split("### Response:")[-1].strip()
    else:
        response = full[len(prompt):].strip()
    return response.split("\n\n")[0].strip() if response else ""


def compute_bleu(references: List[List[str]], hypotheses: List[str]) -> float:
    if not HAS_BLEU:
        return float("nan")
    smooth = SmoothingFunction()
    scores = [sentence_bleu([r.split() for r in ref], hyp.split(), smoothing_function=smooth.method1) for ref, hyp in zip(references, hypotheses)]
    return sum(scores) / len(scores) if scores else 0.0


def compute_rouge(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    if not HAS_ROUGE:
        return {"rouge1": float("nan"), "rouge2": float("nan"), "rougeL": float("nan")}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for ref, hyp in zip(references, hypotheses):
        s = scorer.score(ref, hyp)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    n = len(r1)
    return {"rouge1": sum(r1) / n if n else 0, "rouge2": sum(r2) / n if n else 0, "rougeL": sum(rl) / n if n else 0}


def compute_perplexity(model, tokenizer, texts: List[str], max_length: int = 512) -> float:
    total_loss, count = 0.0, 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length", return_attention_mask=True)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        with torch.no_grad():
            total_loss += model(**inputs, labels=inputs["input_ids"]).loss.item()
        count += 1
    return math.exp(total_loss / count) if count else float("nan")


def run_evaluation(base_model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", peft_model_path: Optional[str] = None,
                  data_dir: str = None, max_examples: Optional[int] = 20, max_new_tokens: int = 256) -> Dict:
    if data_dir is None:
        data_dir = str(PROJECT_ROOT / "data" / "training")
    test_data = load_test_data(data_dir)
    if max_examples:
        test_data = test_data[:max_examples]
    model, tokenizer = load_model_and_tokenizer(base_model_id, peft_model_path)
    references = [item["output"] for item in test_data]
    hypotheses = [generate_response(model, tokenizer, item["instruction"], item.get("input", ""), max_new_tokens) for item in test_data]
    rouge = compute_rouge(references, hypotheses)
    ppl_texts = [f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}" for item in test_data]
    return {
        "bleu": compute_bleu([[r] for r in references], hypotheses),
        "rouge1": rouge["rouge1"], "rouge2": rouge["rouge2"], "rougeL": rouge["rougeL"],
        "perplexity": compute_perplexity(model, tokenizer, ppl_texts, 512),
        "num_examples": len(test_data),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--peft_model", type=str, default=None, help="Path to adapter, e.g. car-maintenance-llm")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=20)
    args = parser.parse_args()

    peft_path = args.peft_model
    if peft_path and not Path(peft_path).is_absolute():
        peft_path = str(PROJECT_ROOT / peft_path)

    print("Evaluating base model...")
    base_metrics = run_evaluation(args.base_model, None, args.data_dir, args.max_examples)
    print("Base:", base_metrics)

    if peft_path and Path(peft_path).exists():
        print("Evaluating fine-tuned model...")
        tuned_metrics = run_evaluation(args.base_model, peft_path, args.data_dir, args.max_examples)
        print("Fine-tuned:", tuned_metrics)
        print("\nComparison:")
        for k in base_metrics:
            if k == "num_examples":
                continue
            b, t = base_metrics[k], tuned_metrics.get(k, float("nan"))
            diff = t - b if isinstance(b, (int, float)) and not math.isnan(b) else "N/A"
            print(f"  {k}: base={b} tuned={t} diff={diff}")
    else:
        print("No --peft_model path provided; only base model evaluated.")


if __name__ == "__main__":
    main()
