"""
Prepare dataset for LLM fine-tuning.
Converts Q&A pairs into instruction-following format suitable for training.
Includes data cleaning: HTML removal, deduplication, relevance filtering.
"""

import json
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Set
import random

from _project_root import PROJECT_ROOT

# Keywords that suggest car-maintenance relevance (for filtering)
MAINTENANCE_KEYWORDS = {
    "car", "vehicle", "engine", "oil", "brake", "tire", "battery", "transmission",
    "coolant", "filter", "spark", "alternator", "rotor", "pad", "fluid", "maintenance",
    "overheat", "start", "clicking", "noise", "vibrat", "light", "check engine",
    "change", "replace", "how often", "when should", "what does", "why is", "diagnos",
    "fix", "repair", "service", "schedule", "psi", "mile", "km", "diy", "symptom"
}


def strip_html(text: str) -> str:
    """Remove HTML tags and decode common entities."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str) -> str:
    """Normalize whitespace and strip."""
    if not text or not isinstance(text, str):
        return ""
    return " ".join(text.split()).strip()


def is_maintenance_related(instruction: str, response: str) -> bool:
    """Heuristic: keep only examples that look like car maintenance."""
    combined = (instruction + " " + response).lower()
    return any(kw in combined for kw in MAINTENANCE_KEYWORDS)


def remove_duplicates(dataset: List[Dict], key_fields: tuple = ("instruction", "response")) -> List[Dict]:
    """Remove duplicate examples based on instruction (and optionally response)."""
    seen: Set[str] = set()
    unique = []
    for item in dataset:
        raw = item.get("instruction", "") + "|" + item.get("response", "")
        key = hashlib.md5(raw.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def clean_dataset(dataset: List[Dict], strip_html_text: bool = True,
                  remove_dupes: bool = True, filter_non_maintenance: bool = False) -> List[Dict]:
    """
    Clean dataset: strip HTML, normalize, remove duplicates, optionally filter by relevance.
    """
    cleaned = []
    for item in dataset:
        instr = item.get("instruction", "")
        resp = item.get("response", "")
        if strip_html_text:
            instr = strip_html(instr)
            resp = strip_html(resp)
        instr = normalize_text(instr)
        resp = normalize_text(resp)
        if not instr or not resp:
            continue
        new_item = {**item, "instruction": instr, "response": resp}
        if filter_non_maintenance and not is_maintenance_related(instr, resp):
            continue
        cleaned.append(new_item)
    if remove_dupes:
        cleaned = remove_duplicates(cleaned)
    return cleaned


def format_for_training(dataset: List[Dict], format_type: str = "alpaca") -> List[Dict]:
    """
    Format dataset for training in various formats.

    Args:
        dataset: List of Q&A dictionaries
        format_type: "alpaca", "chatml", or "simple"

    Returns:
        Formatted dataset ready for training
    """
    formatted = []

    for item in dataset:
        instruction = item["instruction"]
        response = item["response"]

        if format_type == "alpaca":
            formatted_item = {
                "instruction": instruction,
                "input": "",
                "output": response
            }
        elif format_type == "chatml":
            formatted_item = {
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response}
                ]
            }
        elif format_type == "simple":
            formatted_item = {
                "text": f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            }
        else:
            raise ValueError(f"Unknown format type: {format_type}")

        if "category" in item:
            formatted_item["category"] = item["category"]
        if "difficulty" in item:
            formatted_item["difficulty"] = item["difficulty"]

        formatted.append(formatted_item)

    return formatted


def split_dataset(dataset: List[Dict], train_ratio: float = 0.8,
                  val_ratio: float = 0.1, test_ratio: float = 0.1):
    """Split dataset into train/validation/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1.0"
    shuffled = dataset.copy()
    random.seed(42)
    random.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def prepare_for_huggingface(dataset_path: str, output_dir: str = None,
                            format_type: str = "alpaca", split: bool = True,
                            strip_html_text: bool = True, remove_dupes: bool = True,
                            filter_non_maintenance: bool = False):
    """Prepare dataset for Hugging Face fine-tuning."""
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "data" / "training")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} Q&A pairs")
    dataset = clean_dataset(
        dataset,
        strip_html_text=strip_html_text,
        remove_dupes=remove_dupes,
        filter_non_maintenance=filter_non_maintenance,
    )
    print(f"After cleaning: {len(dataset)} Q&A pairs")

    formatted = format_for_training(dataset, format_type)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if split:
        train, val, test = split_dataset(formatted)
        print(f"Split dataset: Train {len(train)}, Val {len(val)}, Test {len(test)}")
        for name, data in [("train", train), ("val", val), ("test", test)]:
            with open(output_path / f"{name}.json", 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved to {output_path}/")
    else:
        with open(output_path / "formatted_dataset.json", 'w', encoding='utf-8') as f:
            json.dump(formatted, f, indent=2, ensure_ascii=False)

    return formatted


def analyze_dataset(dataset_path: str):
    """Analyze dataset statistics."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"\nDataset Analysis: {len(dataset)} Q&A pairs")
    categories = {}
    difficulties = {}
    sources = {}
    for item in dataset:
        cat = item.get("category", "unknown")
        diff = item.get("difficulty", "unknown")
        src = item.get("source", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1
        sources[src] = sources.get(src, 0) + 1
    print("By Category:", dict(sorted(categories.items(), key=lambda x: -x[1])))
    print("By Difficulty:", difficulties)
    print("By Source:", sources)
    avg_q = sum(len(item["instruction"]) for item in dataset) / len(dataset)
    avg_a = sum(len(item["response"]) for item in dataset) / len(dataset)
    print(f"Avg question length: {avg_q:.1f} chars, Avg answer length: {avg_a:.1f} chars")


if __name__ == "__main__":
    dataset_path = PROJECT_ROOT / "data" / "auto_maintenance_dataset.json"

    if not dataset_path.exists():
        print(f"Error: {dataset_path} not found. Run: python scripts/collect_dataset.py")
        exit(1)

    analyze_dataset(str(dataset_path))
    print("\n" + "=" * 50)
    print("Preparing dataset for training...")
    prepare_for_huggingface(str(dataset_path), format_type="alpaca", split=True)
    print("\nDataset preparation complete!")
    print("Next step: train in Colab (notebooks/) or run: python scripts/train_lora.py")
