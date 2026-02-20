"""
Prepare dataset for LLM fine-tuning.
Converts Q&A pairs into instruction-following format suitable for training.
"""

import json
from pathlib import Path
from typing import List, Dict
import random

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
            # Alpaca format (used by many instruction-following models)
            formatted_item = {
                "instruction": instruction,
                "input": "",  # Empty input for Q&A format
                "output": response
            }
        
        elif format_type == "chatml":
            # ChatML format (used by some models)
            formatted_item = {
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response}
                ]
            }
        
        elif format_type == "simple":
            # Simple instruction-response format
            formatted_item = {
                "text": f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            }
        
        else:
            raise ValueError(f"Unknown format type: {format_type}")
        
        # Add metadata if present
        if "category" in item:
            formatted_item["category"] = item["category"]
        if "difficulty" in item:
            formatted_item["difficulty"] = item["difficulty"]
        
        formatted.append(formatted_item)
    
    return formatted


def split_dataset(dataset: List[Dict], train_ratio: float = 0.8, 
                  val_ratio: float = 0.1, test_ratio: float = 0.1):
    """
    Split dataset into train/validation/test sets.
    
    Args:
        dataset: Full dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
    
    Returns:
        Tuple of (train, val, test) datasets
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1.0"
    
    # Shuffle dataset
    shuffled = dataset.copy()
    random.seed(42)  # For reproducibility
    random.shuffle(shuffled)
    
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]
    
    return train, val, test


def prepare_for_huggingface(dataset_path: str, output_dir: str = "data/training",
                            format_type: str = "alpaca", split: bool = True):
    """
    Prepare dataset for Hugging Face fine-tuning.
    
    Args:
        dataset_path: Path to JSON dataset file
        output_dir: Directory to save formatted datasets
        format_type: Format to use ("alpaca", "chatml", or "simple")
        split: Whether to split into train/val/test
    """
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} Q&A pairs")
    
    # Format dataset
    formatted = format_for_training(dataset, format_type)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if split:
        # Split dataset
        train, val, test = split_dataset(formatted)
        
        print(f"Split dataset:")
        print(f"  Train: {len(train)} examples")
        print(f"  Val: {len(val)} examples")
        print(f"  Test: {len(test)} examples")
        
        # Save splits
        with open(output_path / "train.json", 'w', encoding='utf-8') as f:
            json.dump(train, f, indent=2, ensure_ascii=False)
        
        with open(output_path / "val.json", 'w', encoding='utf-8') as f:
            json.dump(val, f, indent=2, ensure_ascii=False)
        
        with open(output_path / "test.json", 'w', encoding='utf-8') as f:
            json.dump(test, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved formatted datasets to {output_path}/")
        print(f"  - train.json")
        print(f"  - val.json")
        print(f"  - test.json")
    else:
        # Save full dataset
        with open(output_path / "formatted_dataset.json", 'w', encoding='utf-8') as f:
            json.dump(formatted, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved formatted dataset to {output_path}/formatted_dataset.json")
    
    return formatted


def analyze_dataset(dataset_path: str):
    """Analyze dataset statistics."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"\nDataset Analysis:")
    print(f"Total Q&A pairs: {len(dataset)}")
    
    # Count by category
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
    
    print(f"\nBy Category:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    
    print(f"\nBy Difficulty:")
    for diff, count in sorted(difficulties.items()):
        print(f"  {diff}: {count}")
    
    print(f"\nBy Source:")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")
    
    # Average lengths
    avg_q_len = sum(len(item["instruction"]) for item in dataset) / len(dataset)
    avg_a_len = sum(len(item["response"]) for item in dataset) / len(dataset)
    
    print(f"\nAverage Question Length: {avg_q_len:.1f} characters")
    print(f"Average Answer Length: {avg_a_len:.1f} characters")


if __name__ == "__main__":
    dataset_path = "data/auto_maintenance_dataset.json"
    
    if not Path(dataset_path).exists():
        print(f"Error: {dataset_path} not found. Run collect_dataset.py first.")
        exit(1)
    
    # Analyze dataset
    analyze_dataset(dataset_path)
    
    # Prepare for training
    print("\n" + "="*50)
    print("Preparing dataset for training...")
    prepare_for_huggingface(dataset_path, format_type="alpaca", split=True)
    
    print("\nDataset preparation complete!")
    print("You can now use the formatted datasets for fine-tuning.")
