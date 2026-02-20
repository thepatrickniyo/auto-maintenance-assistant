# Quick Start Guide

## Getting Started with Dataset Collection

### Step 1: Generate Initial Dataset

```bash
python collect_dataset.py
```

This creates:
- `data/auto_maintenance_dataset.json` (20 Q&A pairs)
- `data/auto_maintenance_dataset.csv` (20 Q&A pairs)

### Step 2: Expand Dataset

```bash
python expand_dataset.py
```

This adds 15 more Q&A pairs (total: 35 pairs)

### Step 3: Prepare for Training

```bash
python prepare_training_data.py
```

This:
- Analyzes the dataset
- Formats data for LLM training (Alpaca format)
- Splits into train/val/test sets (80/10/10)
- Saves to `data/training/`

### Step 4: Review Dataset

Check the generated files:
- `data/auto_maintenance_dataset.json` - Full dataset
- `data/training/train.json` - Training set
- `data/training/val.json` - Validation set
- `data/training/test.json` - Test set

## Dataset Expansion Strategies

### Option A: Manual Addition via Code

Edit `expand_dataset.py` and add Q&A pairs to the `scrape_manual_qa_pairs()` function:

```python
{
    "instruction": "Your question?",
    "response": "Your detailed answer.",
    "category": "category_name",
    "difficulty": "beginner"  # or "intermediate" or "advanced"
}
```

### Option B: Programmatic Addition

```python
from collect_dataset import AutoMaintenanceDatasetCollector

collector = AutoMaintenanceDatasetCollector()
collector.load_from_json("data/auto_maintenance_dataset.json")

collector.add_qa_pair(
    question="How do I check my brake fluid?",
    answer="Check brake fluid by locating the reservoir...",
    category="brakes",
    difficulty="beginner"
)

collector.save_json()
```

### Option C: Import from CSV/JSON

If you have external data, convert it to the format and load:

```python
import json
from collect_dataset import AutoMaintenanceDatasetCollector

collector = AutoMaintenanceDatasetCollector()
collector.load_from_json("data/auto_maintenance_dataset.json")

# Add external data
external_data = [
    {"instruction": "Q?", "response": "A.", "category": "cat", "difficulty": "beginner"}
]

for item in external_data:
    collector.add_qa_pair(**item)

collector.save_json()
```

## Dataset Quality Checklist

Before training, ensure:
- ✅ At least 1,000 Q&A pairs (aim for 1,000-5,000)
- ✅ Diverse categories covered
- ✅ Mix of difficulty levels
- ✅ Answers are accurate and verified
- ✅ Answers are clear and complete
- ✅ Safety-critical information emphasized

## Next Steps

1. **Expand to 1,000+ pairs:**
   - Add Q&A from car manuals
   - Scrape automotive websites (with permission)
   - Use automotive forums/Reddit
   - Transcribe YouTube Q&A sections

2. **Fine-tune Model:**
   - Select base model (Gemma, TinyLlama)
   - Implement LoRA fine-tuning
   - Train on Google Colab

3. **Evaluate:**
   - BLEU/ROUGE scores
   - Qualitative testing
   - Compare base vs fine-tuned

4. **Deploy:**
   - Create Gradio interface
   - Allow user interaction

## Current Dataset Status

- **Total Q&A pairs:** 35
- **Categories:** 10+ (oil_change, brakes, tires, engine, etc.)
- **Difficulty levels:** Beginner, Intermediate, Advanced
- **Format:** Ready for Alpaca-style fine-tuning

## Resources for Expansion

1. **Car Manufacturer Sites:**
   - Toyota, Honda, Ford owner's manuals
   - FAQ sections

2. **Automotive Websites:**
   - AutoZone.com
   - NAPAonline.com
   - CarParts.com
   - RepairPal.com

3. **Community Sources:**
   - Reddit: r/cartalk, r/MechanicAdvice
   - Automotive forums
   - YouTube automotive channels

4. **Datasets:**
   - Hugging Face: Search "car maintenance", "automotive"
   - Kaggle: Vehicle maintenance datasets

**Remember:** Always verify accuracy and respect copyright/terms of service when using external sources.
