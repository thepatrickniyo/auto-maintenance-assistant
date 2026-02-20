# Auto-Maintenance Assistant

A domain-specific LLM assistant fine-tuned for car maintenance questions and answers. This project helps new car owners find information about basic car maintenance through natural language queries.

## Project Overview

This project fine-tunes a Large Language Model (LLM) to create a specialized assistant for automotive maintenance. The model understands user queries about car maintenance and provides accurate, helpful responses tailored for beginners and intermediate car owners.

## Dataset

### Current Dataset Status

The dataset contains Q&A pairs covering various car maintenance topics including:
- Oil changes and engine maintenance
- Tire care and rotation
- Brake systems
- Electrical systems (battery, alternator)
- Cooling systems
- Transmission maintenance
- Diagnostics and troubleshooting
- And more...

### Dataset Collection

#### Option 1: Use Pre-generated Dataset

Run the collection script to generate an initial dataset:

```bash
python collect_dataset.py
```

This will create:
- `data/auto_maintenance_dataset.json` - JSON format
- `data/auto_maintenance_dataset.csv` - CSV format

#### Option 2: Expand the Dataset

Add more Q&A pairs using the expansion script:

```bash
python expand_dataset.py
```

#### Option 3: Manual Addition

You can manually add Q&A pairs by editing the scripts or using the `AutoMaintenanceDatasetCollector` class:

```python
from collect_dataset import AutoMaintenanceDatasetCollector

collector = AutoMaintenanceDatasetCollector()
collector.add_qa_pair(
    question="Your question here?",
    answer="Your detailed answer here.",
    category="category_name",
    difficulty="beginner"  # or "intermediate" or "advanced"
)
collector.save_json()
```

### Dataset Sources

**Existing Datasets Available:**
1. **Hugging Face Datasets:**
   - `ananttripathiak/engine-maintenance-dataset` - Engine maintenance data (19,535 rows)
   - `SSS18/predictive-maintenance-engine-data` - Predictive maintenance data
   - Note: Most existing datasets are for predictive maintenance (sensor data), not Q&A pairs

2. **Potential Sources for Expansion:**
   - Car manufacturer owner's manuals and FAQs
   - Automotive repair websites (e.g., AutoZone, NAPA, CarParts.com)
   - YouTube automotive channels (transcribe Q&A sections)
   - Reddit r/cartalk, r/MechanicAdvice
   - Automotive forums
   - Repair manual databases

**Important:** When scraping or using external sources:
- Always respect robots.txt and terms of service
- Verify accuracy of information
- Cite sources appropriately
- Consider copyright and fair use

### Dataset Format

Each Q&A pair follows this structure:

```json
{
  "instruction": "How often should I change my engine oil?",
  "response": "Most vehicles require an oil change every 5,000 to 7,500 miles...",
  "category": "oil_change",
  "source": "generated_basic",
  "difficulty": "beginner"
}
```

### Preparing Dataset for Training

Convert the dataset to training format:

```bash
python prepare_training_data.py
```

This will:
- Format data for LLM training (Alpaca format by default)
- Split into train/validation/test sets (80/10/10)
- Save formatted datasets in `data/training/`

## Dataset Statistics

Run the analysis to see current statistics:

```bash
python prepare_training_data.py
```

This shows:
- Total Q&A pairs
- Distribution by category
- Distribution by difficulty level
- Average question/answer lengths

## Target Dataset Size

**Goal:** 1,000-5,000 high-quality Q&A pairs

**Current Status:** ~30+ pairs (expandable)

**Expansion Strategy:**
1. Start with basic maintenance topics (current)
2. Add intermediate topics (diagnostics, specific systems)
3. Add advanced topics (repairs, troubleshooting)
4. Include edge cases and safety-critical information
5. Cover different vehicle types (sedans, SUVs, trucks, hybrids, EVs)

## Next Steps

1. **Expand Dataset:**
   - Run `expand_dataset.py` to add more Q&A pairs
   - Manually add Q&A pairs from reliable sources
   - Consider web scraping (with permission) from automotive sites

2. **Prepare for Training:**
   - Run `prepare_training_data.py` to format data
   - Review and clean the dataset
   - Ensure diversity across categories and difficulty levels

3. **Fine-tune Model:**
   - Select a base model (Gemma, TinyLlama, etc.)
   - Implement LoRA fine-tuning
   - Train on Google Colab with free GPU

4. **Evaluate:**
   - Use BLEU, ROUGE scores
   - Qualitative testing
   - Compare base vs fine-tuned model

5. **Deploy:**
   - Create web interface (Gradio recommended)
   - Allow user interaction with fine-tuned model

## Project Structure

```
auto-maintenance-assistant/
├── collect_dataset.py          # Initial dataset collection
├── expand_dataset.py           # Dataset expansion utilities
├── prepare_training_data.py    # Format data for training
├── data/
│   ├── auto_maintenance_dataset.json
│   ├── auto_maintenance_dataset.csv
│   └── training/
│       ├── train.json
│       ├── val.json
│       └── test.json
└── README.md
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Quality Guidelines

When adding Q&A pairs, ensure:
- **Accuracy:** Information is correct and verified
- **Clarity:** Answers are clear and easy to understand
- **Completeness:** Answers address the full question
- **Safety:** Critical safety information is emphasized
- **Appropriateness:** Difficulty level matches target audience
- **Diversity:** Cover various topics and scenarios

## Contributing

To add more Q&A pairs:
1. Edit `expand_dataset.py` or `collect_dataset.py`
2. Add Q&A pairs following the format
3. Run the collection script
4. Verify quality and accuracy

## License

[Specify your license here]

## References

- Hugging Face Datasets: https://huggingface.co/datasets
- LoRA Fine-tuning: https://huggingface.co/docs/peft
- Alpaca Dataset Format: https://github.com/tatsu-lab/stanford_alpaca
