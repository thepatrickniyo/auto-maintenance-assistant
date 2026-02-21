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
python scripts/collect_dataset.py
```

This will create:
- `data/auto_maintenance_dataset.json` - JSON format
- `data/auto_maintenance_dataset.csv` - CSV format

#### Option 2: Expand the Dataset

Add more Q&A pairs using the expansion script:

```bash
python scripts/expand_dataset.py
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
python scripts/prepare_training_data.py
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

## Pipeline: From Data to Chatbot

### Step 1 — Scope (done)
Assistant answers: maintenance advice, troubleshooting, service schedules, car symptoms → causes, DIY fixes.

### Step 2 — Prepare dataset
- **Clean:** `prepare_training_data.py` strips HTML, normalizes text, removes duplicates; optional relevance filter.
- **Format:** Alpaca-style `instruction` / `input` / `output`.
- **Split:** 80% train, 10% val, 10% test.

```bash
python prepare_training_data.py
```

### Step 3 — Base model
For Colab free GPU: **TinyLlama-1.1B-Chat** (recommended), or Gemma-2B / Phi-2.

### Step 4 — Environment (Colab)
```bash
pip install transformers datasets peft accelerate bitsandbytes trl
```

### Step 5 — LoRA fine-tune
- **Script (local):** `python scripts/train_lora.py`
- **Colab notebook:** `notebooks/colab_finetune_car_maintenance.ipynb` — upload `data/training/train.json`, run all cells.

Recommended: `learning_rate=2e-4`, `batch_size=2`, `epochs=2`, LoRA `r=16`, `max_length=512`.

### Step 6 — Train and track
- Use `EXPERIMENTS.md` to log: hyperparameters, GPU memory, training time, final loss.
- Save model: `car-maintenance-llm/` (adapter + tokenizer).

### Step 7 — Evaluate
- **Quantitative:** BLEU, ROUGE, perplexity via `evaluate.py`.
- **Qualitative:** Domain questions (e.g. “Why does my engine knock?”, “When change oil?”).
- **Out-of-domain:** e.g. “Who is president?” — should refuse or answer weakly.

```bash
pip install nltk rouge-score
python scripts/evaluate.py --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --peft_model car-maintenance-llm
```

### Step 8 — Save and export
Model and tokenizer are saved under `car-maintenance-llm/`. From Colab, download the folder or zip it.

### Step 9 — Chatbot (Gradio)
```bash
pip install gradio
python app/app_gradio.py --model_path car-maintenance-llm
```
Optional: `--share` for a public link.

### Step 10 — Deploy
- **Colab:** Run `app_gradio.py` in Colab and use “Share” for a public link.
- **Hugging Face Spaces:** Upload model and a Gradio app.
- **Local:** Run `app_gradio.py` and open the local URL.

### Step 11 — Document (report)
Use `EXPERIMENTS.md`: dataset, preprocessing, LoRA settings, hyperparameters, training time, GPU usage, metrics table, base vs tuned comparison, chatbot screenshots.

### Step 12 — Final testing
Example scenarios: “When change timing belt?”, “Car shakes at highway speed”, “Brake pedal soft”, “Grinding noise when turning”.

---

## Next Steps (optional)

1. **Expand dataset** to 1k–5k pairs: run `expand_dataset.py`, add from manuals/FAQs.
2. **Tune hyperparameters:** try different LoRA rank, learning rate, epochs; log in `EXPERIMENTS.md`.
3. **Improve evaluation:** add more test questions, human judgment on clarity and correctness.

## Project Structure

```
auto-maintenance-assistant/
├── app/                            # Chatbot UI
│   ├── _project_root.py
│   └── app_gradio.py
├── notebooks/                      # Colab training
│   └── colab_finetune_car_maintenance.ipynb
├── scripts/                        # Data & training
│   ├── _project_root.py
│   ├── collect_dataset.py         # Initial dataset collection
│   ├── expand_dataset.py          # Dataset expansion
│   ├── prepare_training_data.py   # Clean + format + split (Alpaca)
│   ├── train_lora.py             # LoRA fine-tuning
│   └── evaluate.py               # BLEU, ROUGE, perplexity
├── data/
│   ├── auto_maintenance_dataset.json
│   ├── auto_maintenance_dataset.csv
│   └── training/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── car-maintenance-llm/           # Saved model (after training)
├── RUN_COMMANDS.md                # Step-by-step run commands
├── EXPERIMENTS.md
└── README.md
```

**Run all commands from the project root.** See **RUN_COMMANDS.md** for step-by-step instructions.

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

MIT

## References

- Hugging Face Datasets: https://huggingface.co/datasets
- LoRA Fine-tuning: https://huggingface.co/docs/peft
- Alpaca Dataset Format: https://github.com/tatsu-lab/stanford_alpaca
