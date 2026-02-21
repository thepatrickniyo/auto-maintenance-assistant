"""
Gradio chatbot for Car Maintenance LLM Assistant.
Run from project root: python app/app_gradio.py
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr

from _project_root import PROJECT_ROOT


def load_model(base_model_id: str, model_path: str, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    path = Path(model_path)
    if (path / "tokenizer_config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    if path.exists() and (path / "adapter_config.json").exists():
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, str(path))
        model = model.merge_and_unload()
    elif path.exists() and (path / "config.json").exists():
        model = AutoModelForCausalLM.from_pretrained(
            str(path),
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
    model.eval()
    return model, tokenizer


def reply(message: str, model, tokenizer, max_new_tokens: int = 256) -> str:
    """Generate assistant reply for one user message. Returns only the reply string."""
    if not message or not message.strip():
        return ""
    prompt = f"### Instruction:\n{message}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.3, top_p=0.9,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    if "### Response:" in full:
        response = full.split("### Response:")[-1].strip()
    else:
        response = full[len(prompt):].strip()
    response = response.split("\n\n")[0].strip() if response else "I don't have an answer for that."
    return response


def build_ui(model, tokenizer, max_new_tokens: int = 256):
    # Gradio 5/6 ChatInterface: fn(message, history) must return only the assistant reply (str)
    def chat_fn(message, history):
        return reply(message, model, tokenizer, max_new_tokens)

    chat = gr.ChatInterface(
        fn=chat_fn,
        title="Car Maintenance Assistant",
        description="Ask about car maintenance, troubleshooting, service schedules, and DIY fixes.",
        examples=[
            "Why is my car overheating?",
            "When should I change brake pads?",
            "Car won't start clicking noise",
            "How often should I change my engine oil?",
            "What does the check engine light mean?",
        ],
    )
    with gr.Blocks() as demo:
        gr.Markdown("### Car Maintenance LLM Assistant")
        chat.render()
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--model_path", type=str, default=None, help="Path to fine-tuned model (default: project root / car-maintenance-llm)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server_port", type=int, default=7860)
    args = parser.parse_args()

    model_path = args.model_path
    if model_path is None:
        model_path = str(PROJECT_ROOT / "car-maintenance-llm")
    if not Path(model_path).exists():
        print(f"Warning: {model_path} not found. Using base model only.")
        model_path = args.base_model

    print("Loading model...")
    model, tokenizer = load_model(args.base_model, model_path)
    demo = build_ui(model, tokenizer, args.max_new_tokens)
    demo.launch(share=args.share, server_port=args.server_port, css=".gradio-container { max-width: 700px; margin: auto; }")


if __name__ == "__main__":
    main()
