"""Gradio chat playground for tiny-llm-template."""

import json

import gradio as gr
import torch
from config import Config
from huggingface_hub import hf_hub_download
from model import GPT
from safetensors.torch import load_model
from tokenizers import Tokenizer

REPO = "your-username/tiny-llm-template"  # Replace with your HF model repo
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_K = 20

cfg_path = hf_hub_download(REPO, "config.json")
with open(cfg_path) as f:
    cfg_dict = json.load(f)
cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config.__dataclass_fields__})
model = GPT(cfg).eval()
load_model(model, hf_hub_download(REPO, "model.safetensors"))
tok = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json"))
EOS_ID = tok.token_to_id("<|endoftext|>")


@torch.no_grad()
def respond(message: str, history):
    prompt = PROMPT_TEMPLATE.format(instruction=message.strip())
    ids = torch.tensor([tok.encode(prompt).ids], dtype=torch.long)
    prompt_len = ids.size(1)

    for _ in range(MAX_NEW_TOKENS):
        logits, _ = model(ids[:, -cfg.block_size :])
        logits = logits[:, -1] / TEMPERATURE
        v, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)

        if EOS_ID is not None and next_id.item() == EOS_ID:
            break

        yield tok.decode(ids[0, prompt_len:].tolist())

    yield tok.decode(ids[0, prompt_len:].tolist())


neutral_theme = gr.themes.Base(
    primary_hue="slate",
    secondary_hue="blue",
    neutral_hue="gray",
).set(
    body_background_fill="#f8fafc",
    body_background_fill_dark="#0f172a",
    background_fill_primary="#ffffff",
    background_fill_secondary="#f1f5f9",
    block_background_fill="#ffffff",
    block_border_color="#cbd5e1",
    block_border_width="1px",
    block_label_text_color="#334155",
    block_title_text_color="#334155",
    body_text_color="#1e293b",
    button_primary_background_fill="#3b82f6",
    button_primary_text_color="#ffffff",
    button_primary_background_fill_hover="#2563eb",
    input_background_fill="#ffffff",
    input_background_fill_focus="#ffffff",
    input_border_color="#cbd5e1",
    input_text_size="*text_lg",
)


CUSTOM_CSS = """
/* ---------- Layout ---------- */
html, body, gradio-app, .gradio-container, .main, .contain {
    height: 100vh !important;
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}
.gradio-container { font-size: 18px !important; }
footer { display: none !important; }

/* ---------- Chatbot reset: nuke EVERY visual inside the chatbot ---------- */
[data-testid="chatbot"] *,
[data-testid="chatbot"] *::before,
[data-testid="chatbot"] *::after,
.chatbot *, [class*="chatbot"] * {
    border: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    outline: 0 !important;
    background: transparent !important;
    background-color: transparent !important;
    background-image: none !important;
    color: #1e293b !important;
}
[data-testid="chatbot"] *::before,
[data-testid="chatbot"] *::after { display: none !important; }

/* One single border on the chatbot container. */
[data-testid="chatbot"] {
    border: 1px solid #cbd5e1 !important;
    background: #ffffff !important;
}

/* ---------- Avatars ---------- */
[data-testid="chatbot"] img {
    background: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 50% !important;
    display: inline-block !important;
}

/* ---------- Input textbox ---------- */
textarea, input[type="text"] {
    color: #1e293b !important;
    background: #ffffff !important;
    caret-color: #3b82f6 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
}
textarea::placeholder, input::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}

/* ---------- Example chips ---------- */
.examples button, [class*="example"] button {
    color: #1e293b !important;
    background: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
}
.examples button *, [class*="example"] button * {
    border: 0 !important;
    background: transparent !important;
    color: #1e293b !important;
}

/* ---------- Misc ---------- */
button { border-radius: 6px !important; }
h1, h2, h3 { color: #334155 !important; }
"""

chatbot = gr.Chatbot(
    label="Tiny LLM",
    avatar_images=("assets/avatar.png", "assets/avatar.png"),
    show_label=False,
    height="80vh",
)

textbox = gr.Textbox(
    placeholder="Enter your message…",
    label="User",
    container=True,
    scale=1,
)

with gr.Blocks(title="Tiny LLM") as demo:
    gr.ChatInterface(
        respond,
        chatbot=chatbot,
        textbox=textbox,
        examples=[
            ["Tell me a story."],
            ["Explain quantum computing in simple terms."],
            ["Write a short poem about space."],
        ],
        cache_examples=False,
    )


if __name__ == "__main__":
    demo.launch(theme=neutral_theme, css=CUSTOM_CSS)
