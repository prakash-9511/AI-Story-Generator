"""
AI Story Generator
==================
Text-generation pipeline using GPT-2 model.
Supports: Space, Horror, Motivational, and Custom prompts.

Run locally:
    python app.py

Deploy to Hugging Face Spaces:
    Push this repo — Spaces auto-detects Gradio.
"""

import gradio as gr
from transformers import pipeline, set_seed
import time

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

print("Loading GPT-2 text-generation pipeline...")
generator = pipeline("text-generation", model="gpt2")
print("Pipeline ready.")

# ---------------------------------------------------------------------------
# Default prompts per genre
# ---------------------------------------------------------------------------

GENRE_PROMPTS = {
    "🚀 Space":        "In the year 2050, robots started",
    "🕯️ Horror":       "The old mansion at the end of the street had been abandoned for 30 years, until tonight",
    "✨ Motivational":  "Every great journey begins with a single step, and today",
    "✏️ Custom":        "",
}

# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate_stories(
    genre: str,
    custom_prompt: str,
    max_length: int,
    num_return_sequences: int,
    temperature: float,
    seed: int,
) -> str:
    """
    Run the GPT-2 text-generation pipeline and return formatted output.

    Parameters
    ----------
    genre               : selected story genre
    custom_prompt       : user prompt (used when genre is Custom)
    max_length          : max tokens for each generated sequence
    num_return_sequences: how many stories to generate
    temperature         : sampling temperature (higher = more creative)
    seed                : random seed for reproducibility (-1 = random)
    """

    # Determine prompt
    if genre == "✏️ Custom":
        prompt = custom_prompt.strip()
    else:
        prompt = GENRE_PROMPTS.get(genre, "").strip()

    if not prompt:
        return "⚠️  Please enter a prompt or select a genre."

    # Set seed
    if seed >= 0:
        set_seed(int(seed))

    start = time.time()

    results = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        do_sample=True,
        pad_token_id=50256,   # GPT-2 EOS token — suppresses warning
    )

    elapsed = time.time() - start

    # Format output
    lines = []
    lines.append(f"📌 Prompt : {prompt}")
    lines.append(f"⚙️  Params : max_length={max_length} | num_return_sequences={num_return_sequences} | temperature={temperature:.1f}")
    lines.append(f"⏱️  Time   : {elapsed:.2f}s")
    lines.append("=" * 70)

    for i, result in enumerate(results, 1):
        story_text = result["generated_text"]
        word_count = len(story_text.split())
        lines.append(f"\n── Sequence {i} of {num_return_sequences} ({word_count} words) ──\n")
        lines.append(story_text)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def update_prompt(genre: str) -> str:
    """Return the default prompt when a genre chip is selected."""
    return GENRE_PROMPTS.get(genre, "")


CUSTOM_CSS = """
#header { text-align: center; margin-bottom: 1rem; }
#header h1 { font-size: 2rem; margin: 0; }
#header p  { color: #6b7280; margin: 4px 0 0; }
.output-box textarea { font-family: Georgia, serif; font-size: 15px; line-height: 1.7; }
"""

with gr.Blocks(title="AI Story Generator") as demo:

    # Header
    gr.HTML("""
        <div id="header">
            <h1>📖 AI Story Generator</h1>
            <p>Text-generation pipeline · model: <code>gpt2</code></p>
        </div>
    """)

    with gr.Row():
        # ── Left column: controls ──────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Genre")
            genre = gr.Radio(
                choices=list(GENRE_PROMPTS.keys()),
                value="🚀 Space",
                label="",
                interactive=True,
            )

            prompt_box = gr.Textbox(
                label="Prompt",
                value=GENRE_PROMPTS["🚀 Space"],
                lines=3,
                placeholder="Enter your story prompt here...",
            )

            gr.Markdown("### Pipeline Parameters")

            max_length = gr.Slider(
                minimum=50, maximum=500, value=200, step=10,
                label="max_length  (tokens)",
                info="Maximum number of tokens in the generated output.",
            )

            num_sequences = gr.Slider(
                minimum=1, maximum=4, value=1, step=1,
                label="num_return_sequences",
                info="How many independent stories to generate.",
            )

            temperature = gr.Slider(
                minimum=0.3, maximum=2.0, value=0.8, step=0.1,
                label="temperature",
                info="Higher = more creative/random. Lower = more focused.",
            )

            seed = gr.Number(
                value=-1, label="Random seed  (-1 = random)",
                precision=0,
            )

            with gr.Row():
                generate_btn = gr.Button("✨ Generate", variant="primary", scale=3)
                clear_btn    = gr.Button("🗑 Clear", scale=1)

        # ── Right column: output ───────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### Generated Stories")
            output = gr.Textbox(
                label="",
                lines=28,
                max_lines=60,
                elem_classes=["output-box"],
                placeholder="Your generated stories will appear here...",
            )

    # ── Examples ──────────────────────────────────────────────────────────
    gr.Markdown("### Example Prompts")
    gr.Examples(
        examples=[
            ["🚀 Space",       "In the year 2050, robots started",                                 200, 1, 0.8,  42],
            ["🕯️ Horror",      "The old mansion at the end of the street had been abandoned",      250, 2, 1.0,  -1],
            ["✨ Motivational", "Every great journey begins with a single step, and today",         150, 1, 0.7,  7 ],
            ["✏️ Custom",       "The last dragon on Earth opened its eyes after a thousand years", 200, 1, 0.9,  -1],
        ],
        inputs=[genre, prompt_box, max_length, num_sequences, temperature, seed],
        label="",
    )

    # ── Event wiring ──────────────────────────────────────────────────────
    genre.change(fn=update_prompt, inputs=genre, outputs=prompt_box)

    generate_btn.click(
        fn=generate_stories,
        inputs=[genre, prompt_box, max_length, num_sequences, temperature, seed],
        outputs=output,
    )

    clear_btn.click(fn=lambda: "", outputs=output)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=5000,
        share=False,          # set True to get a public Gradio link
        show_error=True,
        theme=gr.themes.Soft(primary_hue="violet"),
        css=CUSTOM_CSS,
    )
