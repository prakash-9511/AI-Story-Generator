 app# 📖 AI Story Generator

Generate short stories using the **text-generation pipeline** with **GPT-2**.

---

## Project structure

```
ai-story-generator/
├── app.py            # Gradio web app (local + HF Spaces deployment)
├── demo.py           # Quick demo — runs all 3 genres in the terminal
├── generate.py       # Full CLI with flags for every parameter
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone / download the repo
git clone https://github.com/<your-username>/ai-story-generator.git
cd ai-story-generator

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note** — `torch` will download ~2 GB the first time. GPT-2 weights (~500 MB) are cached in `~/.cache/huggingface/` on first use.

---

## Usage

### Quick demo (all 3 genres)

```bash
python demo.py
```

**Output includes:**

| Genre | Example Prompt |
|-------|---------------|
| 🚀 Space | `"In the year 2050, robots started"` |
| 🕯️ Horror | `"The old mansion at the end of the street had been abandoned for 30 years, until tonight"` |
| ✨ Motivational | `"Every great journey begins with a single step, and today"` |

---

### CLI — full control

```bash
# Space story, 2 sequences, creative temperature
python generate.py --genre space --num 2 --temperature 1.1

# Horror story, 300 tokens
python generate.py --genre horror --max-length 300

# Custom prompt
python generate.py --prompt "The last dragon on Earth opened its eyes" --max-length 250

# Reproducible output
python generate.py --genre motivational --seed 42
```

**All flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--genre` | `space` | `space`, `horror`, `motivational` |
| `--prompt` | — | Custom prompt (overrides genre) |
| `--max-length` | `200` | Max tokens in generated text |
| `--num` | `1` | `num_return_sequences` |
| `--temperature` | `0.8` | Sampling temperature (0.3 – 2.0) |
| `--seed` | None | Random seed for reproducibility |

---

### Gradio web app

```bash
python app.py
# → http://localhost:5000
```

---

## Pipeline parameters explained

```python
generator = pipeline("text-generation", model="gpt2")

result = generator(
    prompt,
    max_length=200,           # total tokens (prompt + generated)
    num_return_sequences=1,   # independent stories generated
    temperature=0.8,          # < 1 = focused, > 1 = creative/random
    do_sample=True,           # required when temperature != 1.0
    pad_token_id=50256,       # suppresses GPT-2 padding warning
)
```

| Parameter | Effect |
|-----------|--------|
| `max_length` | Higher → longer story (more memory/time) |
| `num_return_sequences` | Each sequence is an independent continuation |
| `temperature` | 0.3 = very deterministic · 0.8 = balanced · 1.5+ = wild/creative |

---
