"""
generate.py — Command-line story generator
==========================================
Usage examples
--------------
# Generate one space story (default)
python generate.py

# Generate 2 horror stories with high temperature
python generate.py --genre horror --num 2 --temperature 1.2

# Custom prompt
python generate.py --prompt "The alien ship landed silently in the field" --max-length 300

# Full control
python generate.py --genre space --max-length 250 --num 3 --temperature 0.9 --seed 42
"""

import argparse
import time
from transformers import pipeline, set_seed


# ---------------------------------------------------------------------------
# Predefined prompts
# ---------------------------------------------------------------------------

PROMPTS = {
    "space":       "In the year 2050, robots started",
    "horror":      "The old mansion at the end of the street had been abandoned for 30 years, until tonight",
    "motivational":"Every great journey begins with a single step, and today",
}

DIVIDER = "=" * 70


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Story Generator — GPT-2 text-generation pipeline"
    )
    parser.add_argument(
        "--genre",
        choices=list(PROMPTS.keys()),
        default="space",
        help="Story genre (default: space)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt. Overrides --genre.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=200,
        dest="max_length",
        help="Max tokens in generated output (default: 200)",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=1,
        dest="num_return_sequences",
        help="Number of stories to generate (default: 1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature 0.3–2.0 (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    prompt = args.prompt if args.prompt else PROMPTS[args.genre]

    print(f"\n{DIVIDER}")
    print("  AI Story Generator  |  model: gpt2")
    print(DIVIDER)
    print(f"  Prompt             : {prompt}")
    print(f"  max_length         : {args.max_length}")
    print(f"  num_return_seq     : {args.num_return_sequences}")
    print(f"  temperature        : {args.temperature}")
    print(f"  seed               : {args.seed}")
    print(DIVIDER)
    print("\nLoading pipeline...")

    generator = pipeline("text-generation", model="gpt2")

    if args.seed is not None:
        set_seed(args.seed)

    print("Generating...\n")
    start = time.time()

    results = generator(
        prompt,
        max_length=args.max_length,
        num_return_sequences=args.num_return_sequences,
        temperature=args.temperature,
        do_sample=True,
        pad_token_id=50256,
    )

    elapsed = time.time() - start

    for i, result in enumerate(results, 1):
        text = result["generated_text"]
        words = len(text.split())
        print(f"\n── Sequence {i} of {args.num_return_sequences}  ({words} words) ──\n")
        print(text)

    print(f"\n{DIVIDER}")
    print(f"  Done in {elapsed:.2f}s")
    print(DIVIDER)


if __name__ == "__main__":
    main()
