"""
Evaluate the Kleiner Astronaut chat model on the two things it is supposed to do.

`scripts/chat_eval.py` scores ARC / MMLU / GSM8K / HumanEval, which are English
multiple-choice and code benchmarks. None of them say anything about a German
story model, so this is the domain-specific replacement.

Two metrics, both measured on the held-out conversations:

  story    Given the request, does the generated story contain the keywords the
           request asked for? (rl_key1 / rl_key2 on the story turn.)
  answer   Given request + story + question, does the generated answer contain
           the expected answer word? (rl_key1 on the answer turn.) This is the
           reading-comprehension half and only exists for four-turn rows.

Example:
    python -m scripts.chat_astronaut_eval -i sft
    python -m scripts.chat_astronaut_eval -i sft --max-problems 200 --temperature 0.6
"""

import os
import argparse

import torch

from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type, get_base_dir
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

from tasks.kleiner_astronaut import KleinerAstronaut, contains_word

# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Evaluate the Kleiner Astronaut chat model")
parser.add_argument("-i", "--source", type=str, default="sft", help="checkpoint to load: base|sft|rl")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--conversations", type=str, default=None,
                    help="path to the val conversations jsonl (default: NANOCHAT_BASE_DIR)")
parser.add_argument("--max-problems", type=int, default=200, help="how many conversations to score (-1 = all)")
parser.add_argument("--max-new-tokens", type=int, default=512, help="generation budget for the story turn")
parser.add_argument("--max-answer-tokens", type=int, default=64, help="generation budget for the answer turn")
parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
parser.add_argument("--top-k", type=int, default=50, help="top-k, only used when temperature > 0")
args = parser.parse_args()

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

conversations_path = args.conversations or os.path.join(
    get_base_dir(), "kleiner_astronaut_conversations_val.jsonl")
task = KleinerAstronaut(filepath=conversations_path)
print0(f"Loaded {len(task):,} conversations from {conversations_path}")

model, tokenizer, meta = load_model(args.source, device, phase="eval",
                                    model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer)


def generate(messages, max_tokens):
    """Prime the assistant after `messages` and return the decoded completion."""
    prompt_ids = tokenizer.render_for_completion({"messages": messages})
    results, _ = engine.generate_batch(
        prompt_ids,
        num_samples=1,
        max_tokens=max_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.temperature > 0 else None,
    )
    return tokenizer.decode(results[0][len(prompt_ids):])


def required_keys(message):
    return [message[k] for k in KleinerAstronaut.REWARD_KEYS if message.get(k)]


story_passed = story_total = 0
answer_passed = answer_total = 0

num_problems = len(task) if args.max_problems < 0 else min(len(task), args.max_problems)
for i in range(ddp_rank, num_problems, ddp_world_size):
    messages = task[i]["messages"]

    # --- story turn: request -> story ---------------------------------------
    story_keys = required_keys(messages[1])
    if story_keys:
        completion = generate(messages[:2], args.max_new_tokens)
        story_total += 1
        story_passed += int(all(contains_word(completion, key) for key in story_keys))

    # --- answer turn: request, story, question -> answer ---------------------
    # The reference story is kept in the context so this measures reading
    # comprehension rather than the model's ability to remember its own output.
    if len(messages) >= 4:
        answer_keys = required_keys(messages[3])
        if answer_keys:
            completion = generate(messages[:4], args.max_answer_tokens)
            answer_total += 1
            answer_passed += int(all(contains_word(completion, key) for key in answer_keys))

    if (story_total + answer_total) % 20 == 0:
        print(f"\r\033[KRank {ddp_rank} | story {story_passed}/{story_total} | "
              f"answer {answer_passed}/{answer_total}", end="", flush=True)
print()

if ddp:
    import torch.distributed as dist
    counts = torch.tensor([story_passed, story_total, answer_passed, answer_total],
                          dtype=torch.long, device=device)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    story_passed, story_total, answer_passed, answer_total = counts.tolist()

pct = lambda n, d: f"{100 * n / d:.2f}%" if d else "n/a"
print0("=" * 60)
print0(f"Keyword adherence (story)  : {story_passed}/{story_total} ({pct(story_passed, story_total)})")
print0(f"Question answering (answer): {answer_passed}/{answer_total} ({pct(answer_passed, answer_total)})")
print0("=" * 60)

compute_cleanup()
