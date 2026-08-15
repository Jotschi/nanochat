# Goals

Train a small German language model ("Kleiner Astronaut") that can:

1. **Write a children's astronaut story on request.** Given a German request such as
   *"Schreib ein Abenteuer von Mira im Raumschiff"*, produce a short, coherent, child-appropriate
   space adventure that actually contains the requested keywords.
2. **Answer a question about the story it just wrote.** Given a follow-up W-Frage
   (*Wer / Was / Wie / Wo / Womit / Wann / Mit wem / Weshalb / Warum*), answer it correctly from the
   story text.

Everything needed to do that lives in this one repository: the Java data generator
(`dataset-processor/`), the adapted nanochat training code, the run scripts, and these notes.

## Success criteria

| # | Criterion | How it is checked |
| --- | --- | --- |
| G1 | Base model produces fluent German prose from a bare prompt | `scripts/base_eval` samples on the German prompt list |
| G2 | The LR schedule completes, and the horizon ends at the validation minimum | training log: `lrm` reaches ~0 at the final step, and the last `val_bpb` is the run's lowest |
| G3 | SFT model emits well-formed 2- and 4-turn conversations with the chat special tokens | `scripts/chat_cli` |
| G4 | Requested keywords appear in the generated story | `rl_key1`/`rl_key2` presence check (`tasks/kleiner_astronaut.py`) |
| G5 | The answer to the follow-up question contains the expected `answer_word` | German story-QA eval |
| G6 | No regression of the Dec-2025 failure mode | SFT step-0 loss close to the base model's, train and val move together |

## Non-goals

- General conversational ability or English. The SmolTalk translation path stays out of scope.
- Reinstating a separate `mid_train` stage — upstream folded that into `chat_sft`.
- Exporting to HuggingFace format (`convert/` predates the current architecture).
- Multi-GPU training. The 3060 is too small to pair with the 4090 under DDP.
