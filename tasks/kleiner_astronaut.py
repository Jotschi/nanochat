"""
Kleiner Astronaut conversations: German children's astronaut stories, plus a
reading-comprehension question about the story just told.

The jsonl is produced by the Java generator, see spec/DATA.md. One JSON *array*
per line, strictly alternating user/assistant starting with user:

    [{"role":"user","content":"<request>"},
     {"role":"assistant","content":"<story>","rl_key1":"...","rl_key2":"..."},
     {"role":"user","content":"<question>"},
     {"role":"assistant","content":"<answer>","rl_key1":"<answer_word>"}]

`rl_key1`/`rl_key2` are reward keys for the RL stage: German nouns or names the
generated text must contain. They sit on assistant turns only, and only when the
converter could verify the keyword really occurs in the reference text.

This replaces the old `tasks/customjson.py`. Three differences matter:

1. No `it` duplication parameter. Repeating rows N times made the LR schedule
   blind to how much data it was really seeing, which is how the December 2025
   run ended up doing ~200 epochs at peak learning rate. Repetition belongs in
   the training script, as epochs.
2. `rl_key2` is actually checked. The old code re-read `rl_key1` in the
   `rl_key2` branch, so the second keyword was never enforced.
3. A conversation with no reward keys raises instead of scoring 1.0. The old
   code read the keys off the message dict but only validated role/content, so
   missing keys silently made every rollout correct.
"""

import os
import json

from tasks.common import Task

# Mirrors de.jotschi.ai.converter.stage3.Words: German inflects, so we match on a
# shortened needle -- but only when the word is long enough and is not a number.
# Without those guards a short key stems to "" and matches any text at all.
_MIN_STEM_LEN = 5


def stem(word):
    """Lower-cased matching needle, shortened by two characters when it is safe."""
    if word is None:
        return None
    lower = word.lower()
    if len(lower) > _MIN_STEM_LEN and not lower.replace(".", "", 1).isdigit():
        return lower[:-2]
    return lower


def contains_word(text, word):
    """True when text contains word, tolerating German inflection."""
    needle = stem(word)
    if not text or not needle:
        return False
    return needle in text.lower()


class KleinerAstronaut(Task):
    """
    Conversations from a jsonl file of JSON arrays.

    Args:
        filepath: path to the conversations jsonl
        require_reward_keys: raise if a conversation carries no rl_key at all.
            Leave on for RL, where a keyless conversation is a silent reward-1.0
            bug; turn off for plain SFT, where reward keys are not used.
        stop: optionally truncate to the first N conversations
    """

    REWARD_KEYS = ("rl_key1", "rl_key2")

    def __init__(self, filepath, require_reward_keys=False, stop=None, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations = []

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"{filepath} not found. Generate it with the Java converter:\n"
                "  cd dataset-processor && mvn -q compile exec:java "
                "-Dexec.mainClass=de.jotschi.ai.converter.stage3.Dataset2Chat"
            )

        with open(filepath, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                messages = json.loads(line)
                self._validate(messages, lineno)
                if require_reward_keys and not self._has_reward_key(messages):
                    raise ValueError(
                        f"{filepath}:{lineno} has no rl_key on any assistant turn. "
                        "With require_reward_keys=True every conversation must be scorable; "
                        "otherwise the reward is trivially 1.0."
                    )
                self.conversations.append(messages)
                if stop is not None and len(self.conversations) >= stop:
                    break

        self.length = len(self.conversations)

    @staticmethod
    def _validate(messages, lineno):
        assert isinstance(messages, list), f"line {lineno}: expected a list of messages, got {type(messages)}"
        assert len(messages) >= 2, f"line {lineno}: need at least 2 messages, got {len(messages)}"
        for i, message in enumerate(messages):
            assert "role" in message, f"line {lineno}: message {i} missing 'role'"
            assert "content" in message, f"line {lineno}: message {i} missing 'content'"
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, \
                f"line {lineno}: message {i} has role {message['role']}, expected {expected_role}"
            assert isinstance(message["content"], str), f"line {lineno}: message {i} content must be a string"

    @classmethod
    def _has_reward_key(cls, messages):
        return any(k in m for m in messages for k in cls.REWARD_KEYS)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        return {"messages": self.conversations[index]}

    def evaluate(self, conversation, assistant_response):
        """
        Did the sampled response contain every keyword the reference turn requires?

        Scores against the *last* assistant message, which is what the engine was
        asked to produce. Returns False when there is nothing to score, so an
        unscorable conversation can never look like a success.
        """
        assert isinstance(assistant_response, str), "Assuming a simple string response for now"
        assistant_message = conversation["messages"][-1]
        assert assistant_message["role"] == "assistant", "Last message must be from the assistant"

        keys = [assistant_message[k] for k in self.REWARD_KEYS if assistant_message.get(k)]
        if not keys:
            return False
        return all(contains_word(assistant_response, key) for key in keys)

    def reward(self, conversation, assistant_response):
        """Used during RL. Binary keyword presence, same as evaluate()."""
        return float(self.evaluate(conversation, assistant_response))
