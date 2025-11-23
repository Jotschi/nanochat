"""
CustomJSON task for loading conversations from JSONL files.
Each line in the JSONL file should be a JSON array of messages.
"""

import os
import json
from tasks.common import Task

class CustomJSON(Task):
    """
    Load conversations from a JSONL file.
    Each line should be a JSON array of message objects with 'role' and 'content' fields.
    Example line: [{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]
    """

    def __init__(self, it, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations = []


        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # skip empty lines
                    continue
                messages = json.loads(line)
                # Validate the conversation structure
                assert isinstance(messages, list), f"Expected list of messages, got {type(messages)}"
                assert len(messages) >= 2, f"Conversation must have at least 2 messages, got {len(messages)}"
                # Validate message structure and alternating roles
                for i, message in enumerate(messages):
                    assert "role" in message, f"Message {i} missing 'role' field"
                    assert "content" in message, f"Message {i} missing 'content' field"
                    expected_role = "user" if i % 2 == 0 else "assistant"
                    assert message["role"] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"
                    assert isinstance(message["content"], str), f"Message {i} content must be a string"

                # Duplicate records
                for n in range(it):
                    #print("Len: " + str(len(self.conversations)))
                    self.conversations.append(messages)

        self.length = len(self.conversations)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        messages = self.conversations[index]
        conversation = {
            "messages": messages,
        }
        return conversation



    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        Note that:
        - the conversation has both user AND assistant message (containing the ground truth answer)
        - the assistant_response is usually the alternative assistant message achieved via sampling

        TODO: Technically, assistant_response should be a Message (either a string or a list of parts)
              We can handle this later possibly. For now just assume string.
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        # First extract the ground truth answer
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        
        if "rl_key1" in assistant_message:
            key1 = assistant_message["rl_key1"]
            key1 = key1.lower()
            # Poor mans declination handling
            key1 = key1[:-2]
            #print("Checking: " + key1)
            is_present = key1 in assistant_response.lower()
            if not is_present:
                return False

        if "rl_key2" in assistant_message:
            key2 = assistant_message["rl_key1"]
            key2 = key2.lower()
            # Poor mans declination handling
            key2 = key2[:-2]
            #print("Checking: " + key2)
            is_present = key2 in assistant_response.lower()
            if not is_present:
                return False
            
        return True


    def reward(self, conversation, assistant_response):
        """
        Used during RL. To keep things simple, just re-use the evaluation above.
        Later this could be made more complex (e.g. format matching etc.)
        """
        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)
        return is_correct_float

def main():
    print("MAIN")
    path = os.path.join(".cache", "nanochat", "kleiner_astronaut_conversations_v3_val.jsonl")
    ds = CustomJSON(5, filepath=path)
    conv = ds.get_example(10)
    result = ds.evaluate(conv,  "Raumschiff")
    print("Result: " + str(result))

    
if __name__ == "__main__":
    main()
