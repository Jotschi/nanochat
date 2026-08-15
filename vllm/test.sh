#!/bin/bash
#
# Two chat queries against a running vLLM server.
#
#   bash vllm/test.sh
#   BASE_URL=http://127.0.0.1:8001/v1 bash vllm/test.sh
#
# Query 1 asks for a story. Query 2 feeds that story back and asks a question
# about it, which is the whole point of the model - a single-turn test would not
# exercise the reading-comprehension half at all.

set -o nounset
set -o errexit

BASE_URL=${BASE_URL:-http://127.0.0.1:8000/v1}
MODEL=${MODEL:-kleiner-astronaut}
TEMP=${TEMP:-0.6}

command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }

if ! curl -sf --max-time 5 "$BASE_URL/models" >/dev/null; then
    echo "No server at $BASE_URL - start one with: bash vllm/start-vllm.sh" >&2
    exit 1
fi

# Ask the server what it actually loaded, rather than assuming --served-model-name.
SERVED=$(curl -sf "$BASE_URL/models" | jq -r '.data[0].id')
[ -n "$SERVED" ] && [ "$SERVED" != "null" ] && MODEL="$SERVED"
echo "Model: $MODEL"
echo

# chat <json-messages-array> <max-tokens>
# Note: no `curl -f` here. With -f an HTTP 400 exits non-zero and prints nothing,
# so a template or parameter error looks exactly like an empty completion. Read
# the body and surface the server's own message instead.
chat() {
    local messages="$1" max_tokens="$2" response
    response=$(jq -n --argjson messages "$messages" --arg model "$MODEL" \
          --argjson max_tokens "$max_tokens" --argjson temperature "$TEMP" \
        '{model: $model, messages: $messages, max_tokens: $max_tokens, temperature: $temperature}' \
    | curl -s --max-time 180 -X POST "$BASE_URL/chat/completions" \
        -H 'Content-Type: application/json' -d @-)

    if jq -e '.error' >/dev/null 2>&1 <<<"$response"; then
        echo "server error: $(jq -r '.error.message' <<<"$response")" >&2
        exit 1
    fi
    jq -r '.choices[0].message.content' <<<"$response"
}

# --- Query 1: write a story -------------------------------------------------
REQUEST="Schreib ein Abenteuer von Mira in das Raumschiff nur mit Aris"
echo "=== 1. Story request ==="
echo "USER  : $REQUEST"
STORY=$(chat "$(jq -n --arg c "$REQUEST" '[{role:"user",content:$c}]')" 400)
echo "ASSIST: $STORY"
echo

# --- Query 2: ask about that story ------------------------------------------
QUESTION="Was hat Mira entdeckt?"
echo "=== 2. Question about the story ==="
echo "USER  : $QUESTION"
ANSWER=$(chat "$(jq -n --arg req "$REQUEST" --arg story "$STORY" --arg q "$QUESTION" \
    '[{role:"user",content:$req},{role:"assistant",content:$story},{role:"user",content:$q}]')" 64)
echo "ASSIST: $ANSWER"
echo

# The story should contain the two words the request asked for; that is the same
# rl_key check tasks/kleiner_astronaut.py uses for the RL reward.
for word in Mira Aris; do
    if grep -qi -- "$word" <<<"$STORY"; then
        echo "  keyword '$word': present"
    else
        echo "  keyword '$word': MISSING"
    fi
done
