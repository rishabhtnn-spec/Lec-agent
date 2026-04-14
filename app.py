from flask import Flask, request, jsonify, session
import anthropic
from datetime import datetime
import math, os, secrets
import json

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

client = anthropic.Anthropic()
MEMORIES_DIR = "memories"

SYSTEM_PROMPT = """You are Lec, a sharp product strategist.
You have access to tools — use them when calculations or
current date/time are needed. Think step by step."""


def load_memory(user_id):
    os.makedirs(MEMORIES_DIR, exist_ok=True)
    filename = os.path.join(MEMORIES_DIR, f"memory_{user_id}.json")
    default_memory = {
        "user_name": "",
        "key_facts": [],
        "conversation_summary": "",
        "message_count": 0
    }

    if not os.path.exists(filename):
        return default_memory

    try:
        with open(filename, "r", encoding="utf-8") as f:
            memory = json.load(f)
    except (OSError, json.JSONDecodeError):
        return default_memory

    merged_memory = default_memory.copy()
    merged_memory.update(memory)
    if not isinstance(merged_memory.get("key_facts"), list):
        merged_memory["key_facts"] = []
    if not isinstance(merged_memory.get("message_count"), int):
        merged_memory["message_count"] = 0
    return merged_memory


def save_memory(user_id, memory):
    os.makedirs(MEMORIES_DIR, exist_ok=True)
    filename = os.path.join(MEMORIES_DIR, f"memory_{user_id}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)


def build_system_prompt(memory):
    memory_lines = []

    if memory.get("user_name"):
        memory_lines.append(f"- User name: {memory['user_name']}")
    if memory.get("key_facts"):
        memory_lines.append("- Key facts:")
        for fact in memory["key_facts"]:
            if fact:
                memory_lines.append(f"  - {fact}")
    if memory.get("conversation_summary"):
        memory_lines.append(f"- Conversation summary: {memory['conversation_summary']}")
    if memory.get("message_count"):
        memory_lines.append(f"- Message count: {memory['message_count']}")

    if not memory_lines:
        return SYSTEM_PROMPT

    return (
        f"{SYSTEM_PROMPT}\n\n"
        "What you know about this user:\n"
        f"{chr(10).join(memory_lines)}"
    )


def update_memory_from_exchange(user_id, memory, user_message, assistant_message):
    extraction_prompt = """Update the user memory JSON using the latest exchange.
Return JSON only with exactly these keys:
- user_name: string
- key_facts: array of strings
- conversation_summary: string
- message_count: integer

Rules:
- Preserve useful existing facts unless contradicted.
- Only include facts actually supported by the exchange or existing memory.
- Keep key_facts concise and deduplicated.
- Keep conversation_summary short.
- Increment message_count to reflect the new exchange."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=extraction_prompt,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Existing memory:\n{json.dumps(memory, ensure_ascii=True)}\n\n"
                    f"Latest user message:\n{user_message}\n\n"
                    f"Latest assistant reply:\n{assistant_message}"
                )
            }
        ]
    )

    try:
        updated_memory = json.loads(response.content[0].text)
    except (json.JSONDecodeError, IndexError, AttributeError):
        updated_memory = memory.copy()
        updated_memory["message_count"] = memory.get("message_count", 0) + 1

    normalized_memory = {
        "user_name": updated_memory.get("user_name", "") or "",
        "key_facts": updated_memory.get("key_facts", []) or [],
        "conversation_summary": updated_memory.get("conversation_summary", "") or "",
        "message_count": updated_memory.get("message_count", memory.get("message_count", 0) + 1)
    }
    if not isinstance(normalized_memory["key_facts"], list):
        normalized_memory["key_facts"] = []
    if not isinstance(normalized_memory["message_count"], int):
        normalized_memory["message_count"] = memory.get("message_count", 0) + 1

    save_memory(user_id, normalized_memory)
    return normalized_memory

TOOLS = [
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A valid Python math expression e.g. '100 * 0.15'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_current_datetime",
        "description": "Returns the current date and time.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    }
]

def run_tool(tool_name, tool_input):
    if tool_name == "calculate":
        try:
            result = eval(tool_input["expression"], {"math": math, "__builtins__": {}})
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    elif tool_name == "get_current_datetime":
        return datetime.now().strftime("Today is %A, %B %d %Y. Current time: %I:%M %p")

@app.route("/")
def index():
    return open("index.html").read()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    history = data.get("history", [])
    user_id = data.get("user_id", "default")
    memory = load_memory(user_id)
    system_prompt = build_system_prompt(memory)

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    history.append({"role": "user", "content": user_message})

    tool_calls_log = []

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system_prompt,
            tools=TOOLS,
            messages=history
        )

        if response.stop_reason == "tool_use":
            history.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = run_tool(block.name, block.input)
                    tool_calls_log.append(f"{block.name}({block.input}) → {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            history.append({"role": "user", "content": tool_results})
        else:
            final_text = response.content[0].text
            history.append({"role": "assistant", "content": final_text})
            update_memory_from_exchange(user_id, memory, user_message, final_text)
            return jsonify({
                "reply": final_text,
                "history": history,
                "tools_used": tool_calls_log,
                "tokens": response.usage.output_tokens
            })


@app.route("/memory", methods=["GET"])
def get_memory():
    user_id = request.args.get("user_id", "default")
    return jsonify(load_memory(user_id))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True)