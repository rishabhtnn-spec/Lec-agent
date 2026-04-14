from flask import Flask, request, jsonify, session
import anthropic
from datetime import datetime
import math, os, secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are Lec, a sharp product strategist.
You have access to tools — use them when calculations or
current date/time are needed. Think step by step."""

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

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    history.append({"role": "user", "content": user_message})

    tool_calls_log = []

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
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
            return jsonify({
                "reply": final_text,
                "history": history,
                "tools_used": tool_calls_log,
                "tokens": response.usage.output_tokens
            })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True)