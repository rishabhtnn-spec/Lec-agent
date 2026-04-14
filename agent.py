import os
import anthropic
from datetime import datetime
import math

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are Lec, a sharp product strategist. 
You have access to tools — use them when calculations or 
current date/time are needed. Think step by step."""

# --- TOOL DEFINITIONS ---
# This is what you send to Claude so it knows what tools exist
TOOLS = [
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Use this for any arithmetic, percentages, or numeric calculations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A valid Python math expression e.g. '2 + 2' or '100 * 0.15' or 'math.sqrt(144)'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_current_datetime",
        "description": "Returns the current date and time. Use when the user asks anything about today's date, time, day of week, etc.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

# --- TOOL EXECUTION ---
# Your code actually runs the tools here
def run_tool(tool_name, tool_input):
    if tool_name == "calculate":
        try:
            result = eval(tool_input["expression"], {"math": math, "__builtins__": {}})
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    
    elif tool_name == "get_current_datetime":
        now = datetime.now()
        return now.strftime("Today is %A, %B %d %Y. Current time: %I:%M %p")

# --- AGENT LOOP ---
message_history = []
print("Lec ready. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "quit":
        break
    if not user_input:
        continue

    message_history.append({"role": "user", "content": user_input})

    # Agentic loop — keeps going until Claude gives a final text response
    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=message_history
        )

        # Did Claude want to use a tool?
        if response.stop_reason == "tool_use":
            # Add Claude's response (with tool call) to history
            message_history.append({"role": "assistant", "content": response.content})

            # Find and run each tool Claude requested
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  [using tool: {block.name}({block.input})]")
                    result = run_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            # Feed results back to Claude
            message_history.append({"role": "user", "content": tool_results})

        else:
            # Claude gave a final text answer — break inner loop
            final_text = response.content[0].text
            message_history.append({"role": "assistant", "content": final_text})
            print(f"\nLec: {final_text}\n")
            print(f"[tokens: {response.usage.output_tokens}]\n")
            break