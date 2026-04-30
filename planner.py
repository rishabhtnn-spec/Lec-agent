import anthropic
import json

client = anthropic.Anthropic()

PLANNER_PROMPT = """You are a strategic planner. When given a goal,
break it into 3-5 clear, actionable steps.

Respond ONLY with valid JSON in this exact format:
{
  "goal": "the original goal",
  "steps": [
    {"step": 1, "action": "what to do", "why": "why this step matters"},
    {"step": 2, "action": "what to do", "why": "why this step matters"}
  ]
}

No extra text. No markdown. Just the JSON."""

EXECUTOR_PROMPT = """You are a precise executor. You receive one step
from a plan and carry it out by producing a concrete, specific output.

Be direct and practical. Produce real output — not meta-commentary
about what you would do. Actually do it."""

def plan(goal: str) -> dict:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=PLANNER_PROMPT,
        messages=[{"role": "user", "content": f"Goal: {goal}"}]
    )
    raw = response.content[0].text
    return json.loads(raw)

def execute_step(step: dict, goal: str, previous_outputs: list) -> str:
    context = ""
    if previous_outputs:
        context = "\n\nPrevious steps completed:\n" + \
            "\n".join([f"Step {i+1}: {o}" for i, o in enumerate(previous_outputs)])

    prompt = f"""Overall goal: {goal}
{context}

Your current step:
Action: {step['action']}
Why it matters: {step['why']}

Execute this step now. Produce concrete output."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=EXECUTOR_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def run_plan(goal: str) -> dict:
    print(f"\n[Planner] Breaking down: {goal}")
    plan_data = plan(goal)

    results = []
    previous_outputs = []

    for step in plan_data["steps"]:
        print(f"[Executor] Running step {step['step']}: {step['action']}")
        output = execute_step(step, goal, previous_outputs)
        previous_outputs.append(output)
        results.append({
            "step": step["step"],
            "action": step["action"],
            "output": output
        })

    return {
        "goal": goal,
        "plan": plan_data["steps"],
        "results": results
    }
    