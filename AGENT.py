import json
import os
import time
import subprocess
import threading
import signal
import boto3
from dotenv import load_dotenv
# =========================
# CONFIG
# =========================

STATE_FILE = "state.json"
INPUT_FILE = "input.txt"

MODEL_V3 = "deepseek.v3.2"
MODEL_R1 = "deepseek.r1-v1:0"

LOOP_SLEEP = 1.0
MAX_RECENT_EVENTS = 25
REFLECTION_INTERVAL = 10



load_dotenv()



bedrock = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

# =========================
# STATE
# =========================

def load_state():
    if not os.path.exists(STATE_FILE):
        return {
            "goal": None,
            "plan": "",
            "task": "",
            "beliefs": "",
            "summary": "",
            "recent_events": [],
            "iteration": 0,
            "pending_human": []
        }
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# =========================
# HUMAN INPUT
# =========================

def read_input_file():
    if not os.path.exists(INPUT_FILE):
        return None
    with open(INPUT_FILE, "r") as f:
        content = f.read().strip()
    if content:
        open(INPUT_FILE, "w").close()
        return content
    return None

# =========================
# PROCESS CONTROL
# =========================

current_process_group = None

def kill_current_job():
    global current_process_group
    if current_process_group:
        try:
            os.killpg(current_process_group, signal.SIGTERM)
        except:
            pass
        current_process_group = None

# =========================
# LLM CALL
# =========================

def call_bedrock(model_id, prompt):
    response = bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps({
            "inputText": prompt
        })
    )
    output = json.loads(response["body"].read())
    return output["outputText"]

# =========================
# JSON PARSER
# =========================

def safe_json_load(text):
    try:
        return json.loads(text)
    except:
        return {
            "thought": "parse_error",
            "plan": "",
            "current_task": "",
            "actions": []
        }

# =========================
# PROMPT BUILDER
# =========================

def build_prompt(state):
    return f"""
You are an autonomous agent.

GOAL:
{state['goal']}

PLAN:
{state['plan']}

CURRENT TASK:
{state['task']}

BELIEFS:
{state['beliefs']}

SUMMARY:
{state['summary']}

RECENT EVENTS:
{json.dumps(state['recent_events'][-MAX_RECENT_EVENTS:], indent=2)}

HUMAN INPUT:
{json.dumps(state['pending_human'], indent=2)}

Return ONLY valid JSON:
{{
  "thought": "...",
  "plan": "...",
  "current_task": "...",
  "actions": [
    {{
      "type": "bash",
      "command": "..."
    }}
  ],
  "summary_update": "..."
}}
"""

# =========================
# EXECUTION
# =========================

def run_bash(command):
    global current_process_group

    process = subprocess.Popen(
        command,
        shell=True,
        preexec_fn=os.setsid,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    current_process_group = os.getpgid(process.pid)

    stdout, stderr = process.communicate()

    current_process_group = None

    return {
        "command": command,
        "stdout": stdout[-5000:],
        "stderr": stderr[-5000:],
        "exit_code": process.returncode
    }

def execute_actions(actions, state):
    events = []

    for a in actions:
        if a["type"] == "bash":
            result = run_bash(a["command"])
            events.append(result)
            state["recent_events"].append(result)

    return events

# =========================
# REFLECTION
# =========================

def reflect(state):
    prompt = f"""
You are a reflection system.

GOAL:
{state['goal']}

SUMMARY:
{state['summary']}

RECENT EVENTS:
{json.dumps(state['recent_events'][-50:], indent=2)}

Is the agent making progress?

Return JSON:
{{
  "analysis": "...",
  "should_replan": true/false,
  "new_plan": "..."
}}
"""
    raw = call_bedrock(MODEL_R1, prompt)
    return safe_json_load(raw)

# =========================
# MAIN LOOP
# =========================

def main():
    state = load_state()

    while True:
        state["iteration"] += 1

        # -------------------------
        # HUMAN INPUT
        # -------------------------
        msg = read_input_file()
        if msg:
            if msg.startswith("STOP"):
                kill_current_job()
                state["pending_human"].append(msg)

            else:
                state["pending_human"].append(msg)

        # -------------------------
        # GOAL HANDLING
        # -------------------------
        if not state["goal"]:
            state["goal"] = input("Enter new goal: ")
            save_state(state)
            continue

        # -------------------------
        # REFLECTION
        # -------------------------
        if state["iteration"] % REFLECTION_INTERVAL == 0:
            r = reflect(state)
            if r.get("should_replan"):
                state["plan"] = r.get("new_plan", state["plan"])

        # -------------------------
        # BUILD PROMPT
        # -------------------------
        prompt = build_prompt(state)

        raw = call_bedrock(MODEL_V3, prompt)
        result = safe_json_load(raw)

        # -------------------------
        # UPDATE STATE
        # -------------------------
        state["plan"] = result.get("plan", state["plan"])
        state["task"] = result.get("current_task", state["task"])

        if result.get("summary_update"):
            state["summary"] += "\n" + result["summary_update"]

        # clear human input after consumption
        state["pending_human"] = []

        # -------------------------
        # EXECUTE
        # -------------------------
        actions = result.get("actions", [])
        execute_actions(actions, state)

        # -------------------------
        # MEMORY COMPRESSION (simple)
        # -------------------------
        if len(state["recent_events"]) > 100:
            state["summary"] += "\n" + str(state["recent_events"][:50])
            state["recent_events"] = state["recent_events"][-50:]

        # -------------------------
        # GOAL COMPLETION CHECK
        # -------------------------
        if "goal_complete" in result and result["goal_complete"]:
            print("Goal complete.")
            state["goal"] = None
            state["plan"] = ""
            state["task"] = ""
            state["pending_human"] = []
            save_state(state)
            continue

        save_state(state)
        time.sleep(LOOP_SLEEP)

if __name__ == "__main__":
    main()
