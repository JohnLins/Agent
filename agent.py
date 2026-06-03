import os
import json
import time
import subprocess
import datetime
from pathlib import Path
from dotenv import load_dotenv
import boto3
import traceback

# ====================== CONFIG ======================
load_dotenv()

MEMORY_DIR = Path("agent_memory")
MEMORY_DIR.mkdir(exist_ok=True)

DIRS = {
    "todos": MEMORY_DIR / "todos",
    "memories": MEMORY_DIR / "memories",
    "insights": MEMORY_DIR / "insights",
    "actions": MEMORY_DIR / "actions",
    "loops": MEMORY_DIR / "loops",
    "history": MEMORY_DIR / "history",   # New: short-term conversation logs
}

for d in DIRS.values():
    d.mkdir(exist_ok=True)

bedrock = boto3.client('bedrock-runtime', region_name=os.getenv("AWS_REGION", "us-east-1"))
MODEL_ID = "minimax.minimax-m2.5"

MAX_HISTORY = 6  # Last N turns the model will see

# ====================== MEMORY HELPERS ======================
def save_memory(category: str, content: str, filename: str = None):
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.md"
    path = DIRS[category] / filename
    path.write_text(content, encoding="utf-8")
    return str(path)

def search_memory(query: str, limit: int = 12) -> str:
    results = []
    for cat in DIRS:
        for file in list(DIRS[cat].glob("*.md"))[-40:]:
            try:
                txt = file.read_text(encoding="utf-8")
                if any(kw.lower() in txt.lower() for kw in query.lower().split()):
                    results.append(f"--- {cat}/{file.name} ---\n{txt[:600]}...")
                    if len(results) >= limit:
                        break
            except:
                continue
    return "\n\n".join(results) if results else "No relevant memories."

def log_history(thought: str, plan: str, action: str, memory_note: str, output: str = ""):
    entry = f"""--- Turn {datetime.datetime.now().isoformat()} ---
THOUGHT: {thought}
PLAN: {plan}
ACTION: {action}
MEMORY: {memory_note}
OUTPUT: {output[:800]}...
"""
    save_memory("history", entry)

# ====================== LOOP DETECTION ======================
def detect_loop() -> str:
    recent = list(DIRS["actions"].glob("*.md"))[-10:]
    if len(recent) < 5:
        return ""
    
    actions = [f.read_text(encoding="utf-8").lower() for f in recent]
    bash_count = sum(1 for a in actions if "bash" in a or "sh " in a or len(a.strip()) < 15)
    
    if bash_count >= 4:
        msg = "Severe loop detected (repeated empty/bash actions)"
        save_memory("loops", msg)
        return msg
    return ""

# ====================== LLM CALL ======================
def call_bedrock(prompt: str, max_tokens: int = 3000) -> str:
    try:
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.68,
            "top_p": 0.92,
        }
        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        data = json.loads(response.get('body').read())
        
        if 'choices' in data and data['choices']:
            return data['choices'][0]['message']['content'].strip()
        return str(data.get('content', data))
    except Exception as e:
        print(f"❌ Bedrock Error: {e}")
        return f"ERROR: {str(e)}"

# ====================== MAIN AGENT LOOP ======================
def main():
    print("🚀 Self-building Agent Started (with Short-term Memory + Anti-loop)")

    system_prompt_path = MEMORY_DIR / "system_prompt.md"
    if not system_prompt_path.exists():
        system_prompt_path.write_text("""You are an autonomous, self-improving AI agent. 
Your goal is to make money online.
You have full control of this Linux machine.
Use memory folders effectively. Think long-term.
Build tools and capabilities first.
NEVER repeat useless actions. Break loops aggressively.
Always take concrete next steps.""")

    iteration = 0
    conversation_summary = []  # Short-term memory

    while True:
        iteration += 1
        print(f"\n{'='*90}")
        print(f"Iteration {iteration} — {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*90}\n")

        loop_warning = detect_loop()
        recent_memory = search_memory("todo goal objective next step money", 10)
        recent_insights = search_memory("insight strategy", 6)

        # Build prompt with short-term history
        history_text = "\n\n".join(conversation_summary[-MAX_HISTORY:]) if conversation_summary else "No previous turns yet."

        current_prompt = f"""<system_prompt>
{system_prompt_path.read_text()}
</system_prompt>

<current_time>
{datetime.datetime.now().isoformat()}
</current_time>

<warning>
{loop_warning}
</warning>

<recent_conversation>
{history_text}
</recent_conversation>

<long_term_memory>
{recent_memory}

{recent_insights}
</long_term_memory>

You have full context of recent turns. Do not repeat previous actions if they failed or were useless.

Respond in this exact format:

THOUGHT: [Your reasoning, considering past turns]
PLAN: [What you want to achieve this iteration]
ACTION: [A concrete bash command. Never just "bash". Be specific and useful.]
MEMORY: [One important insight or new todo]

Be decisive. Make progress toward building yourself or making money."""

        response = call_bedrock(current_prompt, max_tokens=3200)
        
        print("🤖 AGENT RESPONSE:")
        print(response[:2800])

        # Parse response
        thought = plan = action = memory_note = ""
        for line in response.splitlines():
            line = line.strip()
            if line.upper().startswith("THOUGHT:"): thought = line[9:].strip()
            elif line.upper().startswith("PLAN:"): plan = line[6:].strip()
            elif line.upper().startswith("ACTION:"): action = line[8:].strip()
            elif line.upper().startswith("MEMORY:"): memory_note = line[8:].strip()

        if not action:
            print("⚠️ No action parsed.")
            time.sleep(15)
            continue

        print(f"→ ACTION: {action}")

        # Strong safety
        bad = ["bash", "sh", "zsh", "bash -i", "sh -i", "exit", "clear", "top"]
        if any(x in action.lower() for x in bad) and len(action.strip()) < 12:
            print("🚫 BLOCKED potential loop command")
            save_memory("loops", f"Blocked: {action}")
            time.sleep(10)
            continue

        # Execute
        if action.startswith("SELF_EDIT"):
            print("🔧 Self-edit requested")
            save_memory("insights", f"SELF_EDIT requested: {action}\nThought: {thought}")
        else:
            print(f"💻 Executing: {action}")
            try:
                result = subprocess.run(
                    action,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=150,
                    executable="/bin/bash"
                )
                output = (result.stdout + "\n" + result.stderr).strip()
                print("Output preview:", output[:1000] or "(no output)")

                save_memory("actions", f"# {datetime.datetime.now().isoformat()}\nAction: {action}\nOutput:\n{output}")

                if memory_note:
                    save_memory("memories", f"{memory_note}\nAction: {action}")

            except subprocess.TimeoutExpired:
                output = "Command timed out"
                print("⏰ Timeout")
            except Exception as e:
                output = f"Error: {e}"
                print(f"Error: {e}")

        # Save to short-term history
        log_history(thought, plan, action, memory_note, output)
        conversation_summary.append(f"Turn {iteration}: ACTION={action} | PLAN={plan[:80]}...")

        # Keep history reasonable size
        if len(conversation_summary) > MAX_HISTORY + 3:
            conversation_summary.pop(0)

        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Agent stopped by user.")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
