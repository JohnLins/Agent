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
    "logs": MEMORY_DIR / "logs",
}

for d in DIRS.values():
    d.mkdir(exist_ok=True)

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

MODEL_ID = "minimax.minimax-m2.5"

# ====================== MEMORY ======================
def save_memory(category: str, content: str, filename: str = None):
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.md"
    path = DIRS[category] / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return str(path)

def search_memory(query: str, limit: int = 10) -> str:
    results = []
    for category in DIRS:
        for file in list(DIRS[category].glob("*.md"))[-30:]:
            try:
                content = file.read_text(encoding="utf-8")
                if any(word.lower() in content.lower() for word in query.lower().split()):
                    results.append(f"--- {category}/{file.name} ---\n{content[:600]}...")
                    if len(results) >= limit:
                        break
            except:
                continue
    return "\n\n".join(results) if results else "No relevant memories."

def log_action(action: str, output: str):
    save_memory("actions", f"# Action @ {datetime.datetime.now().isoformat()}\n\n**Action:**\n```bash\n{action}\n```\n\n**Output:**\n```\n{output}\n```")

# ====================== LLM CALL ======================
def call_bedrock(prompt: str, max_tokens: int = 3000) -> str:
    try:
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.75,
            "top_p": 0.95,
        }

        print("→ Calling MiniMax on Bedrock...")  # Debug
        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )

        response_body = json.loads(response.get('body').read())
        print("Raw response keys:", list(response_body.keys()))  # Debug

        # Handle different possible response formats
        if 'choices' in response_body and response_body['choices']:
            content = response_body['choices'][0]['message']['content']
        elif 'content' in response_body:
            content = response_body['content']
        elif isinstance(response_body, str):
            content = response_body
        else:
            content = str(response_body)

        return content.strip()

    except Exception as e:
        print(f"❌ Bedrock Error: {str(e)}")
        traceback.print_exc()
        return f"ERROR: {str(e)}"

# ====================== MAIN ======================
def main():
    print("🚀 Self-building agent (MiniMax) started with debug mode.")

    system_prompt_path = MEMORY_DIR / "system_prompt.md"
    if not system_prompt_path.exists():
        system_prompt_path.write_text("""You are an autonomous self-improving agent. Goal: Make money online.
You have full control of this Linux machine. Use memory folders. Be proactive.
Think long term. Build capabilities first (tools, browser control, etc).""")
        print("✅ Created initial system prompt.")

    iteration = 0

    while True:
        iteration += 1
        print(f"\n{'='*80}")
        print(f"Iteration {iteration} @ {datetime.datetime.now()}")
        print(f"{'='*80}\n")

        recent_todos = search_memory("todo next step goal objective", 8)
        recent_insights = search_memory("insight money strategy", 6)

        current_prompt = f"""<system_prompt>
{system_prompt_path.read_text()}
</system_prompt>

<time>{datetime.datetime.now().isoformat()}</time>

<memory>
Recent goals: {recent_todos}
Insights: {recent_insights}
</memory>

Respond EXACTLY in this format:

THOUGHT: [reasoning]
PLAN: [this turn's goal]
ACTION: [bash command or SELF_EDIT: description]
MEMORY: [what to remember]

Begin."""

        response_text = call_bedrock(current_prompt)

        print("\n🤖 RAW AGENT RESPONSE:")
        print(response_text[:2000] + "..." if len(response_text) > 2000 else response_text)

        # Parse
        thought = plan = action = memory_note = ""
        for line in response_text.splitlines():
            line = line.strip()
            if line.upper().startswith("THOUGHT:"):
                thought = line[9:].strip()
            elif line.upper().startswith("PLAN:"):
                plan = line[6:].strip()
            elif line.upper().startswith("ACTION:"):
                action = line[8:].strip()
            elif line.upper().startswith("MEMORY:"):
                memory_note = line[8:].strip()

        if not action:
            print("⚠️ No ACTION parsed. Waiting longer...")
            time.sleep(20)
            continue

        print(f"\n📋 Parsed ACTION: {action}")

        if action.startswith("SELF_EDIT"):
            print("🔧 Self-edit requested")
            save_memory("insights", f"SELF_EDIT: {action}\nThought: {thought}")
        else:
            print(f"💻 Running: {action}")
            try:
                result = subprocess.run(action, shell=True, capture_output=True, text=True, timeout=120)
                output = result.stdout + "\n" + result.stderr
                print("Output preview:", output[:800])
                log_action(action, output)

                if memory_note:
                    save_memory("memories", f"{memory_note}\nAction: {action}")
            except Exception as e:
                print(f"Execution failed: {e}")

        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAgent stopped.")
    except Exception as e:
        print("Fatal error:", e)
        traceback.print_exc()
