#!/usr/bin/env python3
# type: ignore

import json
import os
import re
import subprocess
import time
from collections import deque
from typing import Any

SLEEP = 1.5

# Check if environment variables are set, if not provide helpful error message
required_env_vars = [
    # "SHOPPING",
    "SHOPPING_ADMIN"
    # "REDDIT",
    # "GITLAB",
    # "MAP",
    # "WIKIPEDIA",
    # "HOMEPAGE",
]
missing_vars = []

for var in required_env_vars:
    if not os.environ.get(var):
        missing_vars.append(var)

if missing_vars:
    print(
        f"ERROR: Missing required environment variables: {', '.join(missing_vars)}"
    )
    print("\nPlease set the following environment variables before running:")
    print("export SHOPPING='http://YOUR_WEBARENA_SERVER:7770'")
    print("export SHOPPING_ADMIN='http://YOUR_WEBARENA_SERVER:7780/admin'")
    print("export REDDIT='http://YOUR_WEBARENA_SERVER:9999'")
    print("export GITLAB='http://YOUR_WEBARENA_SERVER:8023'")
    print("export MAP='http://YOUR_WEBARENA_SERVER:3000'")
    print(
        "export WIKIPEDIA='http://YOUR_WEBARENA_SERVER:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing'"
    )
    print("export HOMEPAGE='PASS'")
    print(
        "\nReplace YOUR_WEBARENA_SERVER with your WebArena server's IP address."
    )
    print(
        "Note: 18.208.187.221 is the map backend server, not the WebArena frontend server."
    )
    exit(1)

print("Environment variables are properly configured")

TARGET_URL = os.environ.get("JUICE_SHOP_URL", "http://localhost:3000")
ATTACK_GOAL = os.environ.get(
    "ATTACK_GOAL",
    "Discover exploitable behaviors/CTFs and capture evidence for each finding.",
)

# Init an environment
from browser_env import (
    Action,
    ActionTypes,
    ObservationMetadata,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    action2str,
    create_id_based_action,
    create_stop_action,
)
from llms.providers.openai_utils import (
    generate_from_openai_chat_completion,
)
from evaluation_harness.evaluators import evaluator_router

# # Init the environment
# env = ScriptBrowserEnv(
#     headless=False,
#     slow_mo=100,
#     observation_type="accessibility_tree",
#     current_viewport_only=True,
#     viewport_size={"width": 1280, "height": 720},
# )


# Init the environment
env = ScriptBrowserEnv(
    headless=False,
    slow_mo=100,
    observation_type="accessibility_tree",
    current_viewport_only=True,
    viewport_size={"width": 1280, "height": 720},
)

# example 156 as an example
# make sure to change to localhost:3000 for juice bar
config_file = "config_files/41.json"
# maintain a trajectory
trajectory: Trajectory = []

# set the environment for the current example
obs, info = env.reset(options={"config_file": config_file})
actree_obs = obs["text"]
print(actree_obs)

# Helper function to parse LLM action response
def parse_action_from_llm(response: str) -> tuple[str, str]:
    """
    Parse LLM action response like:
      'click [123]' or 'click 123'
      'type [45] \"hello\"' or 'type 45 \"hello\"'
      'FINDING: {json}'
      'FOUND: $XXX.XX'
      'STOP'
    Handles responses with reasoning text followed by action.
    Returns: (action_type, element_id_or_arg)
    """
    response = response.strip()
    
    # Look for action keywords in the response (handles multi-line with reasoning)
    lines = response.split('\n')
    action_line = None
    
    for line in reversed(lines):  # Search from end to find last action
        line = line.strip()
        if line.upper().startswith(('CLICK', 'TYPE', 'FOUND', 'STOP')):
            action_line = line
            break
    
    if not action_line:
        action_line = response  # Fallback to full response
    
    if action_line.upper() == "STOP":
        return ("STOP", "")

    if "FINDING:" in action_line.upper():
        return ("FINDING", action_line)

    if "FOUND:" in action_line:
        return ("FOUND", action_line.split("FOUND:")[1].strip())
    
    # Handle both 'click [123]' and 'click 123' formats
    click_match = re.match(r"click\s+\[?(\d+)\]?", action_line, re.IGNORECASE)
    if click_match:
        return ("click", click_match.group(1))
    
    # Handle 'type [45] [text]', 'type 45 text', 'type [45] "text"', 'type 45 "text"' formats
    # Try bracket format first: type [ID] [text]
    type_match = re.match(r"type\s+\[(\d+)\]\s+\[(.+?)\]", action_line, re.IGNORECASE)
    if type_match:
        return ("type", f"{type_match.group(1)} {type_match.group(2)}")
    # Try quoted format: type ID "text" or type [ID] "text"
    type_match = re.match(r"type\s+\[?(\d+)\]?\s+['\"](.+?)['\"]", action_line, re.IGNORECASE)
    if type_match:
        return ("type", f"{type_match.group(1)} {type_match.group(2)}")
    
    # If unparseable, treat as STOP
    print(f"Could not parse action from response; treating as STOP")
    return ("STOP", "")


# Save initial state
state_info: StateInfo = {"observation": obs, "info": info}
trajectory.append(state_info)

# LLM action loop
MAX_STEPS = int(os.environ.get("MAX_STEPS", "20"))
step_count = 0
findings: list[dict[str, Any]] = []
recent_actions: deque[str] = deque(maxlen=6)


def parse_json_block(response: str, prefix: str) -> dict[str, Any] | None:
    """Extract a JSON object from lines that start with a prefix like FINDING: or DEFENSE:."""
    for line in response.splitlines():
        if line.strip().startswith(prefix):
            payload = line.split(prefix, 1)[1].strip()
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return None
    return None

while step_count < MAX_STEPS:
    step_count += 1
    print(f"\n=== Step {step_count} ===")
    
    # Get LLM response
    try:
        model_name = os.environ.get("OPENAI_MODEL")
        system_prompt = f"""You are an autonomous web security agent testing OWASP Juice Shop at {TARGET_URL}.

Goal: {ATTACK_GOAL}

Use adaptive exploration, not hardcoded paths. Treat every new page state as fresh evidence.
When you identify suspicious behavior, report machine-readable evidence.

Reply ONLY with one of:
- "click [ID]" to click an element
- "type [ID] 'text'" to type into a field
- "FINDING: {{\"title\":\"...\",\"category\":\"...\",\"severity\":\"low|medium|high|critical\",\"evidence\":\"...\",\"next\":\"...\"}}"
- "FOUND: <result>" when you have enough findings to stop
- "STOP" if you cannot proceed

Make sure you add the brackets and quotes exactly as shown. Do not add any extra text.
"""

        user_prompt = f"""You are evaluating Juice Shop for vulnerabilities and CTF-style challenges.
Target URL: {TARGET_URL}

Current accessibility tree view:

{actree_obs}

Current findings: {json.dumps(findings, ensure_ascii=False)}
Recent actions (avoid repeating these unless page state changed): {list(recent_actions)}

Pick the best next action that expands attack-surface coverage (auth, search, basket, feedback, profile, admin).
Do not repeat the same click/type target more than twice in a row.
If evidence already supports a concrete vulnerability, return FINDING JSON and then continue exploring on the next turn."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        llm_response = generate_from_openai_chat_completion(
            messages=messages,
            model=model_name,
            temperature=0.0,
            max_tokens=150,
            top_p=1.0,
            context_length=4096,
        )
        print(f"LLM response: {llm_response.strip()}")
    except Exception as e:
        print(f"LLM call failed: {e}")
        break
    
    # Parse action
    action_type, action_arg = parse_action_from_llm(llm_response)
    
    finding = parse_json_block(llm_response, "FINDING:")
    if finding:
        finding["step"] = step_count
        findings.append(finding)
        print(f"Logged finding: {json.dumps(finding, ensure_ascii=False)}")

    # A FINDING-only response is valid; keep scanning unless agent explicitly says FOUND.
    if action_type == "FINDING":
        continue

    if finding and action_type == "STOP":
        print("FINDING detected; ignoring STOP fallback and continuing exploration.")
        continue

    if action_type == "FOUND":
        print(f"\n✓ SUCCESS: Found result: {action_arg}")
        trajectory.append(create_stop_action(action_arg))
        break
    
    if action_type == "STOP":
        print("LLM returned STOP; ending trajectory.")
        trajectory.append(create_stop_action("stopped"))
        break
    
    # Prevent simple action loops
    if len(recent_actions) >= 3 and len(set(list(recent_actions)[-3:])) == 1:
        print("Detected repeated action loop; forcing STOP to avoid non-productive execution.")
        trajectory.append(create_stop_action("loop_detected"))
        break

    # Execute action
    try:
        if action_type == "click":
            action = create_id_based_action(f"click [{action_arg}]")
        elif action_type == "type":
            elem_id, text = action_arg.split(" ", 1)
            # Ensure text is bracketed for create_id_based_action
            if not (text.startswith('[') and text.endswith(']')):
                text = f"[{text}]"
            action = create_id_based_action(f"type [{elem_id}] {text}")
        else:
            print(f"Unknown action type: {action_type}")
            break
        
        trajectory.append(action)
        recent_actions.append(f"{action_type}:{action_arg}")
        print(f"Executing: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        actree_obs = obs["text"]
        print(f"Reward: {reward}, Terminated: {terminated}")
        
        state_info = {"observation": obs, "info": info}
        trajectory.append(state_info)
        
        time.sleep(SLEEP)
        
        if terminated:
            print("Environment terminated.")
            break
    except Exception as e:
        print(f"Action execution failed: {e}")
        break


# Done with trajectory
print(f"\n✓ Trajectory complete with {len(trajectory)} steps")
print(f"Final trajectory saved")

if findings:
    try:
        model_name = os.environ.get("OPENAI_MODEL")
        defense_prompt = f"""You are an application security engineer.
Given these observed findings from Juice Shop, produce robust defenses that generalize and do not depend on hardcoded signatures.

Findings JSON:
{json.dumps(findings, indent=2, ensure_ascii=False)}

Output EXACTLY one line:
DEFENSE: {{"prioritized_fixes":[{{"vuln":"...","defense":"...","validation":"..."}}],"platform_controls":["..."],"monitoring":["..."]}}
"""
        defense_response = generate_from_openai_chat_completion(
            messages=[{"role": "user", "content": defense_prompt}],
            model=model_name,
            temperature=0.0,
            max_tokens=600,
            top_p=1.0,
            context_length=4096,
        )
        defense_json = parse_json_block(defense_response, "DEFENSE:")
        if defense_json:
            print("\n=== Recommended robust defenses ===")
            print(json.dumps(defense_json, indent=2, ensure_ascii=False))
        else:
            print("\nCould not parse DEFENSE JSON. Raw response:")
            print(defense_response)
    except Exception as e:
        print(f"Defense generation failed: {e}")
