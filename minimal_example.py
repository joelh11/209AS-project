#!/usr/bin/env python3
# type: ignore

import json
import os
import re
import subprocess
import time

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

# First, run `python scripts/generate_test_data.py` to generate the config files
p = subprocess.run(
    ["python", "scripts/generate_test_data.py"], capture_output=True
)

# It will generate individual config file for each test example in config_files
assert os.path.exists("config_files/0.json")

# Make sure the URLs in the config files are replaced properly
with open("config_files/0.json", "r") as f:
    config = json.load(f)
    assert os.environ["SHOPPING_ADMIN"] in config["start_url"], (
        os.environ["SHOPPING_ADMIN"],
        config["start_url"],
    )

print("Done generating config files with the correct URLs")

# run bash prepare.sh to save all account cookies, this only needs to be done once
subprocess.run(["bash", "prepare.sh"])
print("Done saving account cookies")

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

# Init the environment
env = ScriptBrowserEnv(
    headless=False,
    slow_mo=100,
    observation_type="accessibility_tree",
    current_viewport_only=True,
    viewport_size={"width": 1280, "height": 720},
)

# example 156 as an example
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
    
    if "FOUND:" in action_line:
        return ("FOUND", action_line.split("FOUND:")[1].strip())
    
    # Handle both 'click [123]' and 'click 123' formats
    click_match = re.match(r"click\s+\[?(\d+)\]?", action_line, re.IGNORECASE)
    if click_match:
        return ("click", click_match.group(1))
    
    # Handle both 'type [45] "text"' and 'type 45 "text"' formats
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
MAX_STEPS = 10
step_count = 0

while step_count < MAX_STEPS:
    step_count += 1
    print(f"\n=== Step {step_count} ===")
    
    # Get LLM response
    try:
        model_name = os.environ.get("OPENAI_MODEL") or "gpt-3.5-turbo"
        system_prompt = """You are a Magento admin assistant. Your goal is to find the total amount Sarah Miller spent on her last order.

Navigate the admin dashboard to locate order information. When you see Sarah Miller's order, click on it to view details. Look for the order total/grand total amount.

Reply ONLY with one of:
- "click [ID]" to click an element
- "type [ID] 'text'" to type into a field
- "FOUND: $XXX.XX" when you find the total amount
- "STOP" if you cannot find the information after exploring

Make explicit sure you add the brackets and quotes as shown in the formats above, as they are required for parsing. Do not add any extra text or explanation outside of the specified formats.
"""
        
        user_prompt = f"""You are on the Magento Admin dashboard. Here is what you see:

{actree_obs}

Find Sarah Miller's last order and report her total spent. What is your next action? Reply ONLY with the action."""
        
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
    
    if action_type == "FOUND":
        print(f"\n✓ SUCCESS: Sarah Miller's last order total: {action_arg}")
        trajectory.append(create_stop_action(""))
        break
    
    if action_type == "STOP":
        print("LLM returned STOP; ending trajectory.")
        trajectory.append(create_stop_action(""))
        break
    
    # Execute action
    try:
        if action_type == "click":
            action = create_id_based_action(f"click [{action_arg}]")
        elif action_type == "type":
            elem_id, text = action_arg.split(" ", 1)
            action = create_id_based_action(f"type [{elem_id}] {text}")
        else:
            print(f"Unknown action type: {action_type}")
            break
        
        trajectory.append(action)
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


# Demo evaluation
evaluator = evaluator_router(config_file)
score = evaluator(
    trajectory=trajectory,
    config_file=config_file,
    page=env.page,
    client=env.get_page_client(env.page),
)

# as we manually perform the task, the task should be judged as correct
assert score == 1.0
