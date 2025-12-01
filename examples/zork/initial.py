"""Zork agent scaffold for ShinkaEvolve evolution"""

# EVOLVE-BLOCK-START
# All agent logic lives here - must define predict(observation: str) -> str
# The observation format from Jericho/MIND API is:
#   <game text>
#   
#   Available commands: action1 | action2 | action3 | ...

import random

def predict(observation: str) -> str:
    """
    Predict next action based on current game observation.
    
    Args:
        observation: Current game state including description and valid actions
        
    Returns:
        Single action string to execute
    """
    # Persistent state using function attribute
    if not hasattr(predict, 'state'):
        predict.state = {
            "visited": set(),
            "inventory": set(),
            "moves": [],
            "treasures_found": 0,
            "step": 0
        }
    
    state = predict.state
    state["step"] += 1
    
    # Parse observation
    lines = observation.strip().split('\n')
    obs_lower = observation.lower()
    
    # Extract valid actions (format: "Available commands: north | south | east")
    valid_actions = []
    for line in lines:
        if 'Available commands:' in line and '|' in line:
            actions_part = line.split(':', 1)[1]
            valid_actions = [a.strip() for a in actions_part.split('|')]
            break
    
    if not valid_actions:
        return "look"
    # Ultra-optimized treasure hunter baseline (scores 10-20 points)
    
    # Persistent state
    if not hasattr(predict, 'state'):
        predict.state = {
            "visited": set(),
            "inventory": set(),
            "moves": [],
            "treasures_found": 0,
            "step": 0
        }
    
    state = predict.state
    state["step"] += 1
    
    obs_lower = observation.lower()
    
    # Track rooms
    for line in lines:
        if 5 < len(line) < 50 and line and line[0].isupper() and not line.startswith('>'):
            state["visited"].add(line.strip()[:30])
            break
    
    # Track inventory
    if "taken" in obs_lower or "carrying" in obs_lower or "you are carrying" in obs_lower:
        for item in ["lamp", "egg", "jewel", "painting", "coins", "chalice", 
                     "trident", "bauble", "candles", "torch", "diamond", 
                     "ruby", "emerald", "gold", "silver", "sceptre", "sword"]:
            if item in obs_lower:
                state["inventory"].add(item)
    
    # Remember moves
    state["moves"].append(valid_actions[0] if valid_actions else "")
    if len(state["moves"]) > 30:
        state["moves"] = state["moves"][-30:]
    
    # ULTRA PRIORITY: Deposit treasures in trophy case
    if "trophy" in obs_lower or "case" in obs_lower or "glass case" in obs_lower:
        treasure_list = ["egg", "jewel", "painting", "coins", "chalice", 
                        "trident", "bauble", "candles", "torch", "diamond",
                        "ruby", "emerald", "gold", "silver", "sceptre"]
        for treasure in treasure_list:
            if treasure in state["inventory"]:
                for action in valid_actions:
                    if ("put" in action or "drop" in action) and treasure in action:
                        state["inventory"].discard(treasure)
                        return action
    
    # HIGHEST PRIORITY: Take treasures (this is what scores points!)
    treasure_keywords = ["egg", "jewel", "painting", "coins", "chalice",
                        "trident", "bauble", "candles", "torch", "diamond",
                        "ruby", "emerald", "gold", "silver", "sceptre"]
    
    for treasure in treasure_keywords:
        for action in valid_actions:
            if treasure in action and ("take" in action or "get" in action):
                state["treasures_found"] += 1
                return action
    
    # HIGH PRIORITY: Get lamp early (needed for exploration)
    if "lamp" not in state["inventory"] and state["step"] < 15:
        if "open mailbox" in valid_actions:
            return "open mailbox"
        for action in valid_actions:
            if "lamp" in action and ("take" in action or "get" in action):
                return action
    
    # Turn on lamp
    if "lamp" in state["inventory"]:
        for action in valid_actions:
            if "turn on" in action and "lamp" in action:
                return action
    
    # MEDIUM PRIORITY: Open containers (might have treasures)
    if random.random() < 0.6:
        open_actions = [a for a in valid_actions if "open" in a and "mailbox" not in a]
        if open_actions:
            return random.choice(open_actions)
    
    # EXPLORATION: Smart movement avoiding backtracking
    opposites = {
        "north": "south", "south": "north",
        "east": "west", "west": "east",
        "up": "down", "down": "up"
    }
    
    recent_move = state["moves"][-1] if state["moves"] else ""
    avoid_direction = opposites.get(recent_move, "")
    
    # Prioritize certain directions early
    if state["step"] < 20:
        priority_dirs = ["west", "north", "east", "down", "south", "up"]
    else:
        priority_dirs = ["north", "east", "south", "west", "down", "up"]
    
    random.shuffle(priority_dirs)
    
    for direction in priority_dirs:
        if direction in valid_actions and direction != avoid_direction:
            return direction
    
    # Default: random valid action
    return random.choice(valid_actions)
# EVOLVE-BLOCK-END
