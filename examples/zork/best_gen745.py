"""Minimal Zork agent scaffold for ShinkaEvolve evolution"""

# EVOLVE-BLOCK-START
import random
import re
from collections import defaultdict, deque

# ----------------------------------------------------------------------
# Helper parsing utilities
# ----------------------------------------------------------------------
def _parse_observation(observation: str):
    """
    Split the raw observation into:
      - description: the main room text (first non‑empty line not starting with '>')
      - actions: list of strings after the "Available commands:" line
    """
    lines = [l.strip() for l in observation.strip().split("\n") if l.strip()]
    description = ""
    actions = []

    # Find description (first line that looks like a room title)
    for line in lines:
        if line.lower().startswith("available commands"):
            continue
        if line and not line.startswith(">"):
            description = line
            break

    # Extract actions
    for line in lines:
        if "available commands:" in line.lower():
            # everything after ':' contains the actions separated by '|'
            parts = line.split(":", 1)[1]
            actions = [a.strip() for a in parts.split("|") if a.strip()]
            break

    return description, actions

def _extract_inventory(observation: str):
    """
    Very simple heuristic: look for known treasure / item words in the
    observation text that indicate the player is carrying them.
    """
    inventory_items = set()
    inventory_keywords = [
        "lamp", "egg", "jewel", "painting", "coins", "chalice",
        "trident", "bauble", "candles", "torch", "diamond",
        "ruby", "emerald", "gold", "silver", "sceptre", "sword"
    ]
    text = observation.lower()
    for word in inventory_keywords:
        if re.search(rf"\b{word}\b", text):
            # Check for phrases indicating possession
            if any(p in text for p in ["you are carrying", "carrying", "taken", "have"]):
                inventory_items.add(word)
    return inventory_items

def _detect_room_name(description: str):
    """
    Heuristic room identifier – first line up to 30 chars.
    """
    return description[:30].strip().lower() if description else ""

# ----------------------------------------------------------------------
# Core decision logic
# ----------------------------------------------------------------------
def _choose_deposit(state, actions):
    """
    If we are in a trophy case / display area, drop any treasure we hold.
    """
    deposit_triggers = ["trophy", "case", "glass case"]
    if any(tok in state["last_obs"].lower() for tok in deposit_triggers):
        for treasure in state["inventory"]:
            for act in actions:
                if ("put" in act or "drop" in act) and treasure in act:
                    state["inventory"].discard(treasure)
                    return act
    return None

def _choose_take(state, actions):
    treasure_words = [
        "egg", "jewel", "painting", "coins", "chalice",
        "trident", "bauble", "candles", "torch", "diamond",
        "ruby", "emerald", "gold", "silver", "sceptre"
    ]
    for word in treasure_words:
        for act in actions:
            if word in act and ("take" in act or "get" in act):
                state["inventory"].add(word)
                state["treasures_found"] += 1
                return act
    return None

def _choose_lamp_actions(state, actions):
    # Acquire lamp early
    if "lamp" not in state["inventory"] and state["step"] < 20:
        for act in actions:
            if "open mailbox" == act:
                return act
        for act in actions:
            if "lamp" in act and ("take" in act or "get" in act):
                return act

    # Turn it on if we have it but it's not lit yet
    if "lamp" in state["inventory"]:
        for act in actions:
            if "turn on" in act and "lamp" in act:
                return act
    return None

def _choose_open_container(state, actions):
    # Prefer opening containers that are not mailboxes (already handled)
    container_actions = [a for a in actions if "open" in a and "mailbox" not in a]
    if container_actions:
        # Random but weighted towards unexplored rooms (simple heuristic)
        return random.choice(container_actions)
    return None

def _choose_explore(state, actions):
    """
    Depth‑first exploration using a stack of pending directions.
    Avoid immediate backtracking and prefer unvisited directions.
    """
    opposites = {
        "north": "south", "south": "north",
        "east": "west", "west": "east",
        "up": "down", "down": "up"
    }

    # Record the direction we just came from (if any) to avoid immediate reversal
    avoid = opposites.get(state["last_move"], "")

    # If we have a planned path (stack), follow it
    while state["explore_stack"]:
        nxt = state["explore_stack"][-1]
        if nxt in actions and nxt != avoid:
            return nxt
        else:
            state["explore_stack"].pop()

    # No planned path – generate new ordering
    dirs = [d for d in ["north", "east", "south", "west", "up", "down"]
            if d in actions and d != avoid]

    # Shuffle for some variability but keep deterministic ordering early on
    if state["step"] < 30:
        priority = ["west", "north", "east", "down", "south", "up"]
        dirs.sort(key=lambda d: priority.index(d) if d in priority else 99)
    else:
        random.shuffle(dirs)

    # Push remaining directions onto stack for future back‑tracking
    for d in reversed(dirs[1:]):  # keep the first as immediate move
        state["explore_stack"].append(d)

    return dirs[0] if dirs else None

def _fallback_random(actions):
    return random.choice(actions) if actions else "look"

# ----------------------------------------------------------------------
# Main predict function
# ----------------------------------------------------------------------
def predict(observation: str) -> str:
    """
    Zork agent: receives the full observation string (room description +
    list of valid commands) and returns a single action to execute.
    """
    # ------------------------------------------------------------------
    # Persistent state initialisation
    # ------------------------------------------------------------------
    if not hasattr(predict, "state"):
        predict.state = {
            "visited": set(),
            "inventory": set(),
            "moves": deque(maxlen=30),
            "treasures_found": 0,
            "step": 0,
            "last_move": "",          # last direction taken
            "explore_stack": [],      # directions to explore later
            "last_obs": ""            # raw text of the previous observation
        }

    state = predict.state
    state["step"] += 1
    state["last_obs"] = observation

    # ------------------------------------------------------------------
    # Parse observation
    # ------------------------------------------------------------------
    description, actions = _parse_observation(observation)

    # Update visited rooms
    room_id = _detect_room_name(description)
    if room_id:
        state["visited"].add(room_id)

    # Update inventory from textual hints
    state["inventory"].update(_extract_inventory(observation))

    # ------------------------------------------------------------------
    # Decision pipeline (high‑to‑low priority)
    # ------------------------------------------------------------------
    # 1. Deposit treasures if we are at a case
    act = _choose_deposit(state, actions)
    if act:
        return act

    # 2. Take any visible treasure
    act = _choose_take(state, actions)
    if act:
        return act

    # 3. Acquire / turn on lamp
    act = _choose_lamp_actions(state, actions)
    if act:
        return act

    # 4. Open containers (mailbox already covered)
    act = _choose_open_container(state, actions)
    if act:
        return act

    # 5. Exploration – move to new rooms while avoiding loops
    act = _choose_explore(state, actions)
    if act:
        # Record move for back‑tracking avoidance
        state["last_move"] = act
        state["moves"].append(act)
        return act

    # 6. Fallback random action (should rarely be hit)
    return _fallback_random(actions)

# EVOLVE-BLOCK-END
