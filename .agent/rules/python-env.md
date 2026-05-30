---
trigger: always_on
---

## Instructions
- The virtual environment for this project is located in the parent directory: `/mnt/f/Agentic_AI/Code/`.
- ALWAYS use the absolute path to the Python interpreter: `/mnt/f/Agentic_AI/Code/bin/python`.
- Before running scripts, the agent must check if this interpreter exists.
- DO NOT use system python or `python3` from `/usr/bin/`.

##Target Virtual Environment
- The python virtual environement for this project is located one directory level up from the current working directory

## Terminal Execution
- Prepend all execution commands with the environment path:
  `wsl ../bin/python -u [script_name].py`