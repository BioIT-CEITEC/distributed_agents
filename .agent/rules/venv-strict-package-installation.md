---
trigger: always_on
---

Trigger: When asked to install, update, or manage any Python packages, libraries, or dependencies.
Context: The project operates inside a strict virtual environment, requiring isolated dependency management.
Instruction:
 
"You are strictly forbidden from using global package managers like pip, pip3, or apt-get to install Python libraries. You must ALWAYS use the pip executable located within the virtual environment's binary directory.
 
Command Pattern:
../bin/pip install <package_name>
 
Strict Constraints:
 
Do not use python -m pip unless specifying the venv Python (../bin/python3 -m pip).
 
If a requirements.txt file is mentioned, use ../bin/pip install -r requirements.txt.
 
Never modify the global system environment."