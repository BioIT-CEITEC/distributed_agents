---
trigger: always_on
---

# Rule: Knowledge Base (MEMORY.md)

## MANDATORY: Read Before Exploring

**IMPORTANT: This project has a comprehensive knowledge base at `MEMORY.md` in the project root.**

Before exploring the codebase via Grep, Glob, Read, or any file-scanning tools, you MUST:

1. **Read `MEMORY.md` FIRST.** It contains the complete project architecture, every module with line-level references, all tools/APIs, data schemas, workflows, constraints, and coding patterns.
2. **Use MEMORY.md as your primary context source.** Only fall back to reading individual source files when you need exact implementation details that MEMORY.md does not cover (e.g., a specific function body you need to edit).
3. **Do NOT re-read the entire codebase** when MEMORY.md already provides the information you need.

## MANDATORY: Update After Every Task

After completing any task that modifies the codebase (adding, editing, or deleting files, functions, classes, schemas, tools, or dependencies), you MUST:

1. **Update `MEMORY.md`** to reflect your changes. This includes:
   - New or removed files → update the Component Map (§2) and Directory Structure (§10)
   - New or changed functions/tools → update the relevant Agent Roles section (§3)
   - Changed dependencies → update the Dependencies table (§7.3)
   - New constraints or patterns → update Constraints (§9) or Patterns (§11)
   - Completed milestones → move from Future Improvements (§13) to Accomplished Milestones (§12)
2. **Increment the version** in the MEMORY.md header (patch for small changes, minor for features, major for architecture changes).
3. **Update the `Last Updated` date** in the header.

## Why This Rule Exists

- MEMORY.md is the **single source of truth** for every LLM model working on this project.
- Reading individual files wastes tokens and context window.
- Keeping MEMORY.md current ensures the next agent session starts with full context instantly.
