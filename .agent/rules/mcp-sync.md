# Rule: mcp-sync
- **Pre-flight Requirement**: You MUST execute the `recall_memories` tool before starting any task to read the recent modifications made to the codebase.
- **Post-flight Requirement**: You MUST execute the `store_memory` tool immediately after completing a code change or structural modification.
- **Identity Tagging**: Every memory you store MUST include a tag identifying your specific model name (e.g., Gemini_Pro or Claude_Opus).
- **Payload Details**: The stored memory must explicitly state the files modified, the core logic altered, and the architectural reasoning.
