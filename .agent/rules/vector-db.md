---
trigger: always_on
---

# Rule: Ephemeral Vector DB Lifecycle
- On session start, the Node Agent MUST re-run the `crawler.py` to ensure the metadata index matches the current file state.
- Before adding points to Qdrant, extract and embed only "Column Metadata" (Name, Type, Description) to prevent sensitive content from entering the vector space.
- When the session ends, explicitly call `client.close()` to ensure the in-memory store is wiped.