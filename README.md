# BioAgents: Distributed Genetic Variant Analysis System

A privacy-preserving distributed system for rare disease genetic variant analysis using Fisher's exact test. Built with pydantic-ai agents that coordinate across multiple clinical center nodes without sharing patient-level data.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    User (Natural Language)                       │
│        "Is CFTR rs113993960 associated with Cystic Fibrosis?"   │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Orchestrator Agent (GPT-4o)                    │
│  • Interprets queries • Broadcasts to nodes • Aggregates results │
└─────────┬─────────────┬─────────────┬─────────────┬─────────────┘
          ▼             ▼             ▼             ▼
     ┌────────┐    ┌────────┐   ┌────────┐    ┌────────┐
     │ Node 1 │    │ Node 2 │   │ Node 3 │    │ Node 4 │
     │ :5001  │    │ :5002  │   │ :5003  │    │ :5004  │
     └────────┘    └────────┘   └────────┘    └────────┘
       3000 pts      3000 pts     3000 pts      3000 pts
```

**Key Features:**
- Natural language queries for genetic variant analysis
- Privacy-preserving: only aggregated statistics shared (never patient records)
- Fisher's exact test with meta-analysis across nodes
- Co-occurrence analysis for variant combinations

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pydantic-ai pydantic pandas numpy scipy flask httpx python-dotenv openai
```

### 2. Configure API Key

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=sk-your-api-key-here
```

Or export directly:

```bash
export OPENAI_API_KEY=sk-your-api-key-here
```

> ⚠️ **Never commit your API key to git.** Add `.env` to your `.gitignore`.

### 3. Generate Sample Data

```bash
python generate_patient_data.py
```

This creates:
- `patients_node1-4.csv` — Synthetic patient cohorts (3000 patients each)
- `variant_metadata.json` — Variant information (30 variants, 6 diseases)

### 4. Run the System

**Basic Mode** (all nodes equal, aggregated stats only):
```bash
python launch_agentic.py
```

**Extended Mode** (home node + investigation loop):
```bash
python launch_extended.py
```

### 5. Ask Questions

```
Your question: Is CFTR rs113993960 significantly associated with Cystic Fibrosis?

📊 RESPONSE
--------------------------------------------------
💡 Interpretation: Very strong statistical association (p < 0.001)
📝 Summary: Combined analysis across 4 nodes shows OR=726, p=9.88e-15
📡 Network: 4/4 nodes had relevant data
```

## Example Queries

- `Is CFTR rs113993960 significantly associated with Cystic Fibrosis?`
- `What variants co-occur with rs80357906 in breast cancer patients?`
- `Show me Fisher's test results for HBB rs334 and Sickle Cell Disease`
- `Which variants are linked to Lynch Syndrome?`

## Configuration

The system has configurable settings in `orchestrator_extended.py`:

```python
# Rate limiting (prevents 429 Too Many Requests)
DELAY_BETWEEN_EXTERNAL_CALLS = 1.0  # seconds between external node queries
DELAY_BETWEEN_LLM_CALLS = 0.5       # seconds between OpenAI API calls
MAX_INVESTIGATION_STEPS = 15        # Maximum tool calls before forcing stop

# Logging
LOG_DIR = Path("logs")
LOG_RESPONSES = True  # Enable/disable detailed response logging
```

### Rate Limiting

External node queries are executed **sequentially with delays** to prevent:
- OpenAI API rate limits (429 errors)
- Overwhelming external nodes with parallel requests

### Investigation Limits

The agent is limited to **15 tool calls** per investigation to prevent:
- Runaway loops
- Excessive API costs
- Overly long investigations

The system prompt guides the agent to prioritize important queries and stop early when evidence is strong.

## Project Structure

```
BioAgents/
├── .env                        # API key (create this, don't commit)
├── requirements.txt            # Python dependencies
├── generate_patient_data.py    # Creates synthetic patient data
├── node_agent.py               # Node agent (Flask + pydantic-ai)
├── orchestrator_agent.py       # Basic orchestrator agent
├── orchestrator_extended.py    # Extended orchestrator with home node
├── launch_agentic.py           # Basic system launcher
├── launch_extended.py          # Extended system launcher
├── launch_agentic_debug.py     # Debug launcher with logging
├── demo_agentic.py             # Demonstration script
├── variant_metadata.json       # Generated variant info
├── patients_node1-4.csv        # Generated patient data
└── logs/                       # Investigation logs (auto-created)
    └── investigation_*.json    # Detailed logs per session
```

## Privacy Architecture

Nodes share **only aggregated statistics**, never individual records:

| ✅ Shared | ❌ Never Shared |
|-----------|-----------------|
| Contingency table counts | Patient IDs |
| P-values, odds ratios | Individual variant status |
| Sample sizes | Disease status per patient |
| Confidence intervals | Any row-level data |

```python
# What gets transmitted (counts only):
{"variant_disease": 10, "variant_no_disease": 2, "p_value": 9.88e-15}

# What stays local (never transmitted):
{"patient_id": "P001", "has_disease": True, "rs113993960": True}
```

## Available Tools

### Node Agent Tools
| Tool | Description |
|------|-------------|
| `perform_fisher_test` | Fisher's exact test for variant-disease association |
| `find_co_occurring_variants` | Find variants appearing together in disease cases |
| `get_variant_frequencies` | Variant frequency across diseases |
| `check_available_data` | List available diseases and variants |

### Orchestrator Tool
| Tool | Description |
|------|-------------|
| `query_all_nodes_and_aggregate` | Broadcast query + automatic result aggregation |

## Diseases & Variants

**6 Rare Diseases:**
- Cystic Fibrosis (CFTR)
- Huntington Disease (HTT)
- Duchenne Muscular Dystrophy (DMD)
- Sickle Cell Disease (HBB)
- Hereditary Breast Cancer (BRCA1/BRCA2)
- Lynch Syndrome (MLH1, MSH2, MSH6, PMS2)

**30 Genetic Variants:**
- 20 pathogenic variants with known disease associations
- 10 benign control variants

## Troubleshooting

### Nodes not responding
```bash
python launch_agentic_debug.py  # Shows detailed logs
```

### API key issues
```bash
# Verify key is set
echo $OPENAI_API_KEY

# Or check .env file exists
cat .env
```

### Check node health
```bash
curl http://localhost:5001/health
# Expected: {"status": "healthy", "node_id": "node1", "agent": "pydantic-ai"}
```

### Rate limiting (429 Too Many Requests)

If you see `429 Too Many Requests` errors:

1. **Increase delays** in `orchestrator_extended.py`:
   ```python
   DELAY_BETWEEN_EXTERNAL_CALLS = 2.0  # Increase from 1.0
   ```

2. **Reduce max steps**:
   ```python
   MAX_INVESTIGATION_STEPS = 10  # Reduce from 15
   ```

3. **Check OpenAI rate limits** for your API tier at [platform.openai.com](https://platform.openai.com)

### Node 500 Internal Server Error

Check node logs:
```bash
python launch_agentic_debug.py --logs
```

Or check individual log files:
```bash
cat node1_stderr.log
```

## Investigation Logging

Extended mode logs all agent activity to JSON files for debugging and audit.

### Log Location

```bash
logs/investigation_YYYYMMDD_HHMMSS.json
```

### What's Logged

| Event Type | Information |
|------------|-------------|
| `tool_call` | Tool name, inputs, outputs |
| `external_node_response` | Node ID, has_data, full response |
| `llm_decision` | Step number, decision description |
| `final_result` | Complete investigation result |

### Viewing Logs

```bash
# List all logs
ls -la logs/

# Pretty-print latest log
cat logs/investigation_*.json | python -m json.tool

# Or use jq if installed
jq '.' logs/investigation_20251212_101504.json
```

### Example Log Entry

```json
{
  "timestamp": "2025-12-12T10:15:05.835",
  "type": "tool_call",
  "tool": "query_external_nodes_fisher",
  "inputs": {
    "variant_id": "rs113993960",
    "disease": "Cystic Fibrosis"
  },
  "outputs": {
    "combined_p_value": 4.24e-44,
    "combined_odds_ratio": 776.2,
    "nodes_with_data": 3
  }
}
```

### Disabling Logging

Set in `orchestrator_extended.py`:
```python
LOG_RESPONSES = False
```
```

## Tech Stack

| Package | Purpose | Version |
|---------|---------|---------|
| **pydantic-ai** | LLM agent framework with typed outputs | ≥0.0.20 |
| **pydantic** | Data validation and serialization | ≥2.0.0 |
| **Flask** | HTTP endpoints for node agents | ≥3.0.0 |
| **httpx** | Async HTTP client for orchestrator | ≥0.27.0 |
| **OpenAI** | GPT-4o API client | ≥1.0.0 |
| **scipy** | Fisher's exact test | ≥1.11.0 |
| **pandas** | Data manipulation | ≥2.0.0 |
| **numpy** | Numerical operations | ≥1.24.0 |

## Extended Mode: Investigative Agent

The extended system adds a **home node** concept where you have direct access to your own patients, plus an **autonomous investigation loop** with rate limiting and logging.

### Run Extended Mode

```bash
python launch_extended.py
```

### Key Features

- **Home Node**: Direct access to your patient records (node1)
- **External Nodes**: Privacy-preserving aggregated statistics (nodes 2-4)
- **Autonomous Loop**: Agent decides when to continue investigating
- **Rate Limiting**: Prevents API throttling (429 errors)
- **Max Steps**: Limits investigation to 15 tool calls
- **Response Logging**: Full audit trail in `logs/` directory

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                         YOU (Clinician)                         │
│    "I think rs113993960 is causal for my CF patients"          │
└────────────────────────────┬───────────────────────────────────┘
                             ▼
┌────────────────────────────────────────────────────────────────┐
│              Extended Orchestrator (Investigation Loop)         │
│  1. Query external nodes for significance                       │
│  2. Get co-occurring variants                                   │
│  3. Check YOUR patients for those variants     ←──┐             │
│  4. Decide: query more? (max 15 steps) ───────────┘             │
│  5. Return findings when sufficient evidence                    │
└──────────┬─────────────────────────────────────┬───────────────┘
           │                                     │
    ┌──────▼──────┐                      ┌──────▼──────┐
    │  HOME NODE  │                      │  EXTERNAL   │
    │  (node1)    │                      │  NODES 2-4  │
    │             │                      │             │
    │ YOUR patients                      │ Aggregated  │
    │ DIRECT access                      │ stats only  │
    └─────────────┘                      └─────────────┘
```

### Extended Tools

| Tool | Access Level | Purpose |
|------|--------------|---------|
| `check_home_patients_for_variant` | Direct | See YOUR patients with a variant |
| `check_home_patients_variant_combination` | Direct | Check YOUR patients for A+B |
| `get_home_patient_details` | Direct | Full details on YOUR patient |
| `list_home_patients_with_disease` | Direct | List YOUR affected patients |
| `query_external_nodes_fisher` | Aggregated | External significance testing |
| `query_external_nodes_cooccurrence` | Aggregated | External co-occurrence data |
| `query_external_nodes_variant_pair` | Aggregated | External A+B significance |

### Example Investigation

```
🔬 Investigation: I suspect rs113993960 is causal for my CF patients. Investigate.

🔄 Steps Taken (5):
   1. Querying external nodes: Fisher test rs113993960 vs Cystic Fibrosis
   2. Checking home patients for variant rs113993960 with Cystic Fibrosis
   3. Querying external nodes: co-occurrence with rs113993960 in Cystic Fibrosis
   4. Checking home patients for combination rs113993960 + rs75961395
   5. Querying external nodes: significance of rs113993960+rs75961395

📊 Result: Strong association (p=9.88e-15, OR=726). 
   8 of your CF patients have this variant.
   Co-occurring variant rs75961395 found in 5 of those patients.
```