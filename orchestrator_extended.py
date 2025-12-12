#!/usr/bin/env python3
"""
BioAgents Extended Orchestrator Agent
- Home node with direct patient visibility (user's own patients)
- Agentic investigation loop (queries → checks → re-queries)
- Autonomous decision-making on when to continue/stop
"""

from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import httpx
import asyncio
import json
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
import logging
from datetime import datetime
from pathlib import Path
import dotenv

dotenv.load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

# Rate limiting settings
DELAY_BETWEEN_EXTERNAL_CALLS = 1.0  # seconds between external node queries
DELAY_BETWEEN_LLM_CALLS = 0.5       # seconds between OpenAI API calls
MAX_INVESTIGATION_STEPS = 15        # Maximum tool calls before forcing stop

# Logging settings
LOG_DIR = Path("logs")
LOG_RESPONSES = True  # Enable/disable detailed response logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# ============================================================================
# Response Logger
# ============================================================================

class ResponseLogger:
    """Logs all agent responses and tool outputs to file and console"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = None
        self.logger = logging.getLogger("ResponseLogger")
        
        if enabled:
            LOG_DIR.mkdir(exist_ok=True)
            self.log_file = LOG_DIR / f"investigation_{self.session_id}.json"
            self.entries = []
            self.logger.info(f"Response logging enabled: {self.log_file}")
    
    def log_tool_call(self, tool_name: str, inputs: Dict, outputs: Dict):
        """Log a tool call with inputs and outputs"""
        if not self.enabled:
            return
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "tool_call",
            "tool": tool_name,
            "inputs": self._sanitize(inputs),
            "outputs": self._sanitize(outputs)
        }
        self.entries.append(entry)
        self.logger.info(f"Tool [{tool_name}] called with {list(inputs.keys())}")
        self.logger.debug(f"Tool [{tool_name}] output: {json.dumps(outputs, default=str)[:500]}")
    
    def log_external_response(self, node_id: str, response: Dict):
        """Log response from external node"""
        if not self.enabled:
            return
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "external_node_response",
            "node_id": node_id,
            "has_data": response.get("has_data"),
            "response": self._sanitize(response)
        }
        self.entries.append(entry)
        self.logger.debug(f"External node [{node_id}] response: has_data={response.get('has_data')}")
    
    def log_llm_decision(self, decision: str, context: Dict = None):
        """Log LLM decision points"""
        if not self.enabled:
            return
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "llm_decision",
            "decision": decision,
            "context": self._sanitize(context) if context else {}
        }
        self.entries.append(entry)
        self.logger.info(f"LLM Decision: {decision}")
    
    def log_final_result(self, result: Dict):
        """Log the final investigation result"""
        if not self.enabled:
            return
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "final_result",
            "result": self._sanitize(result)
        }
        self.entries.append(entry)
        self._save_to_file()
        self.logger.info(f"Investigation complete. Log saved to {self.log_file}")
    
    def _sanitize(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'model_dump'):
            return obj.model_dump()
        else:
            try:
                json.dumps(obj)
                return obj
            except:
                return str(obj)
    
    def _save_to_file(self):
        """Save all entries to JSON file"""
        if self.log_file:
            with open(self.log_file, 'w') as f:
                json.dump(self.entries, f, indent=2, default=str)


# Global response logger instance
response_logger = ResponseLogger(enabled=LOG_RESPONSES)


# ============================================================================
# Response Models
# ============================================================================

class PatientFinding(BaseModel):
    """Finding about a specific patient"""
    patient_id: str
    has_variant: bool
    variant_id: str
    disease: str
    has_disease: bool
    additional_variants: Optional[List[str]] = None


class InvestigationResult(BaseModel):
    """Result of an agentic investigation"""
    query: str
    investigation_summary: str
    primary_variant: Optional[str] = None
    co_occurring_variants: Optional[List[Dict[str, Any]]] = None
    affected_patients: Optional[List[PatientFinding]] = None
    statistical_findings: Optional[Dict[str, Any]] = None
    external_nodes_queried: int = 0
    investigation_steps: List[str] = Field(default_factory=list)
    recommendations: Optional[List[str]] = None
    requires_further_investigation: bool = False


def extract_agent_result(result, expected_type=None, logger=None) -> Optional[Dict]:
    """Extract data from pydantic-ai AgentRunResult."""
    if isinstance(result, dict):
        return result
    if expected_type and isinstance(result, expected_type):
        return result.model_dump()
    for attr_name in ['data', 'output', 'result', 'value', 'response']:
        if hasattr(result, attr_name):
            attr_value = getattr(result, attr_name)
            if attr_value is None:
                continue
            if expected_type and isinstance(attr_value, expected_type):
                return attr_value.model_dump()
            if hasattr(attr_value, 'model_dump'):
                try:
                    return attr_value.model_dump()
                except Exception:
                    pass
            if isinstance(attr_value, dict):
                return attr_value
    return None


# ============================================================================
# Extended Orchestrator Context with Home Node
# ============================================================================

class ExtendedOrchestratorContext:
    """
    Orchestrator context with:
    - Direct access to home node patient data (privileged)
    - HTTP access to external nodes (aggregated stats only)
    - Rate limiting for API calls
    - Response logging
    """
    
    def __init__(
        self, 
        external_node_urls: Dict[str, str],
        home_node_data_file: str,
        variant_metadata_file: str = "variant_metadata.json"
    ):
        self.external_node_urls = external_node_urls
        self.logger = logging.getLogger("ExtendedOrchestrator")
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Load HOME NODE data - direct access (privileged)
        self.logger.info(f"Loading home node data from {home_node_data_file}")
        self.home_patients = pd.read_csv(home_node_data_file)
        
        # Convert boolean columns
        bool_columns = ['has_disease'] + [col for col in self.home_patients.columns if col.startswith('rs')]
        for col in bool_columns:
            if col in self.home_patients.columns:
                self.home_patients[col] = self.home_patients[col].map(
                    {'True': True, 'False': False, True: True, False: False}
                )
        
        # Load variant metadata
        try:
            with open(variant_metadata_file, 'r') as f:
                self.variant_metadata = json.load(f)
        except:
            self.variant_metadata = {}
        
        # Identify available data
        self.home_variant_columns = [col for col in self.home_patients.columns if col.startswith('rs')]
        self.home_diseases = list(self.home_patients['disease'].unique())
        
        self.logger.info(f"Home node: {len(self.home_patients)} patients, "
                        f"{len(self.home_diseases)} diseases, {len(self.home_variant_columns)} variants")
        
        # Rate limiting state
        self.last_external_call_time = 0
        self.last_llm_call_time = 0
        
        # Investigation tracking
        self.investigation_steps: List[str] = []
        self.step_count = 0
    
    def log_step(self, step: str):
        """Log an investigation step with rate limit check"""
        self.step_count += 1
        
        # Check if we've exceeded max steps
        if self.step_count > MAX_INVESTIGATION_STEPS:
            self.logger.warning(f"Max investigation steps ({MAX_INVESTIGATION_STEPS}) reached!")
            raise StopIteration(f"Investigation stopped: exceeded {MAX_INVESTIGATION_STEPS} steps")
        
        self.investigation_steps.append(step)
        self.logger.info(f"Investigation step {self.step_count}/{MAX_INVESTIGATION_STEPS}: {step}")
        response_logger.log_llm_decision(f"Step {self.step_count}: {step}")
    
    def clear_steps(self):
        """Clear investigation steps for new query"""
        self.investigation_steps = []
        self.step_count = 0
    
    async def _rate_limit_external(self):
        """Apply rate limiting for external node calls"""
        import time
        current_time = time.time()
        elapsed = current_time - self.last_external_call_time
        
        if elapsed < DELAY_BETWEEN_EXTERNAL_CALLS:
            wait_time = DELAY_BETWEEN_EXTERNAL_CALLS - elapsed
            self.logger.debug(f"Rate limiting: waiting {wait_time:.2f}s before external call")
            await asyncio.sleep(wait_time)
        
        self.last_external_call_time = time.time()
    
    async def query_external_node(self, node_id: str, query: str) -> Dict:
        """Query a single external node (aggregated stats only) with rate limiting"""
        await self._rate_limit_external()
        
        try:
            url = f"{self.external_node_urls[node_id]}/query"
            response = await self.client.post(
                url,
                json={"query": query, "context": {}}
            )
            
            if response.status_code == 200:
                result = response.json()
                response_logger.log_external_response(node_id, result)
                return result
            else:
                error_result = {"node_id": node_id, "has_data": False, "message": f"HTTP {response.status_code}"}
                response_logger.log_external_response(node_id, error_result)
                return error_result
                
        except Exception as e:
            error_result = {"node_id": node_id, "has_data": False, "message": str(e)}
            response_logger.log_external_response(node_id, error_result)
            return error_result
    
    async def broadcast_to_external_nodes(self, query: str) -> List[Dict]:
        """Broadcast query to all external nodes with staggered timing"""
        results = []
        
        # Query nodes sequentially with rate limiting to avoid 429s
        for node_id in self.external_node_urls.keys():
            result = await self.query_external_node(node_id, query)
            results.append(result)
        
        return results
    
    async def close(self):
        await self.client.aclose()


# ============================================================================
# Create the Extended Orchestrator Agent
# ============================================================================

extended_orchestrator = Agent(
    'openai:gpt-4o',
    deps_type=ExtendedOrchestratorContext,
    output_type=InvestigationResult,
    model_settings={
        'max_tokens': 8192,  # Increase from default 3000
        'temperature': 0.7,
    },
    system_prompt="""You are an advanced clinical investigation agent for genetic variant analysis.

YOU HAVE TWO TYPES OF DATA ACCESS:
1. HOME NODE (Direct Access): You can see individual patient records for the user's own patients
2. EXTERNAL NODES (Privacy-Preserving): You only get aggregated statistics from other clinical centers

YOUR INVESTIGATION WORKFLOW:
When a user asks about a variant's significance or wants to investigate a potential causal variant:

1. QUERY EXTERNAL NODES for statistical significance of the primary variant
2. GET CO-OCCURRING VARIANTS from external nodes
3. CHECK HOME PATIENTS to see which of user's patients have the primary variant and co-occurring variants
4. DECIDE: Based on findings, should you query external nodes again for:
   - Co-occurrence significance of variant pairs?
   - Additional variant combinations?
5. LOOP until you have sufficient evidence, then SYNTHESIZE findings

IMPORTANT CONSTRAINTS:
- You have a MAXIMUM of 15 tool calls per investigation
- Prioritize the most important queries first
- Don't test every possible variant combination - focus on:
  * The primary variant mentioned by user
  * Top 2-3 co-occurring variants by frequency
  * Variants the specific patient has
- If you're running low on steps, summarize what you've found

AVAILABLE TOOLS:
- query_external_nodes_fisher: Get Fisher's test results from external nodes
- query_external_nodes_cooccurrence: Get co-occurring variants from external nodes  
- check_home_patients_for_variant: See which of YOUR patients have a specific variant
- check_home_patients_variant_combination: Check YOUR patients for multiple variants together
- get_home_patient_details: Get detailed info about specific home patients
- list_home_patients_with_disease: List YOUR patients with a specific disease

DECISION CRITERIA FOR CONTINUING INVESTIGATION:
- If co-occurring variants are found, check if home patients have them
- If home patients have co-occurring variants, query significance of TOP 2-3 combinations only
- If p-value is very significant (< 0.001), you likely have enough evidence
- Stop early if primary variant shows strong significance (OR > 50, p < 1e-10)

RETURN InvestigationResult with:
- investigation_summary: What you found and your reasoning
- affected_patients: Which of the user's patients are relevant
- statistical_findings: Combined statistical evidence
- investigation_steps: List of steps you took
- recommendations: Clinical/research recommendations

Be thorough but EFFICIENT. Prioritize quality over quantity of queries."""
)


# ============================================================================
# HOME NODE TOOLS (Direct Patient Access)
# ============================================================================

@extended_orchestrator.tool
async def check_home_patients_for_variant(
    ctx: RunContext[ExtendedOrchestratorContext],
    variant_id: str,
    disease: Optional[str] = None
) -> Dict[str, Any]:
    """
    Check which of YOUR patients (home node) have a specific variant.
    This gives you direct patient-level information.
    
    Args:
        variant_id: The variant to check (e.g., 'rs113993960')
        disease: Optional - filter to patients with specific disease
    """
    ctx.deps.log_step(f"Checking home patients for variant {variant_id}" + 
                      (f" with {disease}" if disease else ""))
    
    if variant_id not in ctx.deps.home_variant_columns:
        result = {"error": f"Variant {variant_id} not in home node data"}
        response_logger.log_tool_call("check_home_patients_for_variant", 
                                      {"variant_id": variant_id, "disease": disease}, result)
        return result
    
    df = ctx.deps.home_patients
    
    if disease:
        df = df[df['disease'] == disease]
    
    patients_with_variant = df[df[variant_id] == True]
    
    patient_list = []
    for _, row in patients_with_variant.iterrows():
        # Find other variants this patient has
        other_variants = [v for v in ctx.deps.home_variant_columns 
                         if v != variant_id and row.get(v) == True]
        
        patient_list.append({
            "patient_id": row['patient_id'],
            "disease": row['disease'],
            "has_disease": bool(row['has_disease']),
            "other_variants": other_variants[:10]  # Top 10 for brevity
        })
    
    result = {
        "variant_id": variant_id,
        "disease_filter": disease,
        "total_patients_checked": len(df),
        "patients_with_variant": len(patient_list),
        "patient_details": patient_list[:20],  # Limit for response size
        "has_more": len(patient_list) > 20
    }
    
    response_logger.log_tool_call("check_home_patients_for_variant",
                                  {"variant_id": variant_id, "disease": disease}, result)
    return result


@extended_orchestrator.tool
async def check_home_patients_variant_combination(
    ctx: RunContext[ExtendedOrchestratorContext],
    variant_a: str,
    variant_b: str,
    disease: Optional[str] = None
) -> Dict[str, Any]:
    """
    Check which of YOUR patients have BOTH variants together.
    Useful for investigating co-occurrence in your own patient population.
    
    Args:
        variant_a: First variant
        variant_b: Second variant  
        disease: Optional disease filter
    """
    ctx.deps.log_step(f"Checking home patients for combination {variant_a} + {variant_b}")
    
    for v in [variant_a, variant_b]:
        if v not in ctx.deps.home_variant_columns:
            return {"error": f"Variant {v} not in home node data"}
    
    df = ctx.deps.home_patients
    if disease:
        df = df[df['disease'] == disease]
    
    # Patients with both variants
    both = df[(df[variant_a] == True) & (df[variant_b] == True)]
    only_a = df[(df[variant_a] == True) & (df[variant_b] == False)]
    only_b = df[(df[variant_a] == False) & (df[variant_b] == True)]
    neither = df[(df[variant_a] == False) & (df[variant_b] == False)]
    
    patient_list = []
    for _, row in both.iterrows():
        patient_list.append({
            "patient_id": row['patient_id'],
            "disease": row['disease'],
            "has_disease": bool(row['has_disease'])
        })
    
    return {
        "variant_a": variant_a,
        "variant_b": variant_b,
        "disease_filter": disease,
        "patients_with_both": len(both),
        "patients_with_only_a": len(only_a),
        "patients_with_only_b": len(only_b),
        "patients_with_neither": len(neither),
        "patient_details": patient_list[:20],
        "co_occurrence_rate": len(both) / len(df) if len(df) > 0 else 0
    }


@extended_orchestrator.tool
async def get_home_patient_details(
    ctx: RunContext[ExtendedOrchestratorContext],
    patient_id: str
) -> Dict[str, Any]:
    """
    Get detailed information about a specific patient in YOUR home node.
    
    Args:
        patient_id: The patient ID to look up
    """
    ctx.deps.log_step(f"Getting details for patient {patient_id}")
    
    patient = ctx.deps.home_patients[ctx.deps.home_patients['patient_id'] == patient_id]
    
    if len(patient) == 0:
        return {"error": f"Patient {patient_id} not found in home node"}
    
    row = patient.iloc[0]
    
    # Get all variants this patient has
    variants_present = [v for v in ctx.deps.home_variant_columns if row.get(v) == True]
    
    # Annotate variants with metadata
    variant_details = []
    for v in variants_present:
        meta = ctx.deps.variant_metadata.get(v, {})
        variant_details.append({
            "variant_id": v,
            "gene": meta.get("gene", "Unknown"),
            "pathogenicity": meta.get("pathogenicity", "Unknown"),
            "associated_disease": meta.get("associated_disease")
        })
    
    return {
        "patient_id": patient_id,
        "disease_context": row['disease'],
        "has_disease": bool(row['has_disease']),
        "variants_present": variant_details,
        "total_variants": len(variants_present)
    }


@extended_orchestrator.tool
async def list_home_patients_with_disease(
    ctx: RunContext[ExtendedOrchestratorContext],
    disease: str,
    only_affected: bool = True
) -> Dict[str, Any]:
    """
    List YOUR patients with a specific disease context.
    
    Args:
        disease: The disease to filter by
        only_affected: If True, only return patients who have the disease (cases)
    """
    ctx.deps.log_step(f"Listing home patients with {disease} (affected={only_affected})")
    
    df = ctx.deps.home_patients[ctx.deps.home_patients['disease'] == disease]
    
    if only_affected:
        df = df[df['has_disease'] == True]
    
    patients = []
    for _, row in df.iterrows():
        variants = [v for v in ctx.deps.home_variant_columns if row.get(v) == True]
        pathogenic = [v for v in variants if ctx.deps.variant_metadata.get(v, {}).get('pathogenicity') == 'Pathogenic']
        
        patients.append({
            "patient_id": row['patient_id'],
            "has_disease": bool(row['has_disease']),
            "total_variants": len(variants),
            "pathogenic_variants": pathogenic
        })
    
    return {
        "disease": disease,
        "only_affected": only_affected,
        "patient_count": len(patients),
        "patients": patients[:30]  # Limit response size
    }


# ============================================================================
# EXTERNAL NODE TOOLS (Aggregated Statistics Only)
# ============================================================================

@extended_orchestrator.tool
async def query_external_nodes_fisher(
    ctx: RunContext[ExtendedOrchestratorContext],
    variant_id: str,
    disease: str
) -> Dict[str, Any]:
    """
    Query EXTERNAL nodes for Fisher's exact test on variant-disease association.
    Returns aggregated statistics only (privacy-preserving).
    
    Args:
        variant_id: Variant to test
        disease: Disease to test association with
    """
    ctx.deps.log_step(f"Querying external nodes: Fisher test {variant_id} vs {disease}")
    
    query = f"Perform Fisher's exact test for variant {variant_id} and {disease}"
    responses = await ctx.deps.broadcast_to_external_nodes(query)
    
    # Aggregate results
    valid_results = []
    for resp in responses:
        node_id = resp.get('node_id', 'unknown')
        if resp.get('has_data') and resp.get('results'):
            for res in resp['results']:
                if res.get('p_value') is not None:
                    valid_results.append({
                        'node_id': node_id,
                        'p_value': res['p_value'],
                        'odds_ratio': res.get('odds_ratio'),
                        'sample_size': res.get('sample_size', 0)
                    })
    
    if not valid_results:
        result = {
            "variant_id": variant_id,
            "disease": disease,
            "nodes_queried": len(responses),
            "nodes_with_data": 0,
            "error": "No external nodes had data for this query"
        }
        response_logger.log_tool_call("query_external_nodes_fisher",
                                      {"variant_id": variant_id, "disease": disease}, result)
        return result
    
    # Combine p-values (Fisher's method)
    p_values = [r['p_value'] for r in valid_results if r['p_value'] > 0]
    if p_values:
        from scipy.stats import chi2
        chi_sq = -2 * sum(np.log(p) for p in p_values)
        combined_p = chi2.sf(chi_sq, 2 * len(p_values))
    else:
        combined_p = None
    
    # Combine odds ratios (geometric mean)
    ors = [r['odds_ratio'] for r in valid_results if r.get('odds_ratio') and r['odds_ratio'] != 'infinity']
    combined_or = np.exp(np.mean([np.log(o) for o in ors])) if ors else None
    
    result = {
        "variant_id": variant_id,
        "disease": disease,
        "nodes_queried": len(responses),
        "nodes_with_data": len(valid_results),
        "combined_p_value": float(combined_p) if combined_p else None,
        "combined_odds_ratio": float(combined_or) if combined_or else "infinity",
        "per_node_results": valid_results,
        "interpretation": "Significant" if combined_p and combined_p < 0.05 else "Not significant"
    }
    
    response_logger.log_tool_call("query_external_nodes_fisher",
                                  {"variant_id": variant_id, "disease": disease}, result)
    return result


@extended_orchestrator.tool
async def query_external_nodes_cooccurrence(
    ctx: RunContext[ExtendedOrchestratorContext],
    variant_id: str,
    disease: str
) -> Dict[str, Any]:
    """
    Query EXTERNAL nodes for variants that co-occur with the given variant.
    Returns aggregated co-occurrence statistics (privacy-preserving).
    
    Args:
        variant_id: The primary variant to find co-occurrences for
        disease: Disease context
    """
    ctx.deps.log_step(f"Querying external nodes: co-occurrence with {variant_id} in {disease}")
    
    query = f"Find variants that co-occur with {variant_id} in {disease} patients"
    responses = await ctx.deps.broadcast_to_external_nodes(query)
    
    # Aggregate co-occurring variants across nodes
    variant_summary = {}
    nodes_with_data = 0
    
    for resp in responses:
        if resp.get('has_data') and resp.get('results'):
            for res in resp['results']:
                raw = res.get('raw_data', {})
                if 'co_occurring_variants' in raw:
                    nodes_with_data += 1
                    for v in raw['co_occurring_variants']:
                        vid = v['variant_id']
                        if vid not in variant_summary:
                            variant_summary[vid] = {
                                'gene': v.get('gene', 'Unknown'),
                                'pathogenicity': v.get('pathogenicity', 'Unknown'),
                                'rates': [],
                                'counts': []
                            }
                        variant_summary[vid]['rates'].append(v['co_occurrence_rate'])
                        variant_summary[vid]['counts'].append(v['count'])
    
    # Calculate averages
    aggregated = []
    for vid, info in variant_summary.items():
        aggregated.append({
            'variant_id': vid,
            'gene': info['gene'],
            'pathogenicity': info['pathogenicity'],
            'avg_co_occurrence_rate': float(np.mean(info['rates'])),
            'total_observations': sum(info['counts']),
            'nodes_reporting': len(info['rates'])
        })
    
    aggregated.sort(key=lambda x: x['avg_co_occurrence_rate'], reverse=True)
    
    return {
        "primary_variant": variant_id,
        "disease": disease,
        "nodes_queried": len(responses),
        "nodes_with_data": nodes_with_data,
        "co_occurring_variants": aggregated[:15],
        "total_unique_variants_found": len(aggregated)
    }


@extended_orchestrator.tool
async def query_external_nodes_variant_pair(
    ctx: RunContext[ExtendedOrchestratorContext],
    variant_a: str,
    variant_b: str,
    disease: str
) -> Dict[str, Any]:
    """
    Query EXTERNAL nodes for statistical significance of having BOTH variants.
    Use this after finding co-occurring variants to test if the combination is significant.
    
    Args:
        variant_a: First variant
        variant_b: Second variant
        disease: Disease context
    """
    ctx.deps.log_step(f"Querying external nodes: significance of {variant_a}+{variant_b} in {disease}")
    
    query = f"Test if having both {variant_a} and {variant_b} is significant for {disease}"
    responses = await ctx.deps.broadcast_to_external_nodes(query)
    
    # Collect any significance results
    findings = []
    for resp in responses:
        node_id = resp.get('node_id', 'unknown')
        if resp.get('has_data'):
            findings.append({
                'node_id': node_id,
                'message': resp.get('message', ''),
                'results': resp.get('results', [])
            })
    
    return {
        "variant_a": variant_a,
        "variant_b": variant_b,
        "disease": disease,
        "nodes_queried": len(responses),
        "nodes_responded": len(findings),
        "findings": findings
    }


# ============================================================================
# Extended Orchestrator Interface
# ============================================================================

class ExtendedOrchestratorInterface:
    """Interface for the extended orchestrator with investigation capabilities"""
    
    def __init__(
        self, 
        external_node_urls: Dict[str, str],
        home_node_data_file: str,
        variant_metadata_file: str = "variant_metadata.json"
    ):
        self.context = ExtendedOrchestratorContext(
            external_node_urls,
            home_node_data_file,
            variant_metadata_file
        )
        self.logger = logging.getLogger("ExtendedOrchestratorInterface")
    
    async def investigate(self, user_query: str) -> Dict[str, Any]:
        """
        Run an agentic investigation based on user query.
        The agent will autonomously decide how many steps to take.
        """
        try:
            self.context.clear_steps()
            self.logger.info(f"Starting investigation: {user_query}")
            response_logger.log_llm_decision(f"Starting investigation: {user_query}")
            
            result = await extended_orchestrator.run(
                user_query,
                deps=self.context
            )
            
            result_dict = extract_agent_result(result, InvestigationResult, self.logger)
            
            if result_dict:
                # Add the investigation steps we tracked
                result_dict['investigation_steps'] = self.context.investigation_steps
                
                # Log final result
                response_logger.log_final_result(result_dict)
                
                return result_dict
            else:
                error_result = {
                    "query": user_query,
                    "investigation_summary": "Error: Could not parse agent response",
                    "investigation_steps": self.context.investigation_steps
                }
                response_logger.log_final_result(error_result)
                return error_result
        
        except StopIteration as e:
            # Max steps reached - return partial results
            self.logger.warning(f"Investigation stopped: {e}")
            partial_result = {
                "query": user_query,
                "investigation_summary": f"Investigation stopped after {MAX_INVESTIGATION_STEPS} steps. Partial findings gathered.",
                "investigation_steps": self.context.investigation_steps,
                "partial": True,
                "stop_reason": str(e)
            }
            response_logger.log_final_result(partial_result)
            return partial_result
                
        except Exception as e:
            self.logger.error(f"Investigation error: {e}")
            import traceback
            traceback.print_exc()
            error_result = {
                "query": user_query,
                "investigation_summary": f"Error: {str(e)}",
                "investigation_steps": self.context.investigation_steps
            }
            response_logger.log_final_result(error_result)
            return error_result
    
    async def close(self):
        await self.context.close()


# ============================================================================
# Interactive CLI
# ============================================================================

async def run_extended_interactive():
    """Run interactive session with extended orchestrator"""
    
    external_nodes = {
        "node2": "http://localhost:5002",
        "node3": "http://localhost:5003",
        "node4": "http://localhost:5004"
    }
    
    # Node1 is our HOME node - we have direct access
    home_node_file = "patients_node1.csv"
    
    orchestrator = ExtendedOrchestratorInterface(
        external_nodes,
        home_node_file
    )
    
    print("\n" + "="*70)
    print("BioAgents EXTENDED - Investigative Agent")
    print("="*70)
    print("\nYou have DIRECT ACCESS to your home node (node1) patient data.")
    print("External nodes (2-4) provide aggregated statistics only.")
    print("\nExample investigation queries:")
    print("  • I think rs113993960 might be causal for my CF patients. Investigate.")
    print("  • Check if my breast cancer patients have BRCA1 rs80357906 and find co-occurring variants")
    print("  • Investigate variant rs334 in sickle cell - check my patients and external evidence")
    print("\nType 'exit' to quit\n")
    
    while True:
        try:
            query = input("🔬 Your investigation: ").strip()
            
            if query.lower() == 'exit':
                break
            if not query:
                continue
            
            print("\n🤖 Investigating (agent will loop autonomously)...\n")
            
            result = await orchestrator.investigate(query)
            
            print("="*70)
            print("📊 INVESTIGATION RESULTS")
            print("="*70)
            
            print(f"\n📝 Summary:\n{result.get('investigation_summary', 'N/A')}")
            
            if result.get('investigation_steps'):
                print(f"\n🔄 Investigation Steps ({len(result['investigation_steps'])}):")
                for i, step in enumerate(result['investigation_steps'], 1):
                    print(f"   {i}. {step}")
            
            if result.get('affected_patients'):
                print(f"\n👥 Your Affected Patients:")
                for p in result['affected_patients'][:10]:
                    print(f"   • {p}")
            
            if result.get('statistical_findings'):
                print(f"\n📈 Statistical Findings:")
                print(json.dumps(result['statistical_findings'], indent=2, default=str))
            
            if result.get('co_occurring_variants'):
                print(f"\n🧬 Co-occurring Variants Found:")
                for v in result['co_occurring_variants'][:5]:
                    print(f"   • {v.get('variant_id')} ({v.get('gene')}) - rate: {v.get('avg_co_occurrence_rate', 'N/A'):.2%}")
            
            if result.get('recommendations'):
                print(f"\n💡 Recommendations:")
                for rec in result['recommendations']:
                    print(f"   • {rec}")
            
            print("\n" + "="*70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    await orchestrator.close()
    print("\nGoodbye!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        asyncio.run(run_extended_interactive())
    else:
        print("Extended Orchestrator with Home Node + Investigation Loop")
        print("\nUsage:")
        print("  python orchestrator_extended.py --demo")
        print("\nMake sure external node agents (5002-5004) are running")
        print("Node1 data (patients_node1.csv) is used as your home node")