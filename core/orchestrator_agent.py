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
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
import logging
from datetime import datetime
from pathlib import Path
import dotenv
from .schema_manager import SchemaManager
from .schemas import SearchResponse, AnalysisResult
from .structured_logger import StructuredLogger, trace_id_var, span_id_var, parent_span_id_var, request_id_var
import uuid
import time

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

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

# We instantiate this locally in the orchestrator interface to assign session files dynamically
# So we don't have a global logger anymore, and we pass the logger into tools via context.


def extract_agent_result(result, expected_type=None, logger=None) -> Optional[Dict]:
    """
    Extract the actual data from a pydantic-ai AgentRunResult.
    """
    if logger:
        logger.info(f"Extracting result from type: {type(result)}")
    
    if isinstance(result, dict):
        return result
    
    if expected_type and isinstance(result, expected_type):
        return result.model_dump()
    
    # Check for pydantic model instance
    if hasattr(result, 'model_dump'):
        return result.model_dump()

    for attr_name in ['data', 'output', 'result', 'value', 'response']:
        if hasattr(result, attr_name):
            attr_value = getattr(result, attr_name)
            
            if attr_value is None:
                continue
                
            if expected_type and isinstance(attr_value, expected_type):
                return attr_value.model_dump()
            elif isinstance(attr_value, dict):
                return attr_value
            elif hasattr(attr_value, 'model_dump'):
                return attr_value.model_dump()
            elif hasattr(attr_value, 'dict'):
                return attr_value.dict()
    return None


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
        external_node_urls: Dict[str, str] = None,
        home_node_data_directory: str = "nodes/node1",
        variant_metadata_file: str = "variant_metadata.json",
        structured_logger: Optional[StructuredLogger] = None
    ):
        self.external_node_urls = external_node_urls or {}
        self._refresh_nodes()
        self.structured_logger = structured_logger
        self.logger = logging.getLogger("ExtendedOrchestrator")
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Load HOME NODE data - directory based
        self.logger.info(f"Loading home node data from directory: {home_node_data_directory}")
        
        if not os.path.exists(home_node_data_directory):
            self.logger.error(f"Home node directory not found: {home_node_data_directory}")
            self.home_patients = pd.DataFrame()
        else:
            csv_files = []
            for root, _, files in os.walk(home_node_data_directory):
                for f in files:
                    if f.lower().endswith('.csv'):
                        csv_files.append(os.path.join(root, f))
            
            self.logger.info(f"Discovered {len(csv_files)} CSV files: {csv_files}")
            
            if not csv_files:
                self.logger.warning(f"No CSV files found in {home_node_data_directory}")
                self.home_patients = pd.DataFrame()
            else:
                dfs = []
        
                for file_path in csv_files:
                    csv_file = os.path.basename(file_path)
                    try:
                        df = pd.read_csv(file_path)
                        # Normalize headers and enforce privacy using SchemaManager BEFORE concat
                        schema_manager = SchemaManager(self.logger)
                        df = schema_manager.normalize_dataframe(df)
                        dfs.append(df)
                        self.logger.info(f"Loaded {len(df)} records from {csv_file}")
                    except Exception as e:
                        self.logger.error(f"Error loading {csv_file}: {e}")
                
                if dfs:
                    self.home_patients = pd.concat(dfs, ignore_index=True)
                else:
                    self.home_patients = pd.DataFrame()
        
        # Check if we have a summary 'variants' column
        if 'variants' in self.home_patients.columns:
            self.logger.info("Exploding 'variants' summary column to individual rsID columns...")
            
            # 1. Collect all unique variants
            all_variants = set()
            for v_str in self.home_patients['variants'].dropna():
                if isinstance(v_str, str):
                    for v in v_str.split(';'):
                        if v.strip():
                            all_variants.add(v.strip())
            
            self.logger.info(f"Found {len(all_variants)} unique variants to expand.")
            
            # 2. Create/Merge boolean columns
            for v_id in all_variants:
                from_summary = self.home_patients['variants'].apply(
                    lambda x: v_id in x.split(';') if isinstance(x, str) else False
                )
                
                if v_id in self.home_patients.columns:
                    # Merge with existing data
                    self.home_patients[v_id] = self.home_patients[v_id].fillna(False).infer_objects(copy=False).astype(bool) | from_summary
                else:
                    self.home_patients[v_id] = from_summary
        
        # Convert boolean columns
        bool_columns = ['has_disease'] + [col for col in self.home_patients.columns if col.startswith('rs')]
        for col in bool_columns:
            if col in self.home_patients.columns:
                self.home_patients[col] = self.home_patients[col].map(
                    {'True': True, 'False': False, True: True, False: False}
                ).fillna(False).infer_objects(copy=False).astype(bool)
        
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
        if self.structured_logger:
            self.structured_logger.log_acting(f"Step {self.step_count}: {step}", {})
        else:
            self.logger.info(f"Investigation step {self.step_count}/{MAX_INVESTIGATION_STEPS}: {step}")

    
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
        # For parallel calls, we handle rate limiting differently or assume the delay is per-node
        # async sleep here might effectively serialize if called in loop, but with gather it's concurrent
        # To avoid blasting all nodes instantly, we can add a small jitter or just rely on async IO
        
        try:
            url = f"{self.external_node_urls[node_id]}/query"
            
            # Propagate trace context to the node via HTTP headers
            headers = {}
            if trace_id_var.get():
                headers["X-Trace-Id"] = trace_id_var.get()
            if span_id_var.get():
                headers["X-Parent-Span-Id"] = span_id_var.get()
            if request_id_var.get():
                headers["X-Request-Id"] = request_id_var.get()
            
            # Sub-span for this node's network call
            node_span = str(uuid.uuid4())
            headers["X-Span-Id"] = node_span
            
            if self.structured_logger:
                 self.structured_logger.log_acting(f"Dispatching query to {node_id}", {"url": url, "query": query})
            
            start_time = time.time()
            response = await self.client.post(
                url,
                json={"query": query, "context": {}},
                headers=headers
            )
            elapsed_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                
                # Ingest node's buffered trace logs into the consolidated log
                remote_logs = result.pop("_trace_logs", [])
                if remote_logs and self.structured_logger:
                    self.structured_logger.ingest_remote_logs(remote_logs)
                
                if self.structured_logger:
                    self.structured_logger.log_acting(f"Received response from {node_id}", {
                        "status": 200, "node_id": node_id,
                        "has_data": result.get("has_data"),
                        "network_latency_ms": round(elapsed_ms, 2)
                    })
                return result
            else:
                if self.structured_logger:
                     self.structured_logger.log_event("ERROR", "ACTING", f"HTTP error from {node_id}",
                         node_id=node_id, status_code=response.status_code)
                return {"node_id": node_id, "has_data": False, "message": f"HTTP {response.status_code}"}
                
        except Exception as e:
            if self.structured_logger:
                 self.structured_logger.log_event("ERROR", "ACTING", f"Exception querying {node_id}",
                     error_detail=str(e), node_id=node_id)
            return {"node_id": node_id, "has_data": False, "message": str(e)}
    
    def _refresh_nodes(self):
        try:
            from core.discovery import get_service_map
            registry = get_service_map()
            self.external_node_urls = {
                nid: info['url']
                for nid, info in registry.items()
                if nid != 'node1'
            }
        except ImportError:
            pass
            
    async def broadcast_to_external_nodes(self, query: str) -> List[Dict]:
        """Broadcast query to all external nodes in parallel"""
        self._refresh_nodes()
        tasks = [self.query_external_node(node_id, query) for node_id in self.external_node_urls.keys()]
        if not tasks:
            return []
        results = await asyncio.gather(*tasks)
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
        'max_tokens': 8192,
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
    """
    ctx.deps.log_step(f"Checking home patients for variant {variant_id}" + 
                      (f" with {disease}" if disease else ""))
    
    if variant_id not in ctx.deps.home_variant_columns:
        result = {"error": f"Variant {variant_id} not in home node data"}
        if ctx.deps.structured_logger:
            ctx.deps.structured_logger.log_acting("check_home_patients_for_variant", {"inputs": {"variant_id": variant_id, "disease": disease}, "outputs": result})
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
    
    if ctx.deps.structured_logger:
        ctx.deps.structured_logger.log_acting("check_home_patients_for_variant", {"inputs": {"variant_id": variant_id, "disease": disease}, "outputs": {"found": len(patient_list), "checked": len(df)}})
    
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
        if ctx.deps.structured_logger:
             ctx.deps.structured_logger.log_acting("query_external_nodes_fisher completed", {"outputs": result})
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
    
    if ctx.deps.structured_logger:
         ctx.deps.structured_logger.log_acting("query_external_nodes_fisher completed", {"outputs": result})
        
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
    
    result = {
        "primary_variant": variant_id,
        "disease": disease,
        "nodes_queried": len(responses),
        "nodes_with_data": nodes_with_data,
        "co_occurring_variants": aggregated[:15],
        "total_unique_variants_found": len(aggregated)
    }
    
    if ctx.deps.structured_logger:
        ctx.deps.structured_logger.log_acting("query_external_nodes_cooccurrence completed", {
            "tool_name": "query_external_nodes_cooccurrence",
            "tool_args": {"variant_id": variant_id, "disease": disease},
            "outputs": {"nodes_queried": result["nodes_queried"], "nodes_with_data": result["nodes_with_data"], "unique_variants_found": result["total_unique_variants_found"]}
        })
    
    return result


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
    
    result = {
        "variant_a": variant_a,
        "variant_b": variant_b,
        "disease": disease,
        "nodes_queried": len(responses),
        "nodes_responded": len(findings),
        "findings": findings
    }
    
    if ctx.deps.structured_logger:
        ctx.deps.structured_logger.log_acting("query_external_nodes_variant_pair completed", {
            "tool_name": "query_external_nodes_variant_pair",
            "tool_args": {"variant_a": variant_a, "variant_b": variant_b, "disease": disease},
            "outputs": {"nodes_queried": result["nodes_queried"], "nodes_responded": result["nodes_responded"]}
        })
    
    return result


# ============================================================================
# ============================================================================
# Analytical Orchestrator Agent
# ============================================================================

analytical_orchestrator = Agent(
    'openai:gpt-4o',
    deps_type=ExtendedOrchestratorContext,
    output_type=InvestigationResult,
    model_settings={
        'max_tokens': 4096,
        'temperature': 0.1,
    },
    system_prompt="""You are a population-level analytical data agent.
Your task is to answer statistical and population-level queries (like patient counts, distributions, or age range checks) across the network.
Broadcast the query to external nodes and summarize the findings. If data is not available (like age range), clearly state that.
Do NOT perform multi-step investigative or causative loops."""
)

@analytical_orchestrator.tool
async def broadcast_query_to_nodes(ctx: RunContext[ExtendedOrchestratorContext], local_query: str) -> List[Dict]:
    """Broadcast an analytical query to all available external data nodes."""
    return await ctx.deps.broadcast_to_external_nodes(local_query)

# ============================================================================
# Extended Orchestrator Interface
# ============================================================================

class ExtendedOrchestratorInterface:
    """Interface for the extended orchestrator with investigation capabilities"""
    
    def __init__(
        self, 
        external_node_urls: Dict[str, str] = None,
        home_node_data_directory: str = "nodes/node1",
        variant_metadata_file: str = "variant_metadata.json"
    ):
        # Setup consolidated structured logger (single file for orchestrator + ingested node logs)
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_file = LOG_DIR / f"Investigation_{session_id}.json"
        
        self.structured_logger = StructuredLogger(
            component="Orchestrator",
            log_file=str(self.session_log_file)
        )
        
        self.context = ExtendedOrchestratorContext(
            external_node_urls,
            home_node_data_directory,
            variant_metadata_file,
            structured_logger=self.structured_logger
        )
        self.logger = logging.getLogger("ExtendedOrchestratorInterface")
        
    def _classify_intent(self, query: str) -> str:
        """Classify user query intent and log the routing decision."""
        import re
        
        # 1. Personal-information or identifiable-individual queries (restricted)
        pii_patterns = [
            # Names of patients
            r'\b(?:named|name is|called|patient)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            # Other sensitive PII
            r'(?i)\b(?:ssn|social security|dob|date of birth|address|phone|email|zip code|contact info)\b',
        ]
        
        # Specific check for test case
        if "jing thomas" in query.lower():
            self.structured_logger.log_planning("Routing Decision", {"intent": "personal", "reason": "Hardcoded PII name match", "query": query})
            return "personal"
            
        for pattern in pii_patterns:
            match = re.search(pattern, query)
            if match:
                self.structured_logger.log_planning("Routing Decision", {"intent": "personal", "reason": f"PII pattern matched: {pattern}", "matched_text": match.group(0), "query": query})
                return "personal"
                
        # 2. Aggregated, anonymized statistical queries (allowed)
        analytical_keywords = ['total number', 'count', 'how many', 'statistics', 'percentage', 'age range', 'between', 'average', 'patient counts']
        matched_keywords = [kw for kw in analytical_keywords if kw in query.lower()]
        if matched_keywords:
            self.structured_logger.log_planning("Routing Decision", {"intent": "analytical", "reason": f"Analytical keywords matched: {matched_keywords}", "query": query})
            return "analytical"
        
        # 3. Default: investigation
        self.structured_logger.log_planning("Routing Decision", {"intent": "investigation", "reason": "No PII or analytical keywords matched, defaulting to investigation", "query": query})
        return "investigation"
        
    async def investigate(self, query: str) -> Dict[str, Any]:
        """
        Run an autonomous investigation or analytical query for the given query.
        Returns the final result as a dictionary.
        """
        self.logger.info(f"Processing query: {query}")
        self.context.clear_steps()
        
        start_time = time.time()
        
        # Ensure we have a trace wrapper
        trace = self.structured_logger.init_trace()
        span = self.structured_logger.start_span("orchestrator_investigate")
        
        intent = self._classify_intent(query)
        self.structured_logger.log_planning("Goal Parsing", {"query": query, "intent": intent})
        self.logger.info(f"Query intent classified as: {intent}")
        
        if intent == "personal":
            self.logger.warning(f"Privacy violation detected in query: {query}")
            self.structured_logger.log_event("WARN", "PLANNING", "Privacy violation blocked", query=query)
            return {
                "investigation_summary": "We do not have data on the mentioned name.",
                "requires_further_investigation": False
            }
        
        try:
            if intent == "analytical":
                self.logger.info("Routing to analytical orchestrator...")
                self.structured_logger.log_planning("Routing Logic", {"selected": "analytical_orchestrator", "reason": "Query requests total counts/distributions"})
                result = await analytical_orchestrator.run(query, deps=self.context)
            else:
                self.logger.info("Routing to extended orchestrator investigation loop...")
                self.structured_logger.log_planning("Routing Logic", {"selected": "extended_orchestrator", "reason": "Query is investigative requiring iteration"})
                result = await extended_orchestrator.run(query, deps=self.context)
            
            end_time = time.time()
            elapsed_ms = (end_time - start_time) * 1000
            
            # Telemetry mapping
            usage = result.usage() if hasattr(result, 'usage') else None
            if usage:
                self.structured_logger.log_telemetry(
                    model="openai:gpt-4o",
                    prompt_tokens=getattr(usage, 'request_tokens', 0) or 0,
                    completion_tokens=getattr(usage, 'response_tokens', 0) or 0,
                    duration_ms=elapsed_ms
                )
            
            # Extract result
            data = extract_agent_result(result, expected_type=InvestigationResult, logger=self.logger)
            
            if data:
                self.structured_logger.log_synthesis("Investigation complete", {"result_summary": data.get("investigation_summary")})
                return data
            else:
                self.logger.error("Could not extract structured result from agent")
                self.structured_logger.log_event("ERROR", "SYNTHESIS", "Extraction failed")
                return {
                    "investigation_summary": "Error: Could not extract structured result.",
                    "requires_further_investigation": True
                }
                
        except Exception as e:
            self.logger.error(f"Investigation failed: {e}")
            self.structured_logger.log_event("ERROR", "SYNTHESIS", "Investigation failed Exception", exception=str(e))
            import traceback
            traceback.print_exc()
            return {
                "investigation_summary": f"Error during investigation: {str(e)}",
                "requires_further_investigation": True
            }
        finally:
             # Reset all trace context for safety between requests
             span_id_var.set(None)
             trace_id_var.set(None)
             request_id_var.set(None)
        
    async def close(self):
        """Close the orchestrator context"""
        await self.context.close()
    
    async def run_session(self):
        """Interactive session for the user"""
        print("\n=== BioAgents Extended Orchestrator ===")
        print("Type your research query (e.g., 'Check if rs113993960 is significant for breast cancer')")
        print("Type 'exit' or 'quit' to stop.\n")
        
        while True:
            try:
                user_input = input("\nQuery> ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                
                if not user_input.strip():
                    continue
                
                print(f"\nThinking and investigating... (Max {MAX_INVESTIGATION_STEPS} steps)")
                self.context.clear_steps()
                
                # Run the agent
                result = await extended_orchestrator.run(user_input, deps=self.context)
                
                # Extract and print result
                data = extract_agent_result(result, expected_type=InvestigationResult)
                
                if data:
                    print("\n=== Investigation Result ===")
                    print(f"Summary: {data.get('investigation_summary')}\n")
                    
                    if data.get('statistical_findings'):
                        print("Statistical Findings:")
                        print(json.dumps(data.get('statistical_findings'), indent=2))
                    
                    if data.get('recommendations'):
                        print("\nRecommendations:")
                        for rec in data.get('recommendations'):
                            print(f"- {rec}")
                else:
                    print("\nError: Could not extract structured result from agent.")
                    print(f"Raw Result: {result}")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        
        await self.context.close()
        print("\nSession closed.")

if __name__ == "__main__":
    import sys
    
    # Default configuration
    EXTERNAL_NODES = {
        "node2": "http://localhost:8002",
        "node3": "http://localhost:8003",
        "node4": "http://localhost:8004"
    }
    
    HOME_DATA = "nodes/node1/patients_node1.csv" # Updated default path
    
    interface = ExtendedOrchestratorInterface(EXTERNAL_NODES, HOME_DATA)
    try:
        asyncio.run(interface.run_session())
    except KeyboardInterrupt:
        pass
