#!/usr/bin/env python3
"""
BioAgents Node Agent - Pydantic-AI based agent for each node
FIXED VERSION: Handles pydantic-ai result extraction properly
"""

from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
import json
import logging
from flask import Flask, request, jsonify
import dotenv
import asyncio


dotenv.load_dotenv()  

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class QueryRequest(BaseModel):
    """Natural language query from orchestrator"""
    query: str
    context: Optional[Dict[str, Any]] = Field(default={})


class StatisticalResult(BaseModel):
    """Statistical analysis result"""
    analysis_type: str
    variant_id: Optional[str] = None
    disease: Optional[str] = None
    p_value: Optional[float] = None
    odds_ratio: Optional[float] = None
    confidence_interval: Optional[List[float]] = None
    contingency_table: Optional[Dict[str, int]] = None
    sample_size: int
    interpretation: str
    raw_data: Dict[str, Any] = Field(default_factory=dict)


class NodeResponse(BaseModel):
    """Response from node agent"""
    node_id: str
    has_data: bool
    message: str
    results: Optional[List[StatisticalResult]] = None
    available_diseases: Optional[List[str]] = None
    available_variants: Optional[List[str]] = None


class NodeAgentContext:
    """Context for the node agent containing local data"""
    
    def __init__(self, node_id: str, patient_data_file: str, variant_metadata_file: str = "variant_metadata.json"):
        self.node_id = node_id
        self.logger = logging.getLogger(f"NodeAgent-{node_id}")
        
        # Load patient data
        self.patients = pd.read_csv(patient_data_file)
        
        # Normalize headers
        header_map = {
            "PatientID": "patient_id", "SUBJ_NO": "patient_id", "pid": "patient_id", "Patient_Identifier": "patient_id",
            "disease_name": "disease", "DX_T2D": "disease", "condition": "disease",
            "disease_condition": "has_disease", "disease_status": "has_disease", "status": "has_disease", "diagnosis_status": "has_disease",
            "Sex": "sex", "gender": "sex", "Gender": "sex"
        }
        self.patients.rename(columns=header_map, inplace=True)
        
        # Convert boolean columns properly
        bool_columns = ['has_disease'] + [col for col in self.patients.columns if col.startswith('rs')]
        for col in bool_columns:
            if col in self.patients.columns:
                self.patients[col] = self.patients[col].map({'True': True, 'False': False, True: True, False: False})
        
        # Load variant metadata
        try:
            with open(variant_metadata_file, 'r') as f:
                self.variant_metadata = json.load(f)
        except:
            self.variant_metadata = {}
        
        # Get available data
        self.variant_columns = [col for col in self.patients.columns if col.startswith('rs')]
        self.diseases = list(self.patients['disease'].unique())
        
        self.logger.info(f"Node {node_id} initialized with {len(self.patients)} patients, "
                        f"{len(self.diseases)} diseases, {len(self.variant_columns)} variants")


# Helper function to convert numpy types to Python types
def to_python_type(value):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    return value


def extract_agent_result(result, logger=None) -> Optional[Dict]:
    """
    Extract the actual data from a pydantic-ai AgentRunResult.
    Handles different versions of the pydantic-ai API.
    """
    if logger:
        logger.info(f"Extracting result from type: {type(result)}")
        logger.info(f"Result attributes: {dir(result)}")
    
    # Already a dict
    if isinstance(result, dict):
        return result
    
    # Already a NodeResponse
    if isinstance(result, NodeResponse):
        return result.model_dump()
    
    # Try various attribute names used in different pydantic-ai versions
    for attr_name in ['data', 'output', 'result', 'value', 'response']:
        if hasattr(result, attr_name):
            attr_value = getattr(result, attr_name)
            if logger:
                logger.info(f"Found attribute '{attr_name}' with type: {type(attr_value)}")
            
            if attr_value is None:
                continue
                
            if isinstance(attr_value, NodeResponse):
                return attr_value.model_dump()
            elif isinstance(attr_value, dict):
                return attr_value
            elif hasattr(attr_value, 'model_dump'):
                return attr_value.model_dump()
            elif hasattr(attr_value, 'dict'):
                return attr_value.dict()
    
    # Try to access as if it's the output directly (some versions)
    if hasattr(result, 'model_dump'):
        try:
            return result.model_dump()
        except Exception as e:
            if logger:
                logger.warning(f"model_dump() failed: {e}")
    
    # Last resort: try to convert to dict
    if hasattr(result, '__dict__'):
        if logger:
            logger.info(f"Trying __dict__: {result.__dict__}")
        # Check if there's a nested result object
        for key, value in result.__dict__.items():
            if isinstance(value, NodeResponse):
                return value.model_dump()
            elif isinstance(value, dict) and 'node_id' in value:
                return value
    
    return None


# Create the node agent with output_type
node_agent = Agent(
    'openai:gpt-4o',
    deps_type=NodeAgentContext,
    output_type=NodeResponse,
    model_settings={
        'max_tokens': 8192,  # Increase from default
    },
    system_prompt="""You are a node agent in a distributed biomedical data network. 
    You have access to local patient data with genetic variants and disease information.
    
    Your responsibilities:
    1. Understand queries about variant-disease associations
    2. Check if you have the requested data
    3. Perform statistical analyses (Fisher's exact test, co-occurrence analysis)
    4. Return results in a structured NodeResponse format
    5. Protect privacy by only returning aggregated statistics
    
    When you respond, you MUST return a valid NodeResponse with:
    - node_id: Your node identifier
    - has_data: Whether you have relevant data
    - message: A brief description
    - results: List of StatisticalResult objects (if applicable)
    
    Available tools:
    - check_available_data: See what data this node has
    - perform_fisher_test: Run Fisher's exact test for variant-disease association
    - find_co_occurring_variants: Find variants that co-occur with a given variant
    - get_variant_frequencies: Get variant frequencies across diseases
    
    Always be precise about what data you have and don't have.
    If you don't have specific data requested, set has_data=False and explain what's missing."""
)


@node_agent.tool
async def check_available_data(ctx: RunContext[NodeAgentContext]) -> Dict[str, Any]:
    """Check what data is available at this node"""
    return {
        "node_id": ctx.deps.node_id,
        "total_patients": int(len(ctx.deps.patients)),
        "diseases": ctx.deps.diseases,
        "total_variants": int(len(ctx.deps.variant_columns)),
        "sample_variants": ctx.deps.variant_columns[:5]
    }


@node_agent.tool
async def perform_fisher_test(
    ctx: RunContext[NodeAgentContext],
    variant_id: str,
    disease: str
) -> Dict[str, Any]:
    """Perform Fisher's exact test for variant-disease association"""
    
    # Check if we have the disease
    if disease not in ctx.deps.diseases:
        return {"error": f"No data for disease: {disease}"}
    
    # Check if we have the variant
    if variant_id not in ctx.deps.variant_columns:
        return {"error": f"No data for variant: {variant_id}"}
    
    # Filter for disease
    disease_data = ctx.deps.patients[ctx.deps.patients['disease'] == disease]
    
    # Build contingency table - convert to Python bool for comparison
    A = int(len(disease_data[(disease_data[variant_id] == True) & (disease_data['has_disease'] == True)]))
    B = int(len(disease_data[(disease_data[variant_id] == True) & (disease_data['has_disease'] == False)]))
    C = int(len(disease_data[(disease_data[variant_id] == False) & (disease_data['has_disease'] == True)]))
    D = int(len(disease_data[(disease_data[variant_id] == False) & (disease_data['has_disease'] == False)]))
    
    # Calculate Fisher's test
    contingency = [[A, B], [C, D]]
    odds_ratio, p_value = fisher_exact(contingency)
    
    # Calculate confidence interval
    if A > 0 and B > 0 and C > 0 and D > 0:
        log_or = np.log(odds_ratio)
        se = np.sqrt(1/A + 1/B + 1/C + 1/D)
        ci_lower = np.exp(log_or - 1.96 * se)
        ci_upper = np.exp(log_or + 1.96 * se)
    else:
        ci_lower = 0.0
        ci_upper = float('inf') if odds_ratio > 1 else 1.0
    
    # Get variant info
    variant_info = ctx.deps.variant_metadata.get(variant_id, {})
    
    return {
        "variant_id": variant_id,
        "disease": disease,
        "gene": variant_info.get("gene", "Unknown"),
        "p_value": float(p_value),
        "odds_ratio": float(odds_ratio) if not np.isinf(odds_ratio) else "infinity",
        "confidence_interval": [float(ci_lower), float(ci_upper) if not np.isinf(ci_upper) else "infinity"],
        "contingency_table": {
            "variant_disease": int(A),
            "variant_no_disease": int(B),
            "no_variant_disease": int(C),
            "no_variant_no_disease": int(D)
        },
        "sample_size": int(len(disease_data)),
        "significant": bool(p_value < 0.05)
    }


@node_agent.tool
async def find_co_occurring_variants(
    ctx: RunContext[NodeAgentContext],
    variant_id: str,
    disease: str,
    min_co_occurrence: float = 0.1
) -> Dict[str, Any]:
    """Find variants that co-occur with the given variant in disease cases"""
    
    # Check data availability
    if disease not in ctx.deps.diseases:
        return {"error": f"No data for disease: {disease}"}
    
    if variant_id not in ctx.deps.variant_columns:
        return {"error": f"No data for variant: {variant_id}"}
    
    # Get patients with the disease and variant
    disease_data = ctx.deps.patients[ctx.deps.patients['disease'] == disease]
    target_patients = disease_data[
        (disease_data['has_disease'] == True) & 
        (disease_data[variant_id] == True)
    ]
    
    if len(target_patients) == 0:
        return {
            "message": f"No patients found with both {disease} and {variant_id}",
            "sample_size": 0
        }
    
    co_occurring = []
    
    for other_variant in ctx.deps.variant_columns:
        if other_variant == variant_id:
            continue
        
        # Convert to int explicitly
        co_occur_count = int(target_patients[other_variant].sum())
        
        if co_occur_count > 0:
            co_occur_rate = co_occur_count / len(target_patients)
            
            if co_occur_rate >= min_co_occurrence:
                metadata = ctx.deps.variant_metadata.get(other_variant, {})
                co_occurring.append({
                    "variant_id": other_variant,
                    "gene": metadata.get("gene", "Unknown"),
                    "co_occurrence_rate": float(co_occur_rate),
                    "count": int(co_occur_count),
                    "pathogenicity": metadata.get("pathogenicity", "Unknown")
                })
    
    # Sort by co-occurrence rate
    co_occurring.sort(key=lambda x: x["co_occurrence_rate"], reverse=True)
    
    return {
        "primary_variant": variant_id,
        "disease": disease,
        "co_occurring_variants": co_occurring[:10],  # Top 10
        "sample_size": int(len(target_patients)),
        "total_found": int(len(co_occurring))
    }


@node_agent.tool  
async def get_variant_frequencies(
    ctx: RunContext[NodeAgentContext],
    variant_id: str
) -> Dict[str, Any]:
    """Get frequency of a variant across all diseases"""
    
    if variant_id not in ctx.deps.variant_columns:
        return {"error": f"No data for variant: {variant_id}"}
    
    frequencies = {}
    for disease in ctx.deps.diseases:
        disease_data = ctx.deps.patients[ctx.deps.patients['disease'] == disease]
        cases = disease_data[disease_data['has_disease'] == True]
        controls = disease_data[disease_data['has_disease'] == False]
        
        case_freq = cases[variant_id].sum() / len(cases) if len(cases) > 0 else 0
        control_freq = controls[variant_id].sum() / len(controls) if len(controls) > 0 else 0
        
        frequencies[disease] = {
            "case_frequency": float(case_freq),
            "control_frequency": float(control_freq),
            "case_count": int(cases[variant_id].sum()),
            "control_count": int(controls[variant_id].sum()),
            "total_cases": int(len(cases)),
            "total_controls": int(len(controls))
        }
    
    return {
        "variant_id": variant_id,
        "frequencies_by_disease": frequencies
    }


class NodeAgentServer:
    """Flask server for the node agent"""
    
    def __init__(self, node_id: str, port: int, patient_data_file: str):
        self.app = Flask(f"node_agent_{node_id}")
        self.port = port
        self.context = NodeAgentContext(node_id, patient_data_file)
        self.logger = logging.getLogger(f"NodeServer-{node_id}")
        
        self._setup_routes()
    
    def _setup_routes(self):
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                "status": "healthy",
                "node_id": self.context.node_id,
                "agent": "pydantic-ai"
            })
        
        @self.app.route('/query', methods=['POST'])
        def handle_query():
            try:
                data = request.json
                query_text = data.get('query', '')
                
                self.logger.info(f"Received query: {query_text}")
                
                # Run the agent - use existing event loop or create new one
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        raise RuntimeError("Event loop is closed")
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                result = loop.run_until_complete(
                    node_agent.run(query_text, deps=self.context)
                )
                
                # Log the raw result for debugging
                self.logger.info(f"Raw result type: {type(result)}")
                self.logger.info(f"Raw result attributes: {[a for a in dir(result) if not a.startswith('_')]}")
                
                # Use the improved extraction function
                response_dict = extract_agent_result(result, self.logger)
                
                if response_dict:
                    self.logger.info(f"Successfully extracted response: {list(response_dict.keys())}")
                    return jsonify(response_dict)
                else:
                    # Fallback: try to get as much info as possible
                    self.logger.warning(f"Could not extract response from: {type(result)}")
                    self.logger.warning(f"Result repr: {repr(result)[:500]}")
                    
                    # Try one more thing: access .data directly and log what we get
                    if hasattr(result, 'data'):
                        data_val = result.data
                        self.logger.warning(f"result.data type: {type(data_val)}, value: {repr(data_val)[:500]}")
                    
                    return jsonify({
                        "node_id": self.context.node_id,
                        "has_data": False,
                        "message": f"Could not extract structured response. Result type: {type(result).__name__}",
                        "results": None,
                        "debug_info": {
                            "result_type": str(type(result)),
                            "attributes": [a for a in dir(result) if not a.startswith('_')]
                        }
                    })
                    
            except Exception as e:
                self.logger.error(f"Error handling query: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    "node_id": self.context.node_id,
                    "has_data": False,
                    "message": f"Error: {str(e)}",
                    "results": None
                }), 500
    
    def run(self):
        self.logger.info(f"Starting node agent server on port {self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=False)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python node_agent.py <node_id> <port> <data_file>")
        sys.exit(1)
    
    node_id = sys.argv[1]
    port = int(sys.argv[2])
    data_file = sys.argv[3]
    
    server = NodeAgentServer(node_id, port, data_file)
    server.run()