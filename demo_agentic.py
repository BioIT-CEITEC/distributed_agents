#!/usr/bin/env python3
"""
Demonstration of the BioAgents Agentic System
Shows how natural language queries are processed by agents
"""

import asyncio
import httpx
from orchestrator_agent import OrchestratorInterface
import json


async def run_demonstrations():
    """Run example demonstrations of the agentic system"""
    
    # Setup orchestrator
    node_urls = {
        "node1": "http://localhost:5001",
        "node2": "http://localhost:5002",
        "node3": "http://localhost:5003",
        "node4": "http://localhost:5004"
    }
    
    orchestrator = OrchestratorInterface(node_urls)
    
    print("\n" + "="*70)
    print("BioAgents Agentic System - Demonstration")
    print("="*70)
    
    # Check if nodes are healthy
    print("\n🔍 Checking node agents...")
    client = httpx.AsyncClient(timeout=5.0)
    nodes_online = 0
    
    for node_id, url in node_urls.items():
        try:
            response = await client.get(f"{url}/health")
            if response.status_code == 200:
                print(f"  ✓ {node_id} is online")
                nodes_online += 1
            else:
                print(f"  ✗ {node_id} is not responding")
        except:
            print(f"  ✗ {node_id} is offline")
    
    await client.aclose()
    
    if nodes_online == 0:
        print("\n❌ No nodes are online!")
        print("Please run: python launch_agentic.py")
        return
    
    print(f"\n✓ {nodes_online}/4 nodes are online\n")
    
    # Demonstration queries
    demonstrations = [
        {
            "title": "Fisher's Exact Test Query",
            "query": "I need to know if CFTR variant rs113993960 is significantly associated with Cystic Fibrosis. Please perform a statistical analysis.",
            "explanation": "The agent understands this requires Fisher's exact test"
        },
        {
            "title": "Co-occurrence Analysis",
            "query": "What genetic variants commonly appear together with BRCA1 rs80357906 in patients who have Hereditary Breast Cancer?",
            "explanation": "The agent identifies this as a co-occurrence query"
        },
        {
            "title": "Vague Query Handling",
            "query": "Tell me about HBB variants in sickle cell",
            "explanation": "The agent interprets incomplete queries"
        },
        {
            "title": "Multi-Node Aggregation",
            "query": "Calculate the overall significance of rs334 for Sickle Cell Disease across all available data",
            "explanation": "The agent aggregates data from multiple nodes"
        },
        {
            "title": "Data Availability Check",
            "query": "Do you have any data about variants in the APOE gene?",
            "explanation": "The agent checks what data is available"
        }
    ]
    
    for i, demo in enumerate(demonstrations, 1):
        print(f"\n{'='*70}")
        print(f"DEMONSTRATION {i}: {demo['title']}")
        print(f"{'='*70}")
        
        print(f"\n📝 Query: \"{demo['query']}\"")
        print(f"💡 {demo['explanation']}")
        
        print("\n⏳ Processing...")
        
        try:
            result = await orchestrator.process_query(demo['query'])
            
            print("\n📊 Results:")
            print(f"├─ Interpretation: {result['interpretation']}")
            print(f"├─ Summary: {result['summary'][:200]}..." if len(result['summary']) > 200 else f"├─ Summary: {result['summary']}")
            print(f"└─ Nodes responded: {result['nodes_with_data']}/{result['nodes_responded']}")
            
            if result.get('detailed_results'):
                print("\n📈 Key Statistics:")
                details = result['detailed_results']
                if isinstance(details, dict):
                    # Show key statistics
                    if 'combined_p_value' in details:
                        print(f"  • P-value: {details['combined_p_value']}")
                    if 'combined_odds_ratio' in details:
                        print(f"  • Odds Ratio: {details['combined_odds_ratio']}")
                    if 'positive_predictive_value' in details:
                        print(f"  • PPV: {details['positive_predictive_value']}")
                    if 'co_occurring_variants' in details:
                        n_variants = len(details['co_occurring_variants'])
                        print(f"  • Co-occurring variants found: {n_variants}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        # Small delay between demos
        if i < len(demonstrations):
            await asyncio.sleep(1)
    
    await orchestrator.close()
    
    print("\n" + "="*70)
    print("Demonstration Complete")
    print("="*70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Natural language understanding")
    print("  ✓ Automatic query interpretation")
    print("  ✓ Multi-node coordination")
    print("  ✓ Statistical analysis")
    print("  ✓ Result aggregation")
    print("  ✓ Intelligent response generation")


async def quick_test():
    """Quick test of a single query"""
    
    print("\n🧪 Quick Test Mode")
    print("-" * 40)
    
    query = input("Enter your query (or press Enter for default): ").strip()
    
    if not query:
        query = "Is rs113993960 associated with Cystic Fibrosis?"
        print(f"Using default query: {query}")
    
    node_urls = {
        "node1": "http://localhost:5001",
        "node2": "http://localhost:5002",
        "node3": "http://localhost:5003",
        "node4": "http://localhost:5004"
    }
    
    orchestrator = OrchestratorInterface(node_urls)
    
    print("\n⏳ Processing...")
    
    try:
        result = await orchestrator.process_query(query)
        
        print("\n📊 Results:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    await orchestrator.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        asyncio.run(quick_test())
    else:
        asyncio.run(run_demonstrations())
