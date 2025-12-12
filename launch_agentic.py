#!/usr/bin/env python3
"""
Launch the BioAgents Agentic System
Starts all node agents and provides options for interaction
"""

import subprocess
import time
import signal
import sys
import os
import asyncio

def start_node_agents():
    """Start all node agents"""
    processes = []
    
    print("Starting BioAgents Agentic System...")
    print("-" * 50)
    
    # Check for required files
    for i in range(1, 5):
        if not os.path.exists(f"patients_node{i}.csv"):
            print(f"Error: patients_node{i}.csv not found!")
            print("Please run: python generate_patient_data.py")
            return None
    
    if not os.path.exists("variant_metadata.json"):
        print("Error: variant_metadata.json not found!")
        print("Please run: python generate_patient_data.py")
        return None
    
    # Start node agents
    for i in range(1, 5):
        cmd = [
            sys.executable, 
            "node_agent.py",
            f"node{i}",
            str(5000 + i),
            f"patients_node{i}.csv"
        ]
        
        print(f"Starting node{i} agent on port {5000 + i}...")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append(process)
        time.sleep(0.5)
    
    print("\n✓ All node agents started!")
    print("\nNode agents running on:")
    for i in range(1, 5):
        print(f"  node{i}: http://localhost:{5000 + i}")
    
    return processes


def stop_processes(processes):
    """Stop all processes"""
    if not processes:
        return
    
    print("\n\nStopping all node agents...")
    for p in processes:
        p.terminate()
    
    for p in processes:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    
    print("All node agents stopped.")


async def test_agents():
    """Test that agents are responding"""
    import httpx
    
    print("\n" + "="*50)
    print("Testing node agents...")
    print("="*50)
    
    client = httpx.AsyncClient(timeout=5.0)
    all_healthy = True
    
    for i in range(1, 5):
        try:
            response = await client.get(f"http://localhost:{5000 + i}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✓ node{i}: healthy (agent: {data.get('agent', 'unknown')})")
            else:
                print(f"✗ node{i}: unhealthy (status {response.status_code})")
                all_healthy = False
        except Exception as e:
            print(f"✗ node{i}: not responding ({str(e)})")
            all_healthy = False
    
    await client.aclose()
    return all_healthy


async def run_interactive_session():
    """Run the interactive orchestrator"""
    from orchestrator_agent import OrchestratorInterface
    
    node_urls = {
        "node1": "http://localhost:5001",
        "node2": "http://localhost:5002",
        "node3": "http://localhost:5003",
        "node4": "http://localhost:5004"
    }
    
    orchestrator = OrchestratorInterface(node_urls)
    
    print("\n" + "="*70)
    print("BioAgents Agentic System - Natural Language Interface")
    print("="*70)
    print("\nExample queries you can ask:")
    print("  • Is CFTR rs113993960 significantly associated with Cystic Fibrosis?")
    print("  • What variants co-occur with rs80357906 in breast cancer patients?")
    print("  • Show me Fisher's test results for HBB rs334 and Sickle Cell Disease")
    print("  • Which variants are linked to Lynch Syndrome?")
    print("  • What's the odds ratio for rs121909298 in Huntington Disease?")
    print("\nType 'exit' to quit\n")
    
    while True:
        try:
            query = input("Your question: ").strip()
            
            if query.lower() in ['exit', 'quit']:
                break
            
            if not query:
                continue
            
            print("\n🤔 Thinking...")
            result = await orchestrator.process_query(query)
            
            print("\n" + "-"*50)
            print("📊 RESPONSE")
            print("-"*50)
            
            print(f"\n💡 Interpretation:\n{result['interpretation']}")
            print(f"\n📝 Summary:\n{result['summary']}")
            
            if result.get('detailed_results'):
                print("\n🔬 Statistical Details:")
                import json
                details = result['detailed_results']
                if isinstance(details, dict):
                    for key, value in details.items():
                        if key != 'combined_contingency':
                            print(f"  • {key}: {value}")
            
            if result.get('recommendations'):
                print("\n💊 Recommendations:")
                for rec in result['recommendations']:
                    print(f"  • {rec}")
            
            print(f"\n📡 Network: {result['nodes_with_data']}/{result['nodes_responded']} nodes had relevant data")
            print("-"*50 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    await orchestrator.close()


def main():
    """Main entry point"""
    processes = []
    
    def signal_handler(sig, frame):
        stop_processes(processes)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test-only":
            # Just test if agents are running
            asyncio.run(test_agents())
            return
        elif sys.argv[1] == "--orchestrator-only":
            # Just run orchestrator (assumes nodes are already running)
            try:
                asyncio.run(run_interactive_session())
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
            return
    
    # Full launch
    try:
        # Start node agents
        processes = start_node_agents()
        if not processes:
            sys.exit(1)
        
        # Wait for agents to initialize
        print("\nWaiting for agents to initialize...")
        time.sleep(3)
        
        # Test agents
        all_healthy = asyncio.run(test_agents())
        
        if not all_healthy:
            print("\n⚠️  Warning: Some agents are not healthy")
            print("You may experience issues with some queries")
        
        # Run interactive session
        print("\n" + "="*50)
        print("Ready for natural language queries!")
        print("="*50)
        
        asyncio.run(run_interactive_session())
        
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stop_processes(processes)
        print("\nGoodbye!")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    BioAgents Agentic System                      ║
║                 Natural Language Variant Analysis                 ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    main()
