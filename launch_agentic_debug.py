#!/usr/bin/env python3
"""
Launch the BioAgents Agentic System - DEBUG VERSION
Shows all node output for debugging
"""

import shutil
import subprocess
import time
import signal
import sys
import os
import asyncio
import httpx
import traceback


def start_node_agents_debug():
    """Start all node agents with visible output"""
    processes = []
    
    print("Starting BioAgents Agentic System (DEBUG MODE)...")
    print("-" * 50)
    
    log_dir = "node_log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Check for required files
    for i in range(1, 5):
        node_dir = f"nodes/node{i}"
        if not os.path.exists(node_dir):
            print(f"Error: {node_dir} directory not found!")
            return None
    
    if not os.path.exists("variant_metadata.json"):
        print("Error: variant_metadata.json not found!")
        print("Please run: python generate_patient_data.py")
        return None
    
    from core.discovery import wait_for_nodes, get_service_map
    expected_nodes = []
    
    # Start node agents with visible output
    for i in range(1, 5):
        node_id = f"node{i}"
        expected_nodes.append(node_id)
        cmd = [
            sys.executable, 
            "-m", "core.node_agent",
            node_id,
            "0",
            f"nodes/{node_id}"
        ]
        
        print(f"Starting {node_id} agent on dynamic port...")
        
        # Create log files
        stdout_log = open(f"node_log/{node_id}.log", "a", encoding="utf-8")
        stderr_log = open(f"node_log/{node_id}.err.log", "a", encoding="utf-8")
        
        process = subprocess.Popen(
            cmd,
            stdout=stdout_log,
            stderr=stderr_log
        )
        processes.append((process, stdout_log, stderr_log))
        time.sleep(0.5)
    
    print("\nWaiting for nodes to register their dynamic ports...")
    if not wait_for_nodes(expected_nodes, timeout=60):
        print("Error: Not all nodes registered in time!")
        return processes
        
    registry = get_service_map()
    print("\n✓ All node agents started and registered!")
    print("\nNode agents running on:")
    for nid in expected_nodes:
        print(f"  {nid}: {registry.get(nid, {}).get('url', 'Unknown')}")
    
    print("\n📝 Log files created:")
    for nid in expected_nodes:
        print(f"  node_log/{nid}.log")
        print(f"  node_log/{nid}.err.log")
    
    return processes


def stop_processes_debug(processes):
    """Stop all processes and close log files"""
    if not processes:
        return
    
    print("\n\nStopping all node agents...")
    for p, stdout, stderr in processes:
        p.terminate()
        stdout.close()
        stderr.close()
    
    for p, _, _ in processes:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    
    print("All node agents stopped.")


async def test_agents_debug():
    """Test that agents are responding and show errors"""

    print("\n" + "="*50)
    print("Testing node agents...")
    print("="*50)
    
    from core.discovery import get_service_map
    registry = get_service_map()
    all_healthy = True
    
    async with httpx.AsyncClient(timeout=5.0) as client:  # Use async with
        for i in range(1, 5):
            node_id = f"node{i}"
            if node_id not in registry:
                print(f"✗ {node_id}: not registered!")
                all_healthy = False
                continue
            try:
                response = await client.get(f"{registry[node_id]['url']}/health")
                if response.status_code == 200:
                    data = response.json()
                    print(f"✓ {node_id}: healthy (agent: {data.get('agent', 'unknown')})")
                else:
                    print(f"✗ {node_id}: unhealthy (status {response.status_code})")
                    all_healthy = False
            except Exception as e:
                print(f"✗ {node_id}: not responding ({str(e)})")
                all_healthy = False
                
                # Show recent log entries
                print(f"\n  Checking logs for {node_id}...")
                try:
                    with open(f"node_log/{node_id}.err.log", "r") as f:
                        stderr_content = f.read()
                        if stderr_content:
                            print(f"  Last error output:")
                            print("  " + "\n  ".join(stderr_content.split("\n")[-10:]))
                except:
                    pass
    
    # Clean up node_log folder if all nodes are healthy
    if all_healthy:
        log_dir = "node_log"
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            print(f"\n✓ All nodes healthy. Cleaned up {log_dir} folder.")
    else:
        print(f"\n⚠️  Some nodes unhealthy. Logs retained in node_log/ folder for debugging.")
        
    return all_healthy


def show_node_logs():
    """Show recent log entries from all nodes"""
    print("\n" + "="*70)
    print("Recent Node Logs")
    print("="*70)
    
    for i in range(1, 5):
        print(f"\n--- node_log/node{i} error log ---")
        try:
            with open(f"node_log/node{i}.err.log", "r") as f:
                lines = f.readlines()
                if lines:
                    # Show last 20 lines
                    for line in lines[-20:]:
                        print(line.rstrip())
                else:
                    print("(empty)")
        except FileNotFoundError:
            print("(log file not found)")
        
        print(f"\n--- node{i} stdout ---")
        try:
            with open(f"node_log/node{i}.log", "r") as f:
                lines = f.readlines()
                if lines:
                    # Show last 10 lines
                    for line in lines[-10:]:
                        print(line.rstrip())
                else:
                    print("(empty)")
        except FileNotFoundError:
            print("(log file not found)")


async def test_single_query():
    """Test a single query and show detailed error info"""
    
    from core.discovery import get_service_map
    registry = get_service_map()
    url = registry.get("node1", {}).get("url", "http://localhost:5001")
    
    print("\n" + "="*70)
    print("Testing Single Query to node1")
    print("="*70)
    
    test_query = {
        "query": "Is CFTR rs113993960 significantly associated with Cystic Fibrosis?",
        "context": {}
    }
    
    print(f"\nSending query: {test_query['query']}")
    
    client = httpx.AsyncClient(timeout=10.0)
    
    try:
        response = await client.post(
            f"{url}/query",
            json=test_query
        )
        
        print(f"\nResponse Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("\n✓ Success! Response:")
            print(response.json())
        else:
            print("\n✗ Error! Response:")
            print(response.text[:1000])
            
            # Show logs
            print("\n--- Checking node1 logs ---")
            time.sleep(0.5)  # Give it a moment to write
            with open("node_log/node1.err.log", "r") as f:
                stderr = f.read()
                if stderr:
                    print("\nSTDERR (last 50 lines):")
                    lines = stderr.split("\n")
                    for line in lines[-50:]:
                        print(line)
                        
    except Exception as e:
        print(f"\n✗ Connection Error: {e}")
        
    await client.aclose()


def main():
    """Main entry point"""
    processes = []
    
    def signal_handler(sig, frame):
        stop_processes_debug(processes)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--logs":
            show_node_logs()
            return
        elif sys.argv[1] == "--test-query":
            asyncio.run(test_single_query())
            return
        elif sys.argv[1] == "--test-only":
            asyncio.run(test_agents_debug())
            return
    
    # Full launch
    try:
        from core.discovery import _save_registry
        _save_registry({}) # clear registry on startup
        
        # Start node agents
        processes = start_node_agents_debug()
        if not processes:
            sys.exit(1)
        
        # Wait for agents to initialize
        print("\nWaiting for agents to initialize...")
        time.sleep(3)
        
        # Test agents
        all_healthy = asyncio.run(test_agents_debug())
        
        if not all_healthy:
            print("\n⚠️  Warning: Some agents are not healthy")
            print("\nShowing recent logs...")
            time.sleep(1)
            show_node_logs()
            
            print("\n" + "="*70)
            print("To view full logs, check:")
            for i in range(1, 5):
                print(f"  - node_log/node{i}.err.log")
                print(f"  - node_log/node{i}.log")
            print("="*70)
            
            response = input("\nContinue anyway? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                stop_processes_debug(processes)
                sys.exit(1)
        
        # Test a single query
        print("\nTesting a single query to node1...")
        asyncio.run(test_single_query())
        
        print("\n" + "="*70)
        print("Debug session complete!")
        print("="*70)
        print("\nOptions:")
        print("  1. Check logs: python launch_agentic_debug.py --logs")
        print("  2. Test query: python launch_agentic_debug.py --test-query")
        print("  3. Fix issues based on error messages above")
        
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        stop_processes_debug(processes)
        print("\nGoodbye!")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║            BioAgents Agentic System - DEBUG MODE                 ║
║                   Node Error Diagnostics                          ║
╚══════════════════════════════════════════════════════════════════╝

Options:
  python launch_agentic_debug.py              # Full debug session
  python launch_agentic_debug.py --logs       # Show recent logs
  python launch_agentic_debug.py --test-query # Test single query
  python launch_agentic_debug.py --test-only  # Only test health
    """)
    
    main()