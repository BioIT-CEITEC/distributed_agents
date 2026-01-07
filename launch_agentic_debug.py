#!/usr/bin/env python3
"""
Launch the BioAgents Agentic System - DEBUG VERSION
Shows all node output for debugging
"""

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
    
    # Start node agents with visible output
    for i in range(1, 5):
        cmd = [
            sys.executable, 
            "node_agent.py",
            f"node{i}",
            str(5000 + i),
            f"patients_node{i}.csv"
        ]
        
        print(f"Starting node{i} agent on port {5000 + i}...")
        
        # Create log files
        stdout_log = open(f"node{i}_stdout.log", "w")
        stderr_log = open(f"node{i}_stderr.log", "w")
        
        process = subprocess.Popen(
            cmd,
            stdout=stdout_log,
            stderr=stderr_log
        )
        processes.append((process, stdout_log, stderr_log))
        time.sleep(0.5)
    
    print("\n✓ All node agents started!")
    print("\nNode agents running on:")
    for i in range(1, 5):
        print(f"  node{i}: http://localhost:{5000 + i}")
    
    print("\n📝 Log files created:")
    for i in range(1, 5):
        print(f"  node{i}_stdout.log")
        print(f"  node{i}_stderr.log")
    
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
    
    all_healthy = True
    
    async with httpx.AsyncClient(timeout=5.0) as client:  # Use async with
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
                
                # Show recent log entries
                print(f"\n  Checking logs for node{i}...")
                try:
                    with open(f"node{i}_stderr.log", "r") as f:
                        stderr_content = f.read()
                        if stderr_content:
                            print(f"  Last error output:")
                            print("  " + "\n  ".join(stderr_content.split("\n")[-10:]))
                except:
                    pass
    
    return all_healthy


def show_node_logs():
    """Show recent log entries from all nodes"""
    print("\n" + "="*70)
    print("Recent Node Logs")
    print("="*70)
    
    for i in range(1, 5):
        print(f"\n--- node{i} stderr ---")
        try:
            with open(f"node{i}_stderr.log", "r") as f:
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
            with open(f"node{i}_stdout.log", "r") as f:
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
            "http://localhost:5001/query",
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
            with open("node1_stderr.log", "r") as f:
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
                print(f"  - node{i}_stderr.log")
                print(f"  - node{i}_stdout.log")
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