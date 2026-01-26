#!/usr/bin/env python3
"""
Launch BioAgents Extended System
- Node 1 = YOUR home node (direct patient access)
- Nodes 2-4 = External nodes (aggregated stats only)
"""

import shutil
import subprocess
import time
import signal
import sys
import os
import asyncio
import httpx
import json
import traceback


from orchestrator_agent import ExtendedOrchestratorInterface


def start_external_nodes():
    """Start external node agents (nodes 2-4 only)"""
    processes = []
    
    # Create node_log folder if it doesn't exist
    log_dir = "node_log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    print("Starting BioAgents Extended System...")
    print("-" * 50)
    print("🏠 Node 1 = YOUR HOME NODE (direct patient access)")
    print("🌐 Nodes 2-4 = External nodes (privacy-preserving)")
    print("-" * 50)
    
    # Check for required files
    for i in range(1, 5):
        if not os.path.exists(f"patients_node{i}.csv"):
            print(f"Error: patients_node{i}.csv not found!")
            print("Please run: python generate_patient_data.py")
            return None
    
    if not os.path.exists("variant_metadata.json"):
        print("Error: variant_metadata.json not found!")
        return None
    
    # Start EXTERNAL nodes only (2, 3, 4)
    # Node 1 is home node - accessed directly, not via HTTP
    for i in range(2, 5):
        cmd = [
            sys.executable, 
            "node_agent.py",
            f"node{i}",
            str(5000 + i),
            f"patients_node{i}.csv"
        ]
        
        print(f"Starting external node{i} agent on port {5000 + i}...")
        
        # Log stdout/stderr to files in node_log folder
        stdout_log = open(f"node_log/node{i}.log", "a", encoding="utf-8")
        stderr_log = open(f"node_log/node{i}.err.log", "a", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            stdout=stdout_log,
            stderr=stderr_log
        )
        processes.append(process)
        time.sleep(0.5)
    
    print(f"\n✓ External nodes started!")
    print(f"\nNetwork configuration:")
    print(f"  🏠 node1 (HOME): Direct access to patients_node1.csv")
    print(f"  🌐 node2: http://localhost:5002 (external)")
    print(f"  🌐 node3: http://localhost:5003 (external)")
    print(f"  🌐 node4: http://localhost:5004 (external)")
    
    return processes


def stop_processes(processes):
    """Stop all processes"""
    if not processes:
        return
    print("\n\nStopping external node agents...")
    for p in processes:
        p.terminate()
    for p in processes:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    print("All agents stopped.")


async def test_external_nodes(timeout_per_node: float = 15.0, interval: float = 0.2):
    """Test that external nodes are responding"""
    print("\nTesting external nodes...")
    client = httpx.AsyncClient()
    all_healthy = True

    async def wait_for_health(url: str, timeout: float, interval: float):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = await client.get(url, timeout=2.0)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(interval)
        return False

    for i in range(2, 5):
        url = f"http://localhost:{5000 + i}/health"
        ok = await wait_for_health(url, timeout_per_node, interval)
        if ok:
            print(f"  ✓ node{i}: healthy")
        else:
            print(f"  ✗ node{i}: not responding (still down after {timeout_per_node}s). See node_log/node{i}.log / .err.log")
            all_healthy = False

    await client.aclose()
    
    # Clean up node_log folder if all nodes are healthy
    if all_healthy:
        log_dir = "node_log"
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            print(f"\n✓ All nodes healthy. Cleaned up {log_dir} folder.")
    else:
        print(f"\n⚠️  Some nodes unhealthy. Logs retained in node_log/ folder for debugging.")
    
    return all_healthy

# async def test_external_nodes():
#     """Test that external nodes are responding"""    
#     print("\nTesting external nodes...")
#     client = httpx.AsyncClient(timeout=15.0)
#     all_healthy = True
    
#     for i in range(2, 5):
#         try:
#             response = await client.get(f"http://localhost:{5000 + i}/health")
#             if response.status_code == 200:
#                 print(f"  ✓ node{i}: healthy")
#             else:
#                 print(f"  ✗ node{i}: unhealthy")
#                 all_healthy = False
#         except Exception as e:
#             print(f"  ✗ node{i}: not responding ({e})")
#             all_healthy = False
    
#     await client.aclose()
#     return all_healthy


async def run_extended_session():
    """Run the extended orchestrator session"""
    
    external_nodes = {
        "node2": "http://localhost:5002",
        "node3": "http://localhost:5003",
        "node4": "http://localhost:5004"
    }
    
    orchestrator = ExtendedOrchestratorInterface(
        external_node_urls=external_nodes,
        home_node_data_file="patients_node1.csv"  # Home node - direct access
    )
    
    print("\n" + "="*70)
    print("🔬 BioAgents EXTENDED - Investigative Agent")
    print("="*70)
    print("\n📋 YOUR CAPABILITIES:")
    print("   • Direct access to YOUR patients (home node)")
    print("   • Query external nodes for population statistics")
    print("   • Autonomous investigation loops")
    print("\n📝 EXAMPLE QUERIES:")
    print("   • I suspect rs113993960 is causal for my CF patients. Investigate it.")
    print("   • Which of my breast cancer patients have BRCA1 variants? Check external data too.")
    print("   • Investigate rs334 for sickle cell - check my patients and find co-occurring variants.")
    print("   • My patient node1_P0015 has CF - what variants might be relevant?")
    print("\nType 'exit' to quit\n")
    
    while True:
        try:
            query = input("🔬 Investigation query: ").strip()
            
            if query.lower() == 'exit':
                break
            if not query:
                continue
            
            print("\n🤖 Agent is investigating (autonomous loop)...\n")
            
            result = await orchestrator.investigate(query)
            
            print("="*70)
            print("📊 INVESTIGATION COMPLETE")
            print("="*70)
            
            # Summary
            print(f"\n📝 Summary:")
            print(f"   {result.get('investigation_summary', 'N/A')}")
            
            # Steps taken
            steps = result.get('investigation_steps', [])
            if steps:
                print(f"\n🔄 Steps Taken ({len(steps)}):")
                for i, step in enumerate(steps, 1):
                    print(f"   {i}. {step}")
            
            # Affected patients
            patients = result.get('affected_patients', [])
            if patients:
                print(f"\n👥 Your Affected Patients ({len(patients)}):")
                for p in patients[:10]:
                    if isinstance(p, dict):
                        print(f"   • {p.get('patient_id', p)}: {p.get('disease', '')} "
                              f"({'case' if p.get('has_disease') else 'control'})")
                    else:
                        print(f"   • {p}")
            
            # Statistical findings
            stats = result.get('statistical_findings')
            if stats:
                print(f"\n📈 Statistical Findings:")
                if isinstance(stats, dict):
                    for k, v in stats.items():
                        print(f"   • {k}: {v}")
                else:
                    print(f"   {stats}")
            
            # Co-occurring variants
            covar = result.get('co_occurring_variants', [])
            if covar:
                print(f"\n🧬 Co-occurring Variants ({len(covar)}):")
                for v in covar[:7]:
                    rate = v.get('avg_co_occurrence_rate', 0)
                    rate_str = f"{rate:.1%}" if isinstance(rate, float) else str(rate)
                    print(f"   • {v.get('variant_id')} ({v.get('gene', '?')}) - "
                          f"{v.get('pathogenicity', '?')} - co-occur rate: {rate_str}")
            
            # Recommendations
            recs = result.get('recommendations', [])
            if recs:
                print(f"\n💡 Recommendations:")
                for rec in recs:
                    print(f"   • {rec}")
            
            # External nodes queried
            ext = result.get('external_nodes_queried', 0)
            if ext:
                print(f"\n🌐 External nodes queried: {ext}")
            
            print("\n" + "="*70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            traceback.print_exc()
    
    await orchestrator.close()


def main():
    """Main entry point"""
    processes = []
    
    def signal_handler(sig, frame):
        stop_processes(processes)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start external nodes
        # processes = start_external_nodes()
        processes = start_external_nodes()
        # print(processes) # Suppressed to clean output
        
        # Suppress httpx info logs
        import logging
        logging.getLogger("httpx").setLevel(logging.WARNING)

        if not processes:
            sys.exit(1)
        
        # Wait for initialization
        print("\nWaiting for external nodes to initialize...")
        time.sleep(3)
        
        # Test external nodes
        all_healthy = asyncio.run(test_external_nodes())
        
        if not all_healthy:
            print("\n⚠️  Warning: Some external nodes are not responding")
        
        # Run extended session
        print("\n" + "="*50)
        print("Ready for investigative queries!")
        print("="*50)
        
        asyncio.run(run_extended_session())
        
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        stop_processes(processes)
        print("\nGoodbye!")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           BioAgents EXTENDED - Investigative Agent               ║
║                                                                  ║
║   🏠 Home Node: Direct access to YOUR patient records            ║
║   🌐 External: Privacy-preserving aggregated statistics          ║
║   🔄 Autonomous: Agent loops until investigation complete        ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    main()
