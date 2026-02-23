import json
import os
import atexit
import threading
import time
from filelock import FileLock, Timeout

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_FILE = os.path.join(BASE_DIR, "service_registry.json")
LOCK_FILE = f"{REGISTRY_FILE}.lock"
_registry_lock = threading.Lock()

def _load_registry() -> dict:
    if not os.path.exists(REGISTRY_FILE):
        return {}
    try:
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def _save_registry(registry: dict):
    # Write to a temp file and rename for semi-atomic write
    temp_file = f"{REGISTRY_FILE}.{os.getpid()}.tmp"
    with open(temp_file, "w") as f:
        json.dump(registry, f, indent=4)
    os.replace(temp_file, REGISTRY_FILE)

def register_service(node_id: str, port: int):
    """Register a service with its dynamically allocated port."""
    with _registry_lock:
        try:
            with FileLock(LOCK_FILE, timeout=5):
                registry = _load_registry()
                registry[node_id] = {
                    "port": port,
                    "url": f"http://localhost:{port}"
                }
                _save_registry(registry)
        except Timeout:
            print(f"Warning: Could not acquire lock to register {node_id}")

def deregister_service(node_id: str):
    """Remove a service from the registry."""
    with _registry_lock:
        try:
            with FileLock(LOCK_FILE, timeout=5):
                registry = _load_registry()
                if node_id in registry:
                    del registry[node_id]
                    _save_registry(registry)
        except Timeout:
             print(f"Warning: Could not acquire lock to deregister {node_id}")


def get_service_map() -> dict:
    """Get the current map of node_id to service URLs."""
    with _registry_lock:
        try:
            with FileLock(LOCK_FILE, timeout=5):
                 return _load_registry()
        except Timeout:
             print("Warning: Could not acquire lock to read registry. Returning empty map.")
             return {}

_last_mtime = 0.0
_cached_registry = {}

def get_service_map_cached() -> dict:
    """Get the service map only if the registry file has changed."""
    global _last_mtime, _cached_registry
    
    if not os.path.exists(REGISTRY_FILE):
        return {}
        
    try:
        current_mtime = os.path.getmtime(REGISTRY_FILE)
        if current_mtime > _last_mtime:
            registry = get_service_map()
            if registry: # Only update cache if we successfully got it
                _cached_registry = registry
                _last_mtime = current_mtime
        return _cached_registry
    except OSError:
        return _cached_registry

def wait_for_nodes(expected_nodes: list, timeout: int = 60) -> bool:
    """Wait for a specific list of nodes to register."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        registry = get_service_map_cached()
        if all(node in registry for node in expected_nodes):
            return True
        time.sleep(0.5)
    return False

def deregister_on_exit(node_id: str):
    """Helper to ensure service is deregistered when process exits."""
    atexit.register(lambda: deregister_service(node_id))
