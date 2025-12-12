#!/usr/bin/env python3
"""Quick check for pydantic-ai Agent parameter names"""

import inspect

try:
    from pydantic_ai import Agent
    
    # Get the signature of Agent.__init__
    sig = inspect.signature(Agent.__init__)
    params = list(sig.parameters.keys())
    
    print("Agent.__init__ parameters:")
    for p in params:
        param = sig.parameters[p]
        print(f"  - {p}: {param.annotation if param.annotation != inspect.Parameter.empty else 'any'}")
    
    # Check specifically for output/result type
    if 'output_type' in params:
        print("\n✓ Uses 'output_type' parameter")
    elif 'result_type' in params:
        print("\n✓ Uses 'result_type' parameter")
    else:
        print("\n⚠ Neither 'output_type' nor 'result_type' found")
        
except ImportError:
    print("pydantic-ai not installed")
except Exception as e:
    print(f"Error: {e}")
