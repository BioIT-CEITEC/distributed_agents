#!/usr/bin/env python3
"""
Diagnostic script to understand pydantic-ai result structure
Run this to see exactly what your version returns
"""

import asyncio
import sys

def check_pydantic_ai_version():
    """Check installed pydantic-ai version and API"""
    try:
        import pydantic_ai
        version = getattr(pydantic_ai, '__version__', 'unknown')
        print(f"✓ pydantic-ai version: {version}")
        
        # Check what's exported
        exports = [x for x in dir(pydantic_ai) if not x.startswith('_')]
        print(f"  Exports: {exports[:10]}...")
        
        return True
    except ImportError as e:
        print(f"✗ pydantic-ai not installed: {e}")
        return False


async def test_simple_agent():
    """Test a simple agent to see result structure"""
    from pydantic_ai import Agent
    from pydantic import BaseModel
    from typing import Optional
    import os
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠ OPENAI_API_KEY not set. Trying to load from .env...")
        try:
            import dotenv
            dotenv.load_dotenv()
            if os.getenv('OPENAI_API_KEY'):
                print("✓ Loaded API key from .env")
            else:
                print("✗ No API key found in .env")
                return
        except ImportError:
            print("✗ python-dotenv not installed")
            return
    
    class SimpleResponse(BaseModel):
        message: str
        number: int
    
    # Create a simple agent
    agent = Agent(
        'openai:gpt-4o-mini',  # Use cheaper model for testing
        output_type=SimpleResponse,
        system_prompt="You are a test agent. Always respond with a message and a number."
    )
    
    print("\n" + "="*60)
    print("Testing pydantic-ai agent result structure")
    print("="*60)
    
    # Run the agent
    print("\nRunning agent with query: 'Say hello and give me a number'...")
    result = await agent.run("Say hello and give me a number between 1 and 100")
    
    print(f"\n📊 Result Analysis:")
    print(f"  Type: {type(result)}")
    print(f"  Type name: {type(result).__name__}")
    print(f"  Module: {type(result).__module__}")
    
    # List all non-private attributes
    attrs = [a for a in dir(result) if not a.startswith('_')]
    print(f"\n  Public attributes: {attrs}")
    
    # Try to access each attribute
    print(f"\n  Attribute values:")
    for attr in attrs:
        try:
            value = getattr(result, attr)
            if not callable(value):
                value_str = repr(value)[:100]
                print(f"    {attr}: {type(value).__name__} = {value_str}")
        except Exception as e:
            print(f"    {attr}: ERROR - {e}")
    
    # Try specific extraction methods
    print(f"\n🔍 Extraction attempts:")
    
    # Method 1: .data
    if hasattr(result, 'data'):
        data = result.data
        print(f"  result.data: {type(data).__name__}")
        if hasattr(data, 'model_dump'):
            print(f"    .model_dump(): {data.model_dump()}")
        elif isinstance(data, dict):
            print(f"    (dict): {data}")
        else:
            print(f"    value: {repr(data)[:200]}")
    
    # Method 2: .output  
    if hasattr(result, 'output'):
        output = result.output
        print(f"  result.output: {type(output).__name__} = {repr(output)[:200]}")
    
    # Method 3: .response
    if hasattr(result, 'response'):
        response = result.response
        print(f"  result.response: {type(response).__name__}")
    
    # Method 4: Direct model_dump
    if hasattr(result, 'model_dump'):
        try:
            dump = result.model_dump()
            print(f"  result.model_dump(): {dump}")
        except Exception as e:
            print(f"  result.model_dump(): ERROR - {e}")
    
    print("\n" + "="*60)
    print("✓ Diagnostic complete!")
    print("="*60)
    
    # Show recommended extraction code
    if hasattr(result, 'data'):
        print("\n📝 Recommended extraction code for your version:")
        print("""
    result = await agent.run(query, deps=context)
    
    # Extract the response
    if hasattr(result, 'data'):
        response_obj = result.data
        if hasattr(response_obj, 'model_dump'):
            response_dict = response_obj.model_dump()
        elif isinstance(response_obj, dict):
            response_dict = response_obj
        else:
            response_dict = {"error": f"Unexpected data type: {type(response_obj)}"}
    else:
        response_dict = {"error": "No 'data' attribute on result"}
""")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           pydantic-ai Diagnostic Tool                            ║
║           Checking result structure for your version             ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    if not check_pydantic_ai_version():
        print("\nPlease install pydantic-ai: pip install pydantic-ai")
        sys.exit(1)
    
    try:
        asyncio.run(test_simple_agent())
    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
