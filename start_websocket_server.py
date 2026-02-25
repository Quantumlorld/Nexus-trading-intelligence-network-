#!/usr/bin/env python3
"""
Startup script for the Nexus WebSocket Server
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def main():
    """Start the WebSocket server"""
    
    print("🚀 NEXUS WEBSOCKET SERVER STARTUP")
    print("=" * 50)
    
    try:
        # Import WebSocket server
        from websocket.websocket_server import run_websocket_server
        
        print("✅ WebSocket server module imported successfully!")
        
        # Start the server
        print("\n🚀 Starting WebSocket server...")
        print("📡 Server will run on ws://localhost:8765")
        print("🔧 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        await run_websocket_server(host="localhost", port=8765)
        
    except KeyboardInterrupt:
        print("\n🛑 WebSocket server stopped by user")
    except Exception as e:
        print(f"❌ Error starting WebSocket server: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the server
    asyncio.run(main())
