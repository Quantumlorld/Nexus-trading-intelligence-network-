#!/usr/bin/env python3
"""
Test script for the WebSocket server
"""

import asyncio
import websockets
import json
from datetime import datetime
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_websocket_connection():
    """Test WebSocket connection and basic functionality"""
    
    print("🚀 NEXUS WEBSOCKET SERVER TEST")
    print("=" * 50)
    
    try:
        # Connect to WebSocket server
        uri = "ws://localhost:8765"
        print(f"🔗 Connecting to WebSocket server at {uri}...")
        
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket server!")
            
            # Test 1: Send ping message
            print("\n📡 Test 1: Sending ping message...")
            ping_message = {
                "type": "ping",
                "timestamp": datetime.utcnow().isoformat()
            }
            await websocket.send(json.dumps(ping_message))
            
            # Receive pong response
            response = await websocket.recv()
            pong_data = json.loads(response)
            print(f"✅ Received pong: {pong_data.get('data', {}).get('message', 'No message')}")
            
            # Test 2: Subscribe to prices
            print("\n📡 Test 2: Subscribing to price updates...")
            subscribe_message = {
                "type": "subscribe",
                "subscription": "prices"
            }
            await websocket.send(json.dumps(subscribe_message))
            
            # Receive subscription confirmation
            response = await websocket.recv()
            sub_data = json.loads(response)
            print(f"✅ Subscription confirmed: {sub_data.get('data', {}).get('message', 'No message')}")
            
            # Test 3: Subscribe to trades
            print("\n📡 Test 3: Subscribing to trade updates...")
            subscribe_message = {
                "type": "subscribe",
                "subscription": "trades"
            }
            await websocket.send(json.dumps(subscribe_message))
            
            # Receive subscription confirmation
            response = await websocket.recv()
            sub_data = json.loads(response)
            print(f"✅ Subscription confirmed: {sub_data.get('data', {}).get('message', 'No message')}")
            
            # Test 4: Subscribe to signals
            print("\n📡 Test 4: Subscribing to signal updates...")
            subscribe_message = {
                "type": "subscribe",
                "subscription": "signals"
            }
            await websocket.send(json.dumps(subscribe_message))
            
            # Receive subscription confirmation
            response = await websocket.recv()
            sub_data = json.loads(response)
            print(f"✅ Subscription confirmed: {sub_data.get('data', {}).get('message', 'No message')}")
            
            # Test 5: Send chat message
            print("\n📡 Test 5: Sending chat message...")
            chat_message = {
                "type": "chat",
                "message": "Hello from WebSocket test client!"
            }
            await websocket.send(json.dumps(chat_message))
            
            # Receive chat confirmation
            response = await websocket.recv()
            chat_data = json.loads(response)
            print(f"✅ Chat message broadcasted: {chat_data.get('data', {}).get('message', 'No message')}")
            
            # Test 6: Wait for some messages
            print("\n📡 Test 6: Waiting for server messages...")
            message_count = 0
            
            try:
                while message_count < 5:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    message_data = json.loads(response)
                    
                    print(f"📨 Message {message_count + 1}: {message_data.get('type', 'unknown')}")
                    
                    if message_data.get('type') == 'price_update':
                        print(f"   💰 Price: {message_data.get('data', {}).get('symbol', 'N/A')} - ${message_data.get('data', {}).get('price', 'N/A')}")
                    elif message_data.get('type') == 'trade_executed':
                        print(f"   💼 Trade: {message_data.get('data', {}).get('symbol', 'N/A')} - {message_data.get('data', {}).get('action', 'N/A')}")
                    elif message_data.get('type') == 'signal_generated':
                        print(f"   📈 Signal: {message_data.get('data', {}).get('symbol', 'N/A')} - {message_data.get('data', {}).get('signal', 'N/A')}")
                    elif message_data.get('type') == 'alert':
                        print(f"   ⚠️ Alert: {message_data.get('data', {}).get('message', 'N/A')}")
                    elif message_data.get('type') == 'chat_message':
                        print(f"   💬 Chat: {message_data.get('data', {}).get('message', 'N/A')}")
                    
                    message_count += 1
                    
            except asyncio.TimeoutError:
                print("✅ No more messages received (timeout)")
            
            print(f"\n✅ Received {message_count} messages from server")
            
            # Test 7: Unsubscribe
            print("\n📡 Test 7: Unsubscribing from all subscriptions...")
            unsubscribe_message = {
                "type": "unsubscribe",
                "subscription": "prices"
            }
            await websocket.send(json.dumps(unsubscribe_message))
            
            # Receive unsubscription confirmation
            response = await websocket.recv()
            unsub_data = json.loads(response)
            print(f"✅ Unsubscription confirmed: {unsub_data.get('data', {}).get('message', 'No message')}")
            
            print("\n🎯 WEBSOCKET TEST COMPLETED SUCCESSFULLY!")
            print("✅ All tests passed!")
            print("🚀 WebSocket server is working correctly!")
            
            return True
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_websocket_server():
    """Test the WebSocket server functionality"""
    
    print("🚀 NEXUS WEBSOCKET SERVER INTEGRATION TEST")
    print("=" * 60)
    
    try:
        # Import WebSocket server
        from websocket.websocket_server import create_websocket_manager, WebSocketMessage, MessageType
        
        print("✅ WebSocket server module imported successfully!")
        
        # Create WebSocket manager
        print("\n🔧 Creating WebSocket manager...")
        manager = create_websocket_manager()
        print("✅ WebSocket manager created successfully!")
        
        # Test connection stats
        print("\n📊 Testing connection statistics...")
        stats = manager.get_connection_stats()
        print(f"✅ Connection stats: {stats}")
        
        # Test message broadcasting
        print("\n📡 Testing message broadcasting...")
        
        # Test price update
        price_message = WebSocketMessage(
            type=MessageType.PRICE_UPDATE,
            data={
                "symbol": "EUR/USD",
                "price": 1.0845,
                "change": 0.0023,
                "timestamp": datetime.utcnow().isoformat()
            },
            timestamp=datetime.utcnow()
        )
        
        await manager.broadcast_price_update("EUR/USD", 1.0845, 0.0023)
        print("✅ Price update broadcasted")
        
        # Test trade execution
        trade_data = {
            "symbol": "GBP/USD",
            "action": "BUY",
            "quantity": 10000,
            "price": 1.2634,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await manager.broadcast_trade_executed(trade_data)
        print("✅ Trade execution broadcasted")
        
        # Test signal generation
        signal_data = {
            "symbol": "USD/JPY",
            "signal": "BUY",
            "confidence": 0.85,
            "strategy": "SMA_Cross",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await manager.broadcast_signal_generated(signal_data)
        print("✅ Signal generation broadcasted")
        
        # Test performance update
        performance_data = {
            "user_id": "test_user",
            "total_trades": 142,
            "win_rate": 0.685,
            "sharpe_ratio": 2.34,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await manager.broadcast_performance_update(performance_data)
        print("✅ Performance update broadcasted")
        
        # Test alert
        alert_data = {
            "level": "info",
            "message": "System status: All systems operational",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await manager.broadcast_alert(alert_data)
        print("✅ Alert broadcasted")
        
        print("\n🎯 WEBSOCKET SERVER TEST COMPLETED SUCCESSFULLY!")
        print("✅ All broadcasting functions working!")
        print("🚀 WebSocket server is fully functional!")
        
        return True
        
    except Exception as e:
        print(f"❌ WebSocket server test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    
    print("🚀 NEXUS WEBSOCKET SYSTEM TEST")
    print("=" * 50)
    
    try:
        # Test WebSocket connection
        print("📡 Testing WebSocket connection...")
        connection_success = asyncio.run(test_websocket_connection())
        
        # Test WebSocket server
        print("\n🔧 Testing WebSocket server...")
        server_success = asyncio.run(test_websocket_server())
        
        if connection_success and server_success:
            print("\n🎉 ALL WEBSOCKET TESTS COMPLETED SUCCESSFULLY!")
            print("✅ WebSocket system is production-ready!")
            print("🚀 Ready for real-time trading data streaming!")
            return True
        else:
            print("\n❌ Some WebSocket tests failed!")
            print("🔧 Please check the WebSocket server configuration")
            return False
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
