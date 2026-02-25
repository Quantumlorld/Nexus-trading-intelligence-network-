"""
Nexus Trading System - WebSocket Server
Real-time data streaming and live updates for enterprise features
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
import websockets
from websockets.server import WebSocketServerProtocol
import aiofiles
from pathlib import Path
import uuid
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """WebSocket message types"""
    PRICE_UPDATE = "price_update"
    TRADE_EXECUTED = "trade_executed"
    SIGNAL_GENERATED = "signal_generated"
    USER_CONNECTED = "user_connected"
    USER_DISCONNECTED = "user_disconnected"
    SYSTEM_STATUS = "system_status"
    PERFORMANCE_UPDATE = "performance_update"
    ALERT = "alert"
    CHAT_MESSAGE = "chat_message"

@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    type: MessageType
    data: Dict[str, Any]
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class NexusWebSocketServer(WebSocketServerProtocol):
    """WebSocket server for real-time trading data"""
    
    def __init__(self):
        super().__init__()
        self.server = None
        self.user_id = None
        self.session_id = str(uuid.uuid4())
        self.connected_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
    async def on_connect(self, request):
        """Handle new WebSocket connection"""
        self.user_id = request.query_params.get('user_id', 'anonymous')
        
        # Register connection
        if self.server:
            await self.server.register_connection(self)
        
        # Send welcome message
        welcome_message = WebSocketMessage(
            type=MessageType.USER_CONNECTED,
            data={
                "message": f"Welcome to Nexus Trading WebSocket",
                "user_id": self.user_id,
                "session_id": self.session_id,
                "connected_at": self.connected_at.isoformat(),
                "server_time": datetime.utcnow().isoformat()
            },
            timestamp=datetime.utcnow(),
            user_id=self.user_id,
            session_id=self.session_id
        )
        
        await self.send_message(welcome_message)
        logger.info(f"WebSocket connection established for user {self.user_id}")
    
    async def on_message(self, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            message_type = data.get('type', 'unknown')
            
            # Update last activity
            self.last_activity = datetime.utcnow()
            
            if message_type == 'subscribe':
                await self.handle_subscribe(data)
            elif message_type == 'unsubscribe':
                await self.handle_unsubscribe(data)
            elif message_type == 'ping':
                await self.handle_ping()
            elif message_type == 'chat':
                await self.handle_chat_message(data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self.send_error_message(f"Error processing message: {str(e)}")
    
    async def on_close(self, code, reason):
        """Handle WebSocket disconnection"""
        if self.server:
            await self.server.unregister_connection(self)
        
        # Send disconnect notification
        disconnect_message = WebSocketMessage(
            type=MessageType.USER_DISCONNECTED,
            data={
                "user_id": self.user_id,
                "session_id": self.session_id,
                "disconnected_at": datetime.utcnow().isoformat(),
                "reason": reason
            },
            timestamp=datetime.utcnow(),
            user_id=self.user_id,
            session_id=self.session_id
        )
        
        await self.broadcast_to_all(disconnect_message)
        logger.info(f"WebSocket connection closed for user {self.user_id}")
    
    async def send_message(self, message: WebSocketMessage):
        """Send message to this WebSocket client"""
        try:
            await self.send(json.dumps({
                'type': message.type.value,
                'data': message.data,
                'timestamp': message.timestamp.isoformat(),
                'user_id': message.user_id,
                'session_id': message.session_id
            }))
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
    
    async def send_error_message(self, error_message: str):
        """Send error message to client"""
        error_msg = WebSocketMessage(
            type=MessageType.ALERT,
            data={
                "level": "error",
                "message": error_message,
                "timestamp": datetime.utcnow().isoformat()
            },
            timestamp=datetime.utcnow(),
            user_id=self.user_id,
            session_id=self.session_id
        )
        await self.send_message(error_msg)
    
    async def handle_subscribe(self, data):
        """Handle subscription requests"""
        subscription_type = data.get('subscription')
        
        if self.server:
            await self.server.add_subscription(self, subscription_type)
        
        # Send confirmation
        confirm_message = WebSocketMessage(
            type=MessageType.SYSTEM_STATUS,
            data={
                "message": f"Subscribed to {subscription_type}",
                "subscription": subscription_type,
                "timestamp": datetime.utcnow().isoformat()
            },
            timestamp=datetime.utcnow(),
            user_id=self.user_id,
            session_id=self.session_id
        )
        await self.send_message(confirm_message)
    
    async def handle_unsubscribe(self, data):
        """Handle unsubscription requests"""
        subscription_type = data.get('subscription')
        
        if self.server:
            await self.server.remove_subscription(self, subscription_type)
        
        # Send confirmation
        confirm_message = WebSocketMessage(
            type=MessageType.SYSTEM_STATUS,
            data={
                "message": f"Unsubscribed from {subscription_type}",
                "subscription": subscription_type,
                "timestamp": datetime.utcnow().isoformat()
            },
            timestamp=datetime.utcnow(),
            user_id=self.user_id,
            session_id=self.session_id
        )
        await self.send_message(confirm_message)
    
    async def handle_ping(self):
        """Handle ping requests"""
        pong_message = WebSocketMessage(
            type=MessageType.SYSTEM_STATUS,
            data={
                "message": "pong",
                "timestamp": datetime.utcnow().isoformat()
            },
            timestamp=datetime.utcnow(),
            user_id=self.user_id,
            session_id=self.session_id
        )
        await self.send_message(pong_message)
    
    async def handle_chat_message(self, data):
        """Handle chat messages"""
        chat_message = WebSocketMessage(
            type=MessageType.CHAT_MESSAGE,
            data={
                "user_id": self.user_id,
                "message": data.get('message', ''),
                "timestamp": datetime.utcnow().isoformat()
            },
            timestamp=datetime.utcnow(),
            user_id=self.user_id,
            session_id=self.session_id
        )
        
        # Broadcast to all connected users
        if self.server:
            await self.server.broadcast_to_all(chat_message)

class WebSocketManager:
    """Manages WebSocket connections and broadcasting"""
    
    def __init__(self):
        self.connections: Set[NexusWebSocketServer] = set()
        self.user_connections: Dict[str, NexusWebSocketServer] = {}
        self.subscriptions: Dict[str, Set[NexusWebSocketServer]] = {}
        self.server = None
        self.is_running = False
        
        # Statistics
        self.total_connections = 0
        self.total_messages_sent = 0
        self.total_messages_received = 0
        
        logger.info("WebSocket Manager initialized")
    
    async def register_connection(self, websocket: NexusWebSocketServer):
        """Register a new WebSocket connection"""
        self.connections.add(websocket)
        self.user_connections[websocket.user_id] = websocket
        websocket.server = self
        self.total_connections += 1
        
        logger.info(f"Registered connection for user {websocket.user_id}")
    
    async def unregister_connection(self, websocket: NexusWebSocketServer):
        """Unregister a WebSocket connection"""
        self.connections.discard(websocket)
        self.user_connections.pop(websocket.user_id, None)
        
        # Remove from all subscriptions
        for subscription_type, subscribers in self.subscriptions.items():
            subscribers.discard(websocket)
        
        logger.info(f"Unregistered connection for user {websocket.user_id}")
    
    async def add_subscription(self, websocket: NexusWebSocketServer, subscription_type: str):
        """Add subscription for a connection"""
        if subscription_type not in self.subscriptions:
            self.subscriptions[subscription_type] = set()
        
        self.subscriptions[subscription_type].add(websocket)
        logger.info(f"Added subscription {subscription_type} for user {websocket.user_id}")
    
    async def remove_subscription(self, websocket: NexusWebSocketServer, subscription_type: str):
        """Remove subscription for a connection"""
        if subscription_type in self.subscriptions:
            self.subscriptions[subscription_type].discard(websocket)
        
        logger.info(f"Removed subscription {subscription_type} for user {websocket.user_id}")
    
    async def broadcast_to_all(self, message: WebSocketMessage):
        """Broadcast message to all connected clients"""
        if not self.connections:
            return
        
        disconnected = set()
        
        for websocket in self.connections:
            try:
                await websocket.send_message(message)
                self.total_messages_sent += 1
            except Exception as e:
                logger.error(f"Error broadcasting to user {websocket.user_id}: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected connections
        for ws in disconnected:
            await self.unregister_connection(ws)
    
    async def broadcast_to_subscription(self, subscription_type: str, message: WebSocketMessage):
        """Broadcast message to specific subscription"""
        if subscription_type not in self.subscriptions:
            return
        
        disconnected = set()
        
        for websocket in self.subscriptions[subscription_type]:
            try:
                await websocket.send_message(message)
                self.total_messages_sent += 1
            except Exception as e:
                logger.error(f"Error broadcasting to user {websocket.user_id}: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected connections
        for ws in disconnected:
            await self.unregister_connection(ws)
    
    async def broadcast_price_update(self, symbol: str, price: float, change: float):
        """Broadcast price update to all clients"""
        price_message = WebSocketMessage(
            type=MessageType.PRICE_UPDATE,
            data={
                "symbol": symbol,
                "price": price,
                "change": change,
                "timestamp": datetime.utcnow().isoformat()
            },
            timestamp=datetime.utcnow()
        )
        
        await self.broadcast_to_subscription("prices", price_message)
    
    async def broadcast_trade_executed(self, trade_data: Dict[str, Any]):
        """Broadcast trade execution to all clients"""
        trade_message = WebSocketMessage(
            type=MessageType.TRADE_EXECUTED,
            data=trade_data,
            timestamp=datetime.utcnow()
        )
        
        await self.broadcast_to_subscription("trades", trade_message)
    
    async def broadcast_signal_generated(self, signal_data: Dict[str, Any]):
        """Broadcast trading signal to all clients"""
        signal_message = WebSocketMessage(
            type=MessageType.SIGNAL_GENERATED,
            data=signal_data,
            timestamp=datetime.utcnow()
        )
        
        await self.broadcast_to_subscription("signals", signal_message)
    
    async def broadcast_performance_update(self, performance_data: Dict[str, Any]):
        """Broadcast performance update to all clients"""
        performance_message = WebSocketMessage(
            type=MessageType.PERFORMANCE_UPDATE,
            data=performance_data,
            timestamp=datetime.utcnow()
        )
        
        await self.broadcast_to_subscription("performance", performance_message)
    
    async def broadcast_alert(self, alert_data: Dict[str, Any]):
        """Broadcast alert to all clients"""
        alert_message = WebSocketMessage(
            type=MessageType.ALERT,
            data=alert_data,
            timestamp=datetime.utcnow()
        )
        
        await self.broadcast_to_all(alert_message)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "total_connections": len(self.connections),
            "active_users": len(self.user_connections),
            "subscriptions": {k: len(v) for k, v in self.subscriptions.items()},
            "total_messages_sent": self.total_messages_sent,
            "total_messages_received": self.total_messages_received,
            "is_running": self.is_running,
            "server_time": datetime.utcnow().isoformat()
        }
    
    async def start_server(self, host: str = "localhost", port: int = 8765):
        """Start the WebSocket server"""
        self.is_running = True
        
        logger.info(f"Starting WebSocket server on {host}:{port}")
        
        # Create server with custom protocol
        server = await websockets.serve(
            lambda ws, path: NexusWebSocketServer(ws, path),
            host,
            port
        )
        
        self.server = server
        logger.info(f"WebSocket server started on {host}:{port}")
        
        return server
    
    async def stop_server(self):
        """Stop the WebSocket server"""
        self.is_running = False
        
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
        
        logger.info("WebSocket server stopped")

# Global WebSocket manager instance
websocket_manager = WebSocketManager()

# Factory function
def create_websocket_manager() -> WebSocketManager:
    """Create and return WebSocket manager instance"""
    return websocket_manager

# Main server function
async def run_websocket_server(host: str = "localhost", port: int = 8765):
    """Run the WebSocket server"""
    manager = create_websocket_manager()
    await manager.start_server(host, port)
    
    try:
        # Keep server running
        while manager.is_running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("WebSocket server shutdown requested")
    finally:
        await manager.stop_server()

if __name__ == "__main__":
    # Run the WebSocket server
    asyncio.run(run_websocket_server())
