"""
WebSocket Connection Manager with Redis Pub/Sub Support
Handles real-time connections, message routing, and horizontal scaling
"""

import asyncio
import json
import logging
import time
from typing import Dict, Set, Optional, Any, List
from dataclasses import dataclass, asdict
from uuid import uuid4
import redis.asyncio as redis
from fastapi import WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@dataclass
class Connection:
    """WebSocket connection metadata"""
    websocket: WebSocket
    user_id: str
    session_id: str
    workspace_id: str
    permissions: Set[str]
    connected_at: float
    last_activity: float
    metadata: Dict[str, Any]

@dataclass
class Message:
    """Standardized message format"""
    type: str
    payload: Dict[str, Any]
    user_id: str
    session_id: str
    workspace_id: str
    timestamp: float
    message_id: str = None
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid4())

class ConnectionPool:
    """Manages WebSocket connections with load balancing"""
    
    def __init__(self, max_connections_per_workspace: int = 100):
        self.connections: Dict[str, Connection] = {}
        self.workspace_connections: Dict[str, Set[str]] = {}
        self.user_connections: Dict[str, Set[str]] = {}
        self.max_connections_per_workspace = max_connections_per_workspace
        self._connection_lock = asyncio.Lock()
    
    async def add_connection(self, connection: Connection) -> bool:
        """Add a new WebSocket connection"""
        async with self._connection_lock:
            # Check workspace connection limits
            workspace_count = len(self.workspace_connections.get(connection.workspace_id, set()))
            if workspace_count >= self.max_connections_per_workspace:
                logger.warning(f"Workspace {connection.workspace_id} has reached connection limit")
                return False
            
            connection_id = f"{connection.session_id}_{connection.user_id}"
            self.connections[connection_id] = connection
            
            # Track by workspace
            if connection.workspace_id not in self.workspace_connections:
                self.workspace_connections[connection.workspace_id] = set()
            self.workspace_connections[connection.workspace_id].add(connection_id)
            
            # Track by user
            if connection.user_id not in self.user_connections:
                self.user_connections[connection.user_id] = set()
            self.user_connections[connection.user_id].add(connection_id)
            
            logger.info(f"Connection added: {connection_id}")
            return True
    
    async def remove_connection(self, connection_id: str) -> bool:
        """Remove a WebSocket connection"""
        async with self._connection_lock:
            if connection_id not in self.connections:
                return False
            
            connection = self.connections[connection_id]
            
            # Remove from workspace tracking
            if connection.workspace_id in self.workspace_connections:
                self.workspace_connections[connection.workspace_id].discard(connection_id)
                if not self.workspace_connections[connection.workspace_id]:
                    del self.workspace_connections[connection.workspace_id]
            
            # Remove from user tracking
            if connection.user_id in self.user_connections:
                self.user_connections[connection.user_id].discard(connection_id)
                if not self.user_connections[connection.user_id]:
                    del self.user_connections[connection.user_id]
            
            del self.connections[connection_id]
            logger.info(f"Connection removed: {connection_id}")
            return True
    
    def get_workspace_connections(self, workspace_id: str) -> List[Connection]:
        """Get all connections for a workspace"""
        connection_ids = self.workspace_connections.get(workspace_id, set())
        return [self.connections[cid] for cid in connection_ids if cid in self.connections]
    
    def get_user_connections(self, user_id: str) -> List[Connection]:
        """Get all connections for a user"""
        connection_ids = self.user_connections.get(user_id, set())
        return [self.connections[cid] for cid in connection_ids if cid in self.connections]
    
    def update_activity(self, connection_id: str):
        """Update last activity timestamp"""
        if connection_id in self.connections:
            self.connections[connection_id].last_activity = time.time()

class WebSocketManager:
    """Main WebSocket manager with Redis pub/sub support"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.connection_pool = ConnectionPool()
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self._message_handlers: Dict[str, callable] = {}
        self._running = False
    
    async def start(self):
        """Initialize Redis connection and start message handling"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            self.pubsub = self.redis_client.pubsub()
            
            # Subscribe to all workspace channels
            await self.pubsub.psubscribe("workspace:*")
            
            self._running = True
            asyncio.create_task(self._handle_redis_messages())
            
            logger.info("WebSocket manager started with Redis pub/sub")
        except Exception as e:
            logger.error(f"Failed to start WebSocket manager: {e}")
            raise
    
    async def stop(self):
        """Clean shutdown"""
        self._running = False
        if self.pubsub:
            await self.pubsub.unsubscribe()
            await self.pubsub.close()
        if self.redis_client:
            await self.redis_client.close()
        logger.info("WebSocket manager stopped")
    
    @asynccontextmanager
    async def lifespan(self):
        """Context manager for proper startup/shutdown"""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()
    
    def register_message_handler(self, message_type: str, handler: callable):
        """Register a message type handler"""
        self._message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")
    
    async def connect(self, websocket: WebSocket, user_id: str, session_id: str, 
                     workspace_id: str, permissions: Set[str] = None,
                     metadata: Dict[str, Any] = None) -> str:
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        
        connection = Connection(
            websocket=websocket,
            user_id=user_id,
            session_id=session_id,
            workspace_id=workspace_id,
            permissions=permissions or set(),
            connected_at=time.time(),
            last_activity=time.time(),
            metadata=metadata or {}
        )
        
        success = await self.connection_pool.add_connection(connection)
        if not success:
            await websocket.close(code=1000, reason="Connection limit exceeded")
            return None
        
        connection_id = f"{session_id}_{user_id}"
        
        # Notify workspace about new connection
        await self._publish_presence_update(workspace_id, user_id, "joined", metadata)
        
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """Handle connection disconnect"""
        connection = self.connection_pool.connections.get(connection_id)
        if connection:
            # Notify workspace about disconnection
            await self._publish_presence_update(
                connection.workspace_id, 
                connection.user_id, 
                "left"
            )
        
        await self.connection_pool.remove_connection(connection_id)
    
    async def broadcast_to_workspace(self, workspace_id: str, message: Message, 
                                   exclude_user: str = None):
        """Broadcast message to all connections in a workspace"""
        # Publish to Redis for horizontal scaling
        await self._publish_to_redis(f"workspace:{workspace_id}", message)
        
        # Send to local connections
        await self._send_to_local_workspace(workspace_id, message, exclude_user)
    
    async def send_to_user(self, user_id: str, message: Message):
        """Send message to all connections of a specific user"""
        connections = self.connection_pool.get_user_connections(user_id)
        
        tasks = []
        for connection in connections:
            tasks.append(self._send_message(connection, message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def send_to_connection(self, connection_id: str, message: Message):
        """Send message to a specific connection"""
        connection = self.connection_pool.connections.get(connection_id)
        if connection:
            await self._send_message(connection, message)
    
    async def handle_message(self, connection_id: str, raw_message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(raw_message)
            connection = self.connection_pool.connections.get(connection_id)
            
            if not connection:
                logger.warning(f"Message from unknown connection: {connection_id}")
                return
            
            # Update activity
            self.connection_pool.update_activity(connection_id)
            
            # Create message object
            message = Message(
                type=data.get('type', 'unknown'),
                payload=data.get('payload', {}),
                user_id=connection.user_id,
                session_id=connection.session_id,
                workspace_id=connection.workspace_id,
                timestamp=time.time()
            )
            
            # Handle message based on type
            handler = self._message_handlers.get(message.type)
            if handler:
                await handler(connection, message)
            else:
                logger.warning(f"No handler for message type: {message.type}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from connection {connection_id}: {raw_message}")
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")
    
    async def _send_message(self, connection: Connection, message: Message):
        """Send message to a single connection"""
        try:
            message_data = {
                'type': message.type,
                'payload': message.payload,
                'user_id': message.user_id,
                'session_id': message.session_id,
                'workspace_id': message.workspace_id,
                'timestamp': message.timestamp,
                'message_id': message.message_id
            }
            
            await connection.websocket.send_text(json.dumps(message_data))
        except Exception as e:
            logger.error(f"Failed to send message to connection: {e}")
            # Connection might be stale, remove it
            connection_id = f"{connection.session_id}_{connection.user_id}"
            await self.disconnect(connection_id)
    
    async def _send_to_local_workspace(self, workspace_id: str, message: Message, 
                                     exclude_user: str = None):
        """Send message to all local connections in workspace"""
        connections = self.connection_pool.get_workspace_connections(workspace_id)
        
        tasks = []
        for connection in connections:
            if exclude_user and connection.user_id == exclude_user:
                continue
            tasks.append(self._send_message(connection, message))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Log any exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to send message to connection {i}: {result}")
    
    async def _publish_to_redis(self, channel: str, message: Message):
        """Publish message to Redis channel"""
        if not self.redis_client:
            return
        
        try:
            message_data = asdict(message)
            # Remove websocket object if present
            message_data.pop('websocket', None)
            
            await self.redis_client.publish(channel, json.dumps(message_data))
        except Exception as e:
            logger.error(f"Failed to publish to Redis: {e}")
    
    async def _publish_presence_update(self, workspace_id: str, user_id: str, 
                                     action: str, metadata: Dict[str, Any] = None):
        """Publish presence update to workspace"""
        message = Message(
            type="presence_update",
            payload={
                "action": action,
                "metadata": metadata or {}
            },
            user_id=user_id,
            session_id="system",
            workspace_id=workspace_id,
            timestamp=time.time()
        )
        
        await self.broadcast_to_workspace(workspace_id, message, exclude_user=user_id)
    
    async def _handle_redis_messages(self):
        """Handle incoming Redis pub/sub messages"""
        if not self.pubsub:
            return
        
        while self._running:
            try:
                message = await self.pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'pmessage':
                    channel = message['channel']
                    data = json.loads(message['data'])
                    
                    # Reconstruct message object
                    msg = Message(**data)
                    
                    # Extract workspace_id from channel
                    workspace_id = channel.split(':', 1)[1]
                    
                    # Send to local connections
                    await self._send_to_local_workspace(workspace_id, msg)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error handling Redis message: {e}")
    
    def get_workspace_stats(self, workspace_id: str) -> Dict[str, Any]:
        """Get connection statistics for a workspace"""
        connections = self.connection_pool.get_workspace_connections(workspace_id)
        
        return {
            "total_connections": len(connections),
            "unique_users": len(set(c.user_id for c in connections)),
            "connections_by_user": {
                user_id: len([c for c in connections if c.user_id == user_id])
                for user_id in set(c.user_id for c in connections)
            },
            "average_session_duration": sum(
                time.time() - c.connected_at for c in connections
            ) / len(connections) if connections else 0
        }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global connection statistics"""
        all_connections = list(self.connection_pool.connections.values())
        
        return {
            "total_connections": len(all_connections),
            "total_workspaces": len(self.connection_pool.workspace_connections),
            "total_users": len(self.connection_pool.user_connections),
            "connections_by_workspace": {
                workspace_id: len(connections)
                for workspace_id, connections in self.connection_pool.workspace_connections.items()
            },
            "uptime_stats": {
                "average_connection_age": sum(
                    time.time() - c.connected_at for c in all_connections
                ) / len(all_connections) if all_connections else 0,
                "active_in_last_minute": len([
                    c for c in all_connections 
                    if time.time() - c.last_activity < 60
                ])
            }
        }

# Global WebSocket manager instance
websocket_manager = WebSocketManager()