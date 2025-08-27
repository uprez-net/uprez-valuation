"""
Tests for WebSocket Manager and Connection Handling
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.websocket_manager import (
    WebSocketManager, ConnectionPool, Connection, Message
)

class TestConnectionPool:
    """Test connection pool functionality"""
    
    def setup_method(self):
        self.pool = ConnectionPool(max_connections_per_workspace=5)
    
    @pytest.mark.asyncio
    async def test_add_connection_success(self):
        """Test successful connection addition"""
        websocket = Mock(spec=WebSocket)
        connection = Connection(
            websocket=websocket,
            user_id="user1",
            session_id="session1",
            workspace_id="workspace1",
            permissions={"read", "write"},
            connected_at=1234567890,
            last_activity=1234567890,
            metadata={}
        )
        
        success = await self.pool.add_connection(connection)
        assert success is True
        assert len(self.pool.connections) == 1
        assert "workspace1" in self.pool.workspace_connections
        assert "user1" in self.pool.user_connections
    
    @pytest.mark.asyncio
    async def test_connection_limit_exceeded(self):
        """Test connection limit enforcement"""
        websocket = Mock(spec=WebSocket)
        
        # Add maximum allowed connections
        for i in range(5):
            connection = Connection(
                websocket=websocket,
                user_id=f"user{i}",
                session_id=f"session{i}",
                workspace_id="workspace1",
                permissions=set(),
                connected_at=1234567890,
                last_activity=1234567890,
                metadata={}
            )
            success = await self.pool.add_connection(connection)
            assert success is True
        
        # Try to add one more (should fail)
        extra_connection = Connection(
            websocket=websocket,
            user_id="user_extra",
            session_id="session_extra",
            workspace_id="workspace1",
            permissions=set(),
            connected_at=1234567890,
            last_activity=1234567890,
            metadata={}
        )
        success = await self.pool.add_connection(extra_connection)
        assert success is False
    
    @pytest.mark.asyncio
    async def test_remove_connection(self):
        """Test connection removal"""
        websocket = Mock(spec=WebSocket)
        connection = Connection(
            websocket=websocket,
            user_id="user1",
            session_id="session1",
            workspace_id="workspace1",
            permissions=set(),
            connected_at=1234567890,
            last_activity=1234567890,
            metadata={}
        )
        
        await self.pool.add_connection(connection)
        connection_id = "session1_user1"
        
        success = await self.pool.remove_connection(connection_id)
        assert success is True
        assert len(self.pool.connections) == 0
        assert "workspace1" not in self.pool.workspace_connections
        assert "user1" not in self.pool.user_connections
    
    def test_get_workspace_connections(self):
        """Test getting workspace connections"""
        websocket1 = Mock(spec=WebSocket)
        websocket2 = Mock(spec=WebSocket)
        
        connection1 = Connection(
            websocket=websocket1,
            user_id="user1",
            session_id="session1",
            workspace_id="workspace1",
            permissions=set(),
            connected_at=1234567890,
            last_activity=1234567890,
            metadata={}
        )
        
        connection2 = Connection(
            websocket=websocket2,
            user_id="user2",
            session_id="session2",
            workspace_id="workspace1",
            permissions=set(),
            connected_at=1234567890,
            last_activity=1234567890,
            metadata={}
        )
        
        # Add connections using asyncio.run for sync test
        asyncio.run(self.pool.add_connection(connection1))
        asyncio.run(self.pool.add_connection(connection2))
        
        workspace_connections = self.pool.get_workspace_connections("workspace1")
        assert len(workspace_connections) == 2
        assert connection1 in workspace_connections
        assert connection2 in workspace_connections

class TestWebSocketManager:
    """Test WebSocket manager functionality"""
    
    def setup_method(self):
        self.manager = WebSocketManager("redis://localhost:6379/0")
    
    @pytest.mark.asyncio
    async def test_start_stop_manager(self):
        """Test manager startup and shutdown"""
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_pubsub = AsyncMock()
            mock_redis.return_value = mock_client
            mock_client.pubsub.return_value = mock_pubsub
            
            await self.manager.start()
            assert self.manager._running is True
            assert self.manager.redis_client is not None
            
            await self.manager.stop()
            assert self.manager._running is False
    
    @pytest.mark.asyncio
    async def test_connect_websocket(self):
        """Test WebSocket connection"""
        websocket = AsyncMock(spec=WebSocket)
        websocket.accept = AsyncMock()
        
        with patch.object(self.manager, '_publish_presence_update', AsyncMock()):
            connection_id = await self.manager.connect(
                websocket=websocket,
                user_id="user1",
                session_id="session1",
                workspace_id="workspace1",
                permissions={"read"},
                metadata={"name": "Test User"}
            )
            
            assert connection_id == "session1_user1"
            websocket.accept.assert_called_once()
            assert len(self.manager.connection_pool.connections) == 1
    
    @pytest.mark.asyncio
    async def test_disconnect_websocket(self):
        """Test WebSocket disconnection"""
        websocket = AsyncMock(spec=WebSocket)
        websocket.accept = AsyncMock()
        
        with patch.object(self.manager, '_publish_presence_update', AsyncMock()):
            connection_id = await self.manager.connect(
                websocket=websocket,
                user_id="user1",
                session_id="session1",
                workspace_id="workspace1"
            )
            
            await self.manager.disconnect(connection_id)
            assert len(self.manager.connection_pool.connections) == 0
    
    @pytest.mark.asyncio
    async def test_broadcast_to_workspace(self):
        """Test workspace broadcasting"""
        websocket1 = AsyncMock(spec=WebSocket)
        websocket2 = AsyncMock(spec=WebSocket)
        websocket1.send_text = AsyncMock()
        websocket2.send_text = AsyncMock()
        
        message = Message(
            type="test_message",
            payload={"content": "test"},
            user_id="user1",
            session_id="session1",
            workspace_id="workspace1",
            timestamp=1234567890
        )
        
        with patch.object(self.manager, '_publish_to_redis', AsyncMock()):
            # Add connections
            await self.manager.connection_pool.add_connection(Connection(
                websocket=websocket1,
                user_id="user1",
                session_id="session1",
                workspace_id="workspace1",
                permissions=set(),
                connected_at=1234567890,
                last_activity=1234567890,
                metadata={}
            ))
            
            await self.manager.connection_pool.add_connection(Connection(
                websocket=websocket2,
                user_id="user2",
                session_id="session2",
                workspace_id="workspace1",
                permissions=set(),
                connected_at=1234567890,
                last_activity=1234567890,
                metadata={}
            ))
            
            # Broadcast message
            await self.manager.broadcast_to_workspace("workspace1", message)
            
            # Both websockets should receive the message
            websocket1.send_text.assert_called_once()
            websocket2.send_text.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_message(self):
        """Test message handling"""
        websocket = AsyncMock(spec=WebSocket)
        connection_id = "session1_user1"
        
        # Add connection
        await self.manager.connection_pool.add_connection(Connection(
            websocket=websocket,
            user_id="user1",
            session_id="session1",
            workspace_id="workspace1",
            permissions=set(),
            connected_at=1234567890,
            last_activity=1234567890,
            metadata={}
        ))
        
        # Register a message handler
        handler = AsyncMock()
        self.manager.register_message_handler("test_type", handler)
        
        # Handle message
        raw_message = json.dumps({
            "type": "test_type",
            "payload": {"data": "test"}
        })
        
        await self.manager.handle_message(connection_id, raw_message)
        
        # Handler should be called
        handler.assert_called_once()
        call_args = handler.call_args
        connection, message = call_args[0]
        
        assert message.type == "test_type"
        assert message.payload["data"] == "test"
        assert message.user_id == "user1"
        assert message.workspace_id == "workspace1"
    
    def test_register_message_handler(self):
        """Test message handler registration"""
        handler = Mock()
        self.manager.register_message_handler("test_type", handler)
        
        assert "test_type" in self.manager._message_handlers
        assert self.manager._message_handlers["test_type"] == handler
    
    def test_get_workspace_stats(self):
        """Test workspace statistics"""
        # Add some mock connections
        asyncio.run(self._add_mock_connections())
        
        stats = self.manager.get_workspace_stats("workspace1")
        
        assert "total_connections" in stats
        assert "unique_users" in stats
        assert "connections_by_user" in stats
        assert "average_session_duration" in stats
    
    async def _add_mock_connections(self):
        """Helper to add mock connections"""
        websocket1 = Mock(spec=WebSocket)
        websocket2 = Mock(spec=WebSocket)
        
        await self.manager.connection_pool.add_connection(Connection(
            websocket=websocket1,
            user_id="user1",
            session_id="session1",
            workspace_id="workspace1",
            permissions=set(),
            connected_at=1234567890,
            last_activity=1234567890,
            metadata={}
        ))
        
        await self.manager.connection_pool.add_connection(Connection(
            websocket=websocket2,
            user_id="user2",
            session_id="session2",
            workspace_id="workspace1",
            permissions=set(),
            connected_at=1234567890,
            last_activity=1234567890,
            metadata={}
        ))

class TestMessage:
    """Test message class"""
    
    def test_message_creation(self):
        """Test message object creation"""
        message = Message(
            type="test",
            payload={"key": "value"},
            user_id="user1",
            session_id="session1",
            workspace_id="workspace1",
            timestamp=1234567890
        )
        
        assert message.type == "test"
        assert message.payload["key"] == "value"
        assert message.user_id == "user1"
        assert message.message_id is not None
    
    def test_message_id_generation(self):
        """Test automatic message ID generation"""
        message1 = Message(
            type="test",
            payload={},
            user_id="user1",
            session_id="session1",
            workspace_id="workspace1",
            timestamp=1234567890
        )
        
        message2 = Message(
            type="test",
            payload={},
            user_id="user1",
            session_id="session1",
            workspace_id="workspace1",
            timestamp=1234567890
        )
        
        assert message1.message_id != message2.message_id

if __name__ == "__main__":
    pytest.main([__file__, "-v"])