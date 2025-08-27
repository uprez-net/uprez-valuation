"""
WebSocket endpoints for real-time features
"""

from typing import Dict, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
import json
import asyncio

router = APIRouter()

# Connection manager for WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, room_id: str):
        await websocket.accept()
        if room_id not in self.active_connections:
            self.active_connections[room_id] = []
        self.active_connections[room_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, room_id: str):
        if room_id in self.active_connections:
            self.active_connections[room_id].remove(websocket)
            if not self.active_connections[room_id]:
                del self.active_connections[room_id]
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast_to_room(self, message: str, room_id: str):
        if room_id in self.active_connections:
            for connection in self.active_connections[room_id]:
                try:
                    await connection.send_text(message)
                except:
                    # Remove disconnected clients
                    self.active_connections[room_id].remove(connection)

manager = ConnectionManager()


@router.websocket("/collaboration/{project_id}")
async def collaboration_websocket(
    websocket: WebSocket,
    project_id: int
):
    """WebSocket for real-time collaboration"""
    
    room_id = f"project_{project_id}"
    await manager.connect(websocket, room_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Handle different message types
            message_type = message_data.get("type")
            
            if message_type == "comment":
                # Broadcast new comment to all project members
                await manager.broadcast_to_room(
                    json.dumps({
                        "type": "comment",
                        "data": message_data.get("data"),
                        "timestamp": message_data.get("timestamp")
                    }),
                    room_id
                )
            
            elif message_type == "valuation_update":
                # Broadcast valuation changes
                await manager.broadcast_to_room(
                    json.dumps({
                        "type": "valuation_update",
                        "data": message_data.get("data")
                    }),
                    room_id
                )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, room_id)


@router.websocket("/notifications/{user_id}")
async def notifications_websocket(
    websocket: WebSocket,
    user_id: int
):
    """WebSocket for real-time notifications"""
    
    room_id = f"user_{user_id}"
    await manager.connect(websocket, room_id)
    
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(30)
            await websocket.send_text(json.dumps({"type": "ping"}))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, room_id)