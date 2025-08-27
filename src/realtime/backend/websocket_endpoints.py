"""
FastAPI WebSocket Endpoints
Handles WebSocket connections, message routing, and real-time communication
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.security import HTTPBearer
import jwt

from .websocket_manager import websocket_manager, Message
from .operational_transform import ot_engine, doc_sync
from .live_data_sync import data_stream_manager, valuation_stream_handler, financial_metrics_streamer
from .collaboration_features import collaboration_manager

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

app = FastAPI(title="Real-time Collaboration API")

# Authentication helper
async def verify_token(token: str) -> Dict[str, Any]:
    """Verify JWT token and return user info"""
    try:
        # In production, use proper JWT verification
        payload = jwt.decode(token, "your-secret-key", algorithms=["HS256"])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.on_event("startup")
async def startup_event():
    """Initialize WebSocket manager and data streams"""
    await websocket_manager.start()
    data_stream_manager.start()
    
    # Register message handlers
    websocket_manager.register_message_handler("document_edit", handle_document_edit)
    websocket_manager.register_message_handler("cursor_update", handle_cursor_update)
    websocket_manager.register_message_handler("selection_update", handle_selection_update)
    websocket_manager.register_message_handler("comment_add", handle_comment_add)
    websocket_manager.register_message_handler("comment_reply", handle_comment_reply)
    websocket_manager.register_message_handler("annotation_create", handle_annotation_create)
    websocket_manager.register_message_handler("subscribe_stream", handle_stream_subscription)
    websocket_manager.register_message_handler("presence_update", handle_presence_update)
    
    logger.info("Real-time collaboration system started")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    await websocket_manager.stop()
    data_stream_manager.stop()
    logger.info("Real-time collaboration system stopped")

@app.websocket("/ws/{workspace_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    workspace_id: str,
    token: str = Query(...),
    session_id: str = Query(...),
    user_id: str = Query(None)
):
    """Main WebSocket endpoint for real-time collaboration"""
    try:
        # Verify authentication
        user_info = await verify_token(token)
        if not user_id:
            user_id = user_info.get("user_id")
        
        if not user_id:
            await websocket.close(code=1008, reason="Invalid user authentication")
            return
        
        # Get user permissions
        permissions = set(user_info.get("permissions", []))
        metadata = {
            "username": user_info.get("username", ""),
            "display_name": user_info.get("display_name", ""),
            "role": user_info.get("role", "viewer")
        }
        
        # Connect to WebSocket manager
        connection_id = await websocket_manager.connect(
            websocket=websocket,
            user_id=user_id,
            session_id=session_id,
            workspace_id=workspace_id,
            permissions=permissions,
            metadata=metadata
        )
        
        if not connection_id:
            await websocket.close(code=1000, reason="Connection failed")
            return
        
        logger.info(f"WebSocket connected: {connection_id}")
        
        # Update user presence
        collaboration_manager.update_user_presence(
            user_id=user_id,
            workspace_id=workspace_id,
            status="online",
            activity="connected",
            session_id=session_id
        )
        
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                await websocket_manager.handle_message(connection_id, data)
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            # Cleanup
            await websocket_manager.disconnect(connection_id)
            collaboration_manager.remove_user_presence(user_id)
            data_stream_manager.unsubscribe_user(user_id)
            
    except Exception as e:
        logger.error(f"WebSocket endpoint error: {e}")
        await websocket.close(code=1011, reason="Internal server error")

# Message Handlers

async def handle_document_edit(connection, message: Message):
    """Handle document edit operations with OT"""
    try:
        payload = message.payload
        document_id = payload.get("document_id")
        operations = payload.get("operations", [])
        base_version = payload.get("base_version", 0)
        
        if not document_id or not operations:
            return
        
        # Create document if it doesn't exist
        if not ot_engine.get_document(document_id):
            ot_engine.create_document(document_id, "", {"workspace_id": message.workspace_id})
        
        # Convert operations to OT operations
        from .operational_transform import Operation, OperationType, Delta
        ot_operations = []
        
        for op in operations:
            ot_op = Operation(
                type=OperationType(op["type"]),
                position=op["position"],
                content=op.get("content", ""),
                length=op.get("length", 0),
                user_id=message.user_id,
                timestamp=message.timestamp
            )
            ot_operations.append(ot_op)
        
        # Create delta
        delta = Delta(
            operations=ot_operations,
            base_version=base_version,
            result_version=base_version + 1,
            document_id=document_id,
            user_id=message.user_id,
            session_id=message.session_id,
            timestamp=message.timestamp
        )
        
        # Apply delta
        success, error, updated_doc = ot_engine.apply_delta(document_id, delta)
        
        if success:
            # Broadcast update to other users
            update_message = Message(
                type="document_updated",
                payload={
                    "document_id": document_id,
                    "delta": {
                        "operations": [
                            {
                                "type": op.type.value,
                                "position": op.position,
                                "content": op.content,
                                "length": op.length
                            } for op in delta.operations
                        ],
                        "base_version": delta.base_version,
                        "result_version": delta.result_version,
                        "user_id": delta.user_id
                    },
                    "content": updated_doc.content,
                    "version": updated_doc.version
                },
                user_id="system",
                session_id="system",
                workspace_id=message.workspace_id,
                timestamp=message.timestamp
            )
            
            await websocket_manager.broadcast_to_workspace(
                message.workspace_id, update_message, exclude_user=message.user_id
            )
        else:
            # Send error back to user
            error_message = Message(
                type="edit_error",
                payload={"error": error, "document_id": document_id},
                user_id="system",
                session_id="system",
                workspace_id=message.workspace_id,
                timestamp=message.timestamp
            )
            
            await websocket_manager.send_to_user(message.user_id, error_message)
            
    except Exception as e:
        logger.error(f"Error handling document edit: {e}")

async def handle_cursor_update(connection, message: Message):
    """Handle cursor position updates"""
    try:
        payload = message.payload
        document_id = payload.get("document_id")
        position = payload.get("position")
        
        if document_id and position is not None:
            # Update cursor in doc sync
            doc_sync.update_cursor(document_id, message.user_id, position, payload.get("metadata"))
            
            # Update presence
            collaboration_manager.update_user_presence(
                user_id=message.user_id,
                workspace_id=message.workspace_id,
                document_id=document_id,
                cursor_position={"position": position, "timestamp": message.timestamp},
                activity="editing"
            )
            
            # Broadcast cursor update
            cursor_message = Message(
                type="cursor_updated",
                payload={
                    "document_id": document_id,
                    "user_id": message.user_id,
                    "position": position,
                    "metadata": payload.get("metadata", {})
                },
                user_id="system",
                session_id="system",
                workspace_id=message.workspace_id,
                timestamp=message.timestamp
            )
            
            await websocket_manager.broadcast_to_workspace(
                message.workspace_id, cursor_message, exclude_user=message.user_id
            )
            
    except Exception as e:
        logger.error(f"Error handling cursor update: {e}")

async def handle_selection_update(connection, message: Message):
    """Handle text selection updates"""
    try:
        payload = message.payload
        document_id = payload.get("document_id")
        start = payload.get("start")
        end = payload.get("end")
        
        if document_id and start is not None and end is not None:
            # Update selection in doc sync
            doc_sync.update_selection(document_id, message.user_id, start, end, payload.get("metadata"))
            
            # Update presence
            collaboration_manager.update_user_presence(
                user_id=message.user_id,
                workspace_id=message.workspace_id,
                document_id=document_id,
                selection_range={"start": start, "end": end, "timestamp": message.timestamp},
                activity="selecting"
            )
            
            # Broadcast selection update
            selection_message = Message(
                type="selection_updated",
                payload={
                    "document_id": document_id,
                    "user_id": message.user_id,
                    "start": start,
                    "end": end,
                    "metadata": payload.get("metadata", {})
                },
                user_id="system",
                session_id="system",
                workspace_id=message.workspace_id,
                timestamp=message.timestamp
            )
            
            await websocket_manager.broadcast_to_workspace(
                message.workspace_id, selection_message, exclude_user=message.user_id
            )
            
    except Exception as e:
        logger.error(f"Error handling selection update: {e}")

async def handle_comment_add(connection, message: Message):
    """Handle adding comments"""
    try:
        payload = message.payload
        document_id = payload.get("document_id")
        content = payload.get("content")
        position = payload.get("position")
        
        if not document_id or not content:
            return
        
        # Add comment
        comment = collaboration_manager.add_comment(
            document_id=document_id,
            workspace_id=message.workspace_id,
            user_id=message.user_id,
            content=content,
            position=position
        )
        
        # Broadcast comment to workspace
        comment_message = Message(
            type="comment_added",
            payload={
                "comment": {
                    "comment_id": comment.comment_id,
                    "document_id": comment.document_id,
                    "user_id": comment.user_id,
                    "content": comment.content,
                    "position": comment.position,
                    "created_at": comment.created_at
                }
            },
            user_id="system",
            session_id="system",
            workspace_id=message.workspace_id,
            timestamp=message.timestamp
        )
        
        await websocket_manager.broadcast_to_workspace(
            message.workspace_id, comment_message
        )
        
    except Exception as e:
        logger.error(f"Error handling comment add: {e}")

async def handle_comment_reply(connection, message: Message):
    """Handle comment replies"""
    try:
        payload = message.payload
        comment_id = payload.get("comment_id")
        content = payload.get("content")
        
        if not comment_id or not content:
            return
        
        # Add reply
        reply = collaboration_manager.reply_to_comment(
            comment_id=comment_id,
            user_id=message.user_id,
            content=content
        )
        
        if reply:
            # Broadcast reply to workspace
            reply_message = Message(
                type="comment_reply_added",
                payload={
                    "reply": {
                        "reply_id": reply.reply_id,
                        "comment_id": reply.comment_id,
                        "user_id": reply.user_id,
                        "content": reply.content,
                        "created_at": reply.created_at
                    }
                },
                user_id="system",
                session_id="system",
                workspace_id=message.workspace_id,
                timestamp=message.timestamp
            )
            
            await websocket_manager.broadcast_to_workspace(
                message.workspace_id, reply_message
            )
            
    except Exception as e:
        logger.error(f"Error handling comment reply: {e}")

async def handle_annotation_create(connection, message: Message):
    """Handle annotation creation"""
    try:
        payload = message.payload
        document_id = payload.get("document_id")
        annotation_type = payload.get("type")
        content = payload.get("content")
        position = payload.get("position")
        style = payload.get("style", {})
        
        if not all([document_id, annotation_type, content, position]):
            return
        
        # Create annotation
        annotation = collaboration_manager.add_annotation(
            document_id=document_id,
            workspace_id=message.workspace_id,
            user_id=message.user_id,
            annotation_type=annotation_type,
            content=content,
            position=position,
            style=style
        )
        
        # Broadcast annotation to workspace
        annotation_message = Message(
            type="annotation_created",
            payload={
                "annotation": {
                    "annotation_id": annotation.annotation_id,
                    "document_id": annotation.document_id,
                    "user_id": annotation.user_id,
                    "type": annotation.type,
                    "content": annotation.content,
                    "position": annotation.position,
                    "style": annotation.style,
                    "created_at": annotation.created_at
                }
            },
            user_id="system",
            session_id="system",
            workspace_id=message.workspace_id,
            timestamp=message.timestamp
        )
        
        await websocket_manager.broadcast_to_workspace(
            message.workspace_id, annotation_message
        )
        
    except Exception as e:
        logger.error(f"Error handling annotation create: {e}")

async def handle_stream_subscription(connection, message: Message):
    """Handle data stream subscriptions"""
    try:
        payload = message.payload
        stream_types = payload.get("stream_types", [])
        filters = payload.get("filters", {})
        
        # Convert string stream types to enums
        from .live_data_sync import StreamType
        enum_stream_types = []
        for st in stream_types:
            try:
                enum_stream_types.append(StreamType(st))
            except ValueError:
                logger.warning(f"Unknown stream type: {st}")
        
        if not enum_stream_types:
            return
        
        # Create subscription with WebSocket callback
        async def stream_callback(stream_data):
            stream_message = Message(
                type="stream_data",
                payload={
                    "stream_type": stream_data.stream_type.value,
                    "data": stream_data.data,
                    "timestamp": stream_data.timestamp,
                    "source": stream_data.source,
                    "sequence_id": stream_data.sequence_id
                },
                user_id="system",
                session_id="system",
                workspace_id=stream_data.workspace_id,
                timestamp=stream_data.timestamp
            )
            
            await websocket_manager.send_to_user(message.user_id, stream_message)
        
        subscription_id = data_stream_manager.subscribe(
            user_id=message.user_id,
            workspace_id=message.workspace_id,
            stream_types=enum_stream_types,
            filters=filters,
            callback=stream_callback
        )
        
        # Send confirmation
        confirm_message = Message(
            type="subscription_confirmed",
            payload={
                "subscription_id": subscription_id,
                "stream_types": stream_types
            },
            user_id="system",
            session_id="system",
            workspace_id=message.workspace_id,
            timestamp=message.timestamp
        )
        
        await websocket_manager.send_to_user(message.user_id, confirm_message)
        
    except Exception as e:
        logger.error(f"Error handling stream subscription: {e}")

async def handle_presence_update(connection, message: Message):
    """Handle presence updates"""
    try:
        payload = message.payload
        status = payload.get("status", "online")
        document_id = payload.get("document_id")
        activity = payload.get("activity", "viewing")
        
        # Update presence
        collaboration_manager.update_user_presence(
            user_id=message.user_id,
            workspace_id=message.workspace_id,
            document_id=document_id,
            status=status,
            activity=activity,
            session_id=message.session_id
        )
        
        # Broadcast presence update
        presence_message = Message(
            type="presence_updated",
            payload={
                "user_id": message.user_id,
                "status": status,
                "document_id": document_id,
                "activity": activity,
                "timestamp": message.timestamp
            },
            user_id="system",
            session_id="system",
            workspace_id=message.workspace_id,
            timestamp=message.timestamp
        )
        
        await websocket_manager.broadcast_to_workspace(
            message.workspace_id, presence_message, exclude_user=message.user_id
        )
        
    except Exception as e:
        logger.error(f"Error handling presence update: {e}")

# REST API Endpoints for management and status

@app.get("/api/workspace/{workspace_id}/status")
async def get_workspace_status(workspace_id: str):
    """Get workspace collaboration status"""
    return {
        "connections": websocket_manager.get_workspace_stats(workspace_id),
        "collaboration": collaboration_manager.get_workspace_stats(workspace_id),
        "streams": data_stream_manager.get_subscription_stats()
    }

@app.get("/api/document/{document_id}/collaboration")
async def get_document_collaboration(document_id: str):
    """Get document collaboration info"""
    return collaboration_manager.get_document_collaboration_info(document_id)

@app.get("/api/document/{document_id}/comments")
async def get_document_comments(document_id: str, include_resolved: bool = False):
    """Get document comments"""
    comments = collaboration_manager.get_document_comments(document_id, include_resolved)
    return [
        {
            "comment_id": c.comment_id,
            "user_id": c.user_id,
            "content": c.content,
            "position": c.position,
            "status": c.status.value,
            "created_at": c.created_at,
            "replies": [
                {
                    "reply_id": r.reply_id,
                    "user_id": r.user_id,
                    "content": r.content,
                    "created_at": r.created_at
                } for r in c.replies
            ]
        } for c in comments
    ]

@app.get("/api/document/{document_id}/annotations")
async def get_document_annotations(document_id: str):
    """Get document annotations"""
    annotations = collaboration_manager.get_document_annotations(document_id)
    return [
        {
            "annotation_id": a.annotation_id,
            "user_id": a.user_id,
            "type": a.type,
            "content": a.content,
            "position": a.position,
            "style": a.style,
            "created_at": a.created_at
        } for a in annotations
    ]

@app.get("/api/system/stats")
async def get_system_stats():
    """Get overall system statistics"""
    return {
        "websockets": websocket_manager.get_global_stats(),
        "streams": data_stream_manager.get_subscription_stats(),
        "documents": len(ot_engine.documents),
        "total_operations": sum(
            doc.version for doc in ot_engine.documents.values()
        )
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)