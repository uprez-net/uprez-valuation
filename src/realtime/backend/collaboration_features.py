"""
Collaboration Features Module
Handles real-time commenting, annotations, user presence, and team workspace management
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)

class ActivityType(Enum):
    """Types of user activities"""
    DOCUMENT_OPEN = "document_open"
    DOCUMENT_EDIT = "document_edit"
    COMMENT_ADD = "comment_add"
    COMMENT_REPLY = "comment_reply"
    ANNOTATION_CREATE = "annotation_create"
    VALUATION_UPDATE = "valuation_update"
    USER_JOIN = "user_join"
    USER_LEAVE = "user_leave"
    PERMISSION_CHANGE = "permission_change"

class CommentStatus(Enum):
    """Comment status types"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ARCHIVED = "archived"

@dataclass
class User:
    """User information"""
    user_id: str
    username: str
    display_name: str
    email: str
    avatar_url: Optional[str] = None
    role: str = "viewer"
    status: str = "offline"
    last_seen: float = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.last_seen:
            self.last_seen = time.time()

@dataclass
class Comment:
    """Comment on a document"""
    comment_id: str
    document_id: str
    workspace_id: str
    user_id: str
    content: str
    position: Optional[Dict[str, Any]]  # Position in document (line, column, range, etc.)
    status: CommentStatus
    created_at: float
    updated_at: float
    resolved_by: Optional[str] = None
    resolved_at: Optional[float] = None
    replies: List['CommentReply'] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.replies is None:
            self.replies = []
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class CommentReply:
    """Reply to a comment"""
    reply_id: str
    comment_id: str
    user_id: str
    content: str
    created_at: float
    updated_at: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Annotation:
    """Document annotation"""
    annotation_id: str
    document_id: str
    workspace_id: str
    user_id: str
    type: str  # highlight, note, bookmark, etc.
    content: str
    position: Dict[str, Any]  # Start, end positions
    style: Dict[str, Any]  # Color, font, etc.
    created_at: float
    updated_at: float
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class UserPresence:
    """User presence information"""
    user_id: str
    workspace_id: str
    document_id: Optional[str]
    status: str  # online, away, busy, offline
    cursor_position: Optional[Dict[str, Any]]
    selection_range: Optional[Dict[str, Any]]
    current_activity: str
    last_activity: float
    session_id: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Activity:
    """Activity feed entry"""
    activity_id: str
    workspace_id: str
    user_id: str
    activity_type: ActivityType
    target_id: str  # Document ID, comment ID, etc.
    description: str
    data: Dict[str, Any]
    created_at: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class CollaborationManager:
    """Manages collaboration features"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.comments: Dict[str, Comment] = {}  # comment_id -> Comment
        self.annotations: Dict[str, Annotation] = {}  # annotation_id -> Annotation
        self.user_presence: Dict[str, UserPresence] = {}  # user_id -> UserPresence
        self.activities: Dict[str, List[Activity]] = {}  # workspace_id -> List[Activity]
        
        # Indexes for efficient querying
        self.document_comments: Dict[str, Set[str]] = {}  # document_id -> set of comment_ids
        self.document_annotations: Dict[str, Set[str]] = {}  # document_id -> set of annotation_ids
        self.workspace_presence: Dict[str, Set[str]] = {}  # workspace_id -> set of user_ids
        self.document_presence: Dict[str, Set[str]] = {}  # document_id -> set of user_ids
    
    # User Management
    def add_user(self, user: User) -> bool:
        """Add or update user information"""
        self.users[user.user_id] = user
        logger.info(f"User added/updated: {user.user_id}")
        return True
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def update_user_status(self, user_id: str, status: str) -> bool:
        """Update user online status"""
        if user_id in self.users:
            self.users[user_id].status = status
            self.users[user_id].last_seen = time.time()
            return True
        return False
    
    # Comment Management
    def add_comment(self, document_id: str, workspace_id: str, user_id: str,
                   content: str, position: Dict[str, Any] = None) -> Comment:
        """Add a new comment"""
        comment = Comment(
            comment_id=str(uuid4()),
            document_id=document_id,
            workspace_id=workspace_id,
            user_id=user_id,
            content=content,
            position=position,
            status=CommentStatus.ACTIVE,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        self.comments[comment.comment_id] = comment
        
        # Update index
        if document_id not in self.document_comments:
            self.document_comments[document_id] = set()
        self.document_comments[document_id].add(comment.comment_id)
        
        # Add to activity feed
        self._add_activity(workspace_id, user_id, ActivityType.COMMENT_ADD,
                          comment.comment_id, f"Added comment: {content[:50]}...",
                          {"comment_id": comment.comment_id, "document_id": document_id})
        
        logger.info(f"Comment added: {comment.comment_id}")
        return comment
    
    def reply_to_comment(self, comment_id: str, user_id: str, content: str) -> Optional[CommentReply]:
        """Add a reply to a comment"""
        if comment_id not in self.comments:
            return None
        
        comment = self.comments[comment_id]
        reply = CommentReply(
            reply_id=str(uuid4()),
            comment_id=comment_id,
            user_id=user_id,
            content=content,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        comment.replies.append(reply)
        comment.updated_at = time.time()
        
        # Add to activity feed
        self._add_activity(comment.workspace_id, user_id, ActivityType.COMMENT_REPLY,
                          comment_id, f"Replied to comment: {content[:50]}...",
                          {"reply_id": reply.reply_id, "comment_id": comment_id})
        
        logger.info(f"Reply added: {reply.reply_id} to comment {comment_id}")
        return reply
    
    def resolve_comment(self, comment_id: str, user_id: str) -> bool:
        """Mark comment as resolved"""
        if comment_id not in self.comments:
            return False
        
        comment = self.comments[comment_id]
        comment.status = CommentStatus.RESOLVED
        comment.resolved_by = user_id
        comment.resolved_at = time.time()
        comment.updated_at = time.time()
        
        logger.info(f"Comment resolved: {comment_id} by {user_id}")
        return True
    
    def get_document_comments(self, document_id: str, include_resolved: bool = False) -> List[Comment]:
        """Get all comments for a document"""
        comment_ids = self.document_comments.get(document_id, set())
        comments = [self.comments[cid] for cid in comment_ids if cid in self.comments]
        
        if not include_resolved:
            comments = [c for c in comments if c.status == CommentStatus.ACTIVE]
        
        return sorted(comments, key=lambda c: c.created_at)
    
    # Annotation Management
    def add_annotation(self, document_id: str, workspace_id: str, user_id: str,
                      annotation_type: str, content: str, position: Dict[str, Any],
                      style: Dict[str, Any] = None) -> Annotation:
        """Add a new annotation"""
        annotation = Annotation(
            annotation_id=str(uuid4()),
            document_id=document_id,
            workspace_id=workspace_id,
            user_id=user_id,
            type=annotation_type,
            content=content,
            position=position,
            style=style or {},
            created_at=time.time(),
            updated_at=time.time()
        )
        
        self.annotations[annotation.annotation_id] = annotation
        
        # Update index
        if document_id not in self.document_annotations:
            self.document_annotations[document_id] = set()
        self.document_annotations[document_id].add(annotation.annotation_id)
        
        # Add to activity feed
        self._add_activity(workspace_id, user_id, ActivityType.ANNOTATION_CREATE,
                          annotation.annotation_id, f"Added {annotation_type}: {content[:50]}...",
                          {"annotation_id": annotation.annotation_id, "document_id": document_id})
        
        logger.info(f"Annotation added: {annotation.annotation_id}")
        return annotation
    
    def update_annotation(self, annotation_id: str, content: str = None,
                         position: Dict[str, Any] = None, style: Dict[str, Any] = None) -> bool:
        """Update an existing annotation"""
        if annotation_id not in self.annotations:
            return False
        
        annotation = self.annotations[annotation_id]
        
        if content is not None:
            annotation.content = content
        if position is not None:
            annotation.position = position
        if style is not None:
            annotation.style = style
        
        annotation.updated_at = time.time()
        
        logger.info(f"Annotation updated: {annotation_id}")
        return True
    
    def delete_annotation(self, annotation_id: str) -> bool:
        """Delete an annotation"""
        if annotation_id not in self.annotations:
            return False
        
        annotation = self.annotations[annotation_id]
        document_id = annotation.document_id
        
        # Remove from indexes
        if document_id in self.document_annotations:
            self.document_annotations[document_id].discard(annotation_id)
        
        del self.annotations[annotation_id]
        
        logger.info(f"Annotation deleted: {annotation_id}")
        return True
    
    def get_document_annotations(self, document_id: str) -> List[Annotation]:
        """Get all annotations for a document"""
        annotation_ids = self.document_annotations.get(document_id, set())
        annotations = [self.annotations[aid] for aid in annotation_ids if aid in self.annotations]
        return sorted(annotations, key=lambda a: a.created_at)
    
    # Presence Management
    def update_user_presence(self, user_id: str, workspace_id: str, document_id: str = None,
                           status: str = "online", cursor_position: Dict[str, Any] = None,
                           selection_range: Dict[str, Any] = None, activity: str = "viewing",
                           session_id: str = None) -> UserPresence:
        """Update user presence information"""
        presence = UserPresence(
            user_id=user_id,
            workspace_id=workspace_id,
            document_id=document_id,
            status=status,
            cursor_position=cursor_position,
            selection_range=selection_range,
            current_activity=activity,
            last_activity=time.time(),
            session_id=session_id or str(uuid4())
        )
        
        self.user_presence[user_id] = presence
        
        # Update indexes
        if workspace_id not in self.workspace_presence:
            self.workspace_presence[workspace_id] = set()
        self.workspace_presence[workspace_id].add(user_id)
        
        if document_id:
            if document_id not in self.document_presence:
                self.document_presence[document_id] = set()
            self.document_presence[document_id].add(user_id)
        
        # Update user status
        self.update_user_status(user_id, status)
        
        return presence
    
    def remove_user_presence(self, user_id: str):
        """Remove user from presence tracking"""
        if user_id not in self.user_presence:
            return
        
        presence = self.user_presence[user_id]
        
        # Remove from workspace index
        if presence.workspace_id in self.workspace_presence:
            self.workspace_presence[presence.workspace_id].discard(user_id)
        
        # Remove from document index
        if presence.document_id and presence.document_id in self.document_presence:
            self.document_presence[presence.document_id].discard(user_id)
        
        del self.user_presence[user_id]
        
        # Update user status to offline
        self.update_user_status(user_id, "offline")
        
        logger.info(f"User presence removed: {user_id}")
    
    def get_workspace_presence(self, workspace_id: str) -> List[UserPresence]:
        """Get all users present in a workspace"""
        user_ids = self.workspace_presence.get(workspace_id, set())
        return [self.user_presence[uid] for uid in user_ids if uid in self.user_presence]
    
    def get_document_presence(self, document_id: str) -> List[UserPresence]:
        """Get all users present in a document"""
        user_ids = self.document_presence.get(document_id, set())
        return [self.user_presence[uid] for uid in user_ids if uid in self.user_presence]
    
    # Activity Feed
    def _add_activity(self, workspace_id: str, user_id: str, activity_type: ActivityType,
                     target_id: str, description: str, data: Dict[str, Any] = None):
        """Add activity to workspace feed"""
        activity = Activity(
            activity_id=str(uuid4()),
            workspace_id=workspace_id,
            user_id=user_id,
            activity_type=activity_type,
            target_id=target_id,
            description=description,
            data=data or {},
            created_at=time.time()
        )
        
        if workspace_id not in self.activities:
            self.activities[workspace_id] = []
        
        self.activities[workspace_id].append(activity)
        
        # Keep only recent activities (last 1000)
        if len(self.activities[workspace_id]) > 1000:
            self.activities[workspace_id] = self.activities[workspace_id][-1000:]
    
    def get_workspace_activities(self, workspace_id: str, limit: int = 50) -> List[Activity]:
        """Get recent activities for a workspace"""
        activities = self.activities.get(workspace_id, [])
        return sorted(activities, key=lambda a: a.created_at, reverse=True)[:limit]
    
    def get_user_activities(self, workspace_id: str, user_id: str, limit: int = 50) -> List[Activity]:
        """Get activities for a specific user in a workspace"""
        all_activities = self.activities.get(workspace_id, [])
        user_activities = [a for a in all_activities if a.user_id == user_id]
        return sorted(user_activities, key=lambda a: a.created_at, reverse=True)[:limit]
    
    # Team Management
    def get_workspace_stats(self, workspace_id: str) -> Dict[str, Any]:
        """Get collaboration statistics for a workspace"""
        workspace_users = self.get_workspace_presence(workspace_id)
        activities = self.get_workspace_activities(workspace_id, 100)
        
        # Count comments and annotations in workspace
        workspace_comments = 0
        workspace_annotations = 0
        
        for comment in self.comments.values():
            if comment.workspace_id == workspace_id:
                workspace_comments += 1
        
        for annotation in self.annotations.values():
            if annotation.workspace_id == workspace_id:
                workspace_annotations += 1
        
        return {
            "active_users": len([u for u in workspace_users if u.status == "online"]),
            "total_users": len(workspace_users),
            "total_comments": workspace_comments,
            "total_annotations": workspace_annotations,
            "recent_activities": len(activities),
            "activity_types": {
                activity_type.value: len([a for a in activities if a.activity_type == activity_type])
                for activity_type in ActivityType
            }
        }
    
    def get_document_collaboration_info(self, document_id: str) -> Dict[str, Any]:
        """Get collaboration info for a specific document"""
        comments = self.get_document_comments(document_id, include_resolved=True)
        annotations = self.get_document_annotations(document_id)
        present_users = self.get_document_presence(document_id)
        
        return {
            "active_comments": len([c for c in comments if c.status == CommentStatus.ACTIVE]),
            "resolved_comments": len([c for c in comments if c.status == CommentStatus.RESOLVED]),
            "total_replies": sum(len(c.replies) for c in comments),
            "annotations": len(annotations),
            "present_users": len(present_users),
            "cursors": {
                user.user_id: user.cursor_position 
                for user in present_users 
                if user.cursor_position
            },
            "selections": {
                user.user_id: user.selection_range 
                for user in present_users 
                if user.selection_range
            }
        }
    
    # Cleanup Methods
    def cleanup_stale_presence(self, timeout_seconds: int = 300):
        """Remove stale presence data"""
        current_time = time.time()
        stale_users = []
        
        for user_id, presence in self.user_presence.items():
            if current_time - presence.last_activity > timeout_seconds:
                stale_users.append(user_id)
        
        for user_id in stale_users:
            self.remove_user_presence(user_id)
        
        logger.info(f"Cleaned up {len(stale_users)} stale presence records")
    
    def cleanup_old_activities(self, keep_days: int = 30):
        """Remove old activities"""
        cutoff_time = time.time() - (keep_days * 24 * 3600)
        
        for workspace_id, activities in self.activities.items():
            self.activities[workspace_id] = [
                a for a in activities if a.created_at > cutoff_time
            ]
        
        logger.info("Cleaned up old activities")

# Global collaboration manager instance
collaboration_manager = CollaborationManager()