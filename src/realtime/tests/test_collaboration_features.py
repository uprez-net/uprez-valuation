"""
Tests for Collaboration Features
"""

import pytest
import time
from unittest.mock import Mock

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.collaboration_features import (
    CollaborationManager, User, Comment, CommentReply, Annotation, 
    UserPresence, Activity, ActivityType, CommentStatus
)

class TestUser:
    """Test User class"""
    
    def test_user_creation(self):
        """Test user object creation"""
        user = User(
            user_id="user1",
            username="testuser",
            display_name="Test User",
            email="test@example.com",
            role="editor"
        )
        
        assert user.user_id == "user1"
        assert user.username == "testuser"
        assert user.display_name == "Test User"
        assert user.email == "test@example.com"
        assert user.role == "editor"
        assert user.status == "offline"  # default
        assert user.last_seen > 0
        assert user.metadata == {}

class TestComment:
    """Test Comment class"""
    
    def test_comment_creation(self):
        """Test comment object creation"""
        comment = Comment(
            comment_id="comment1",
            document_id="doc1",
            workspace_id="workspace1",
            user_id="user1",
            content="This is a test comment",
            position={"line": 5, "column": 10},
            status=CommentStatus.ACTIVE,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        assert comment.comment_id == "comment1"
        assert comment.content == "This is a test comment"
        assert comment.status == CommentStatus.ACTIVE
        assert comment.position["line"] == 5
        assert len(comment.replies) == 0
        assert len(comment.tags) == 0

class TestCollaborationManager:
    """Test collaboration manager functionality"""
    
    def setup_method(self):
        self.manager = CollaborationManager()
    
    def test_add_user(self):
        """Test adding a user"""
        user = User(
            user_id="user1",
            username="testuser",
            display_name="Test User",
            email="test@example.com"
        )
        
        result = self.manager.add_user(user)
        assert result is True
        assert "user1" in self.manager.users
        assert self.manager.users["user1"] == user
    
    def test_get_user(self):
        """Test getting a user"""
        user = User(
            user_id="user1",
            username="testuser",
            display_name="Test User",
            email="test@example.com"
        )
        self.manager.add_user(user)
        
        retrieved_user = self.manager.get_user("user1")
        assert retrieved_user == user
        
        non_existent = self.manager.get_user("nonexistent")
        assert non_existent is None
    
    def test_update_user_status(self):
        """Test updating user status"""
        user = User(
            user_id="user1",
            username="testuser",
            display_name="Test User",
            email="test@example.com"
        )
        self.manager.add_user(user)
        
        result = self.manager.update_user_status("user1", "online")
        assert result is True
        assert self.manager.users["user1"].status == "online"
        
        # Test non-existent user
        result = self.manager.update_user_status("nonexistent", "online")
        assert result is False
    
    def test_add_comment(self):
        """Test adding a comment"""
        comment = self.manager.add_comment(
            document_id="doc1",
            workspace_id="workspace1",
            user_id="user1",
            content="Test comment",
            position={"line": 1}
        )
        
        assert comment.comment_id is not None
        assert comment.content == "Test comment"
        assert comment.status == CommentStatus.ACTIVE
        assert comment.comment_id in self.manager.comments
        assert comment.comment_id in self.manager.document_comments["doc1"]
    
    def test_reply_to_comment(self):
        """Test replying to a comment"""
        # First add a comment
        comment = self.manager.add_comment(
            document_id="doc1",
            workspace_id="workspace1",
            user_id="user1",
            content="Original comment"
        )
        
        # Reply to the comment
        reply = self.manager.reply_to_comment(
            comment_id=comment.comment_id,
            user_id="user2",
            content="This is a reply"
        )
        
        assert reply is not None
        assert reply.content == "This is a reply"
        assert reply.comment_id == comment.comment_id
        assert len(comment.replies) == 1
        assert comment.replies[0] == reply
    
    def test_resolve_comment(self):
        """Test resolving a comment"""
        comment = self.manager.add_comment(
            document_id="doc1",
            workspace_id="workspace1",
            user_id="user1",
            content="Test comment"
        )
        
        result = self.manager.resolve_comment(comment.comment_id, "user2")
        assert result is True
        assert comment.status == CommentStatus.RESOLVED
        assert comment.resolved_by == "user2"
        assert comment.resolved_at is not None
    
    def test_get_document_comments(self):
        """Test getting document comments"""
        # Add active comment
        active_comment = self.manager.add_comment(
            document_id="doc1",
            workspace_id="workspace1",
            user_id="user1",
            content="Active comment"
        )
        
        # Add resolved comment
        resolved_comment = self.manager.add_comment(
            document_id="doc1",
            workspace_id="workspace1",
            user_id="user2",
            content="Resolved comment"
        )
        self.manager.resolve_comment(resolved_comment.comment_id, "user1")
        
        # Get active comments only
        active_comments = self.manager.get_document_comments("doc1", include_resolved=False)
        assert len(active_comments) == 1
        assert active_comments[0] == active_comment
        
        # Get all comments
        all_comments = self.manager.get_document_comments("doc1", include_resolved=True)
        assert len(all_comments) == 2
    
    def test_add_annotation(self):
        """Test adding an annotation"""
        annotation = self.manager.add_annotation(
            document_id="doc1",
            workspace_id="workspace1",
            user_id="user1",
            annotation_type="highlight",
            content="Important section",
            position={"start": 10, "end": 20},
            style={"backgroundColor": "yellow"}
        )
        
        assert annotation.annotation_id is not None
        assert annotation.type == "highlight"
        assert annotation.content == "Important section"
        assert annotation.position["start"] == 10
        assert annotation.style["backgroundColor"] == "yellow"
        assert annotation.annotation_id in self.manager.annotations
        assert annotation.annotation_id in self.manager.document_annotations["doc1"]
    
    def test_update_annotation(self):
        """Test updating an annotation"""
        annotation = self.manager.add_annotation(
            document_id="doc1",
            workspace_id="workspace1",
            user_id="user1",
            annotation_type="highlight",
            content="Original content",
            position={"start": 10, "end": 20}
        )
        
        result = self.manager.update_annotation(
            annotation_id=annotation.annotation_id,
            content="Updated content",
            style={"backgroundColor": "blue"}
        )
        
        assert result is True
        assert annotation.content == "Updated content"
        assert annotation.style["backgroundColor"] == "blue"
    
    def test_delete_annotation(self):
        """Test deleting an annotation"""
        annotation = self.manager.add_annotation(
            document_id="doc1",
            workspace_id="workspace1",
            user_id="user1",
            annotation_type="highlight",
            content="To be deleted",
            position={"start": 10, "end": 20}
        )
        
        result = self.manager.delete_annotation(annotation.annotation_id)
        assert result is True
        assert annotation.annotation_id not in self.manager.annotations
        assert annotation.annotation_id not in self.manager.document_annotations["doc1"]
    
    def test_get_document_annotations(self):
        """Test getting document annotations"""
        annotation1 = self.manager.add_annotation(
            document_id="doc1",
            workspace_id="workspace1",
            user_id="user1",
            annotation_type="highlight",
            content="First annotation",
            position={"start": 10, "end": 20}
        )
        
        annotation2 = self.manager.add_annotation(
            document_id="doc1",
            workspace_id="workspace1",
            user_id="user2",
            annotation_type="note",
            content="Second annotation",
            position={"start": 30, "end": 40}
        )
        
        annotations = self.manager.get_document_annotations("doc1")
        assert len(annotations) == 2
        assert annotation1 in annotations
        assert annotation2 in annotations
    
    def test_update_user_presence(self):
        """Test updating user presence"""
        presence = self.manager.update_user_presence(
            user_id="user1",
            workspace_id="workspace1",
            document_id="doc1",
            status="online",
            cursor_position={"position": 10, "timestamp": time.time()},
            activity="editing"
        )
        
        assert presence.user_id == "user1"
        assert presence.status == "online"
        assert presence.document_id == "doc1"
        assert presence.current_activity == "editing"
        assert "user1" in self.manager.user_presence
        assert "user1" in self.manager.workspace_presence["workspace1"]
        assert "user1" in self.manager.document_presence["doc1"]
    
    def test_remove_user_presence(self):
        """Test removing user presence"""
        self.manager.update_user_presence(
            user_id="user1",
            workspace_id="workspace1",
            document_id="doc1",
            status="online"
        )
        
        self.manager.remove_user_presence("user1")
        
        assert "user1" not in self.manager.user_presence
        assert "user1" not in self.manager.workspace_presence.get("workspace1", set())
        assert "user1" not in self.manager.document_presence.get("doc1", set())
    
    def test_get_workspace_presence(self):
        """Test getting workspace presence"""
        self.manager.update_user_presence(
            user_id="user1",
            workspace_id="workspace1",
            status="online"
        )
        
        self.manager.update_user_presence(
            user_id="user2",
            workspace_id="workspace1",
            status="online"
        )
        
        presence_users = self.manager.get_workspace_presence("workspace1")
        assert len(presence_users) == 2
        user_ids = [p.user_id for p in presence_users]
        assert "user1" in user_ids
        assert "user2" in user_ids
    
    def test_get_document_presence(self):
        """Test getting document presence"""
        self.manager.update_user_presence(
            user_id="user1",
            workspace_id="workspace1",
            document_id="doc1",
            status="online"
        )
        
        self.manager.update_user_presence(
            user_id="user2",
            workspace_id="workspace1",
            document_id="doc2",  # Different document
            status="online"
        )
        
        doc1_presence = self.manager.get_document_presence("doc1")
        assert len(doc1_presence) == 1
        assert doc1_presence[0].user_id == "user1"
        
        doc2_presence = self.manager.get_document_presence("doc2")
        assert len(doc2_presence) == 1
        assert doc2_presence[0].user_id == "user2"
    
    def test_get_workspace_stats(self):
        """Test getting workspace statistics"""
        # Add some test data
        self.manager.add_comment(
            document_id="doc1",
            workspace_id="workspace1",
            user_id="user1",
            content="Comment 1"
        )
        
        self.manager.add_annotation(
            document_id="doc1",
            workspace_id="workspace1",
            user_id="user1",
            annotation_type="highlight",
            content="Annotation 1",
            position={"start": 0, "end": 10}
        )
        
        self.manager.update_user_presence(
            user_id="user1",
            workspace_id="workspace1",
            status="online"
        )
        
        stats = self.manager.get_workspace_stats("workspace1")
        
        assert stats["total_comments"] == 1
        assert stats["total_annotations"] == 1
        assert stats["active_users"] == 1
        assert stats["total_users"] == 1
    
    def test_get_document_collaboration_info(self):
        """Test getting document collaboration info"""
        # Add comment
        comment = self.manager.add_comment(
            document_id="doc1",
            workspace_id="workspace1",
            user_id="user1",
            content="Test comment"
        )
        
        # Add reply
        self.manager.reply_to_comment(comment.comment_id, "user2", "Test reply")
        
        # Add annotation
        self.manager.add_annotation(
            document_id="doc1",
            workspace_id="workspace1",
            user_id="user1",
            annotation_type="highlight",
            content="Test annotation",
            position={"start": 0, "end": 10}
        )
        
        # Add presence
        self.manager.update_user_presence(
            user_id="user1",
            workspace_id="workspace1",
            document_id="doc1",
            cursor_position={"position": 10, "timestamp": time.time()}
        )
        
        info = self.manager.get_document_collaboration_info("doc1")
        
        assert info["active_comments"] == 1
        assert info["resolved_comments"] == 0
        assert info["total_replies"] == 1
        assert info["annotations"] == 1
        assert info["present_users"] == 1
        assert "user1" in info["cursors"]
    
    def test_cleanup_stale_presence(self):
        """Test cleanup of stale presence data"""
        # Add user with old timestamp
        old_time = time.time() - 400  # 400 seconds ago
        presence = self.manager.update_user_presence(
            user_id="user1",
            workspace_id="workspace1",
            status="online"
        )
        presence.last_activity = old_time
        
        # Cleanup with 300 second timeout
        self.manager.cleanup_stale_presence(300)
        
        assert "user1" not in self.manager.user_presence

if __name__ == "__main__":
    pytest.main([__file__, "-v"])