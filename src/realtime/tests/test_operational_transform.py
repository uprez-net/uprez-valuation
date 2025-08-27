"""
Tests for Operational Transformation Engine
"""

import pytest
import time
from unittest.mock import Mock

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.operational_transform import (
    OperationalTransform, DocumentSync, Operation, OperationType, Delta, Document
)

class TestOperation:
    """Test Operation class"""
    
    def test_operation_creation(self):
        """Test operation object creation"""
        op = Operation(
            type=OperationType.INSERT,
            position=5,
            content="hello",
            user_id="user1"
        )
        
        assert op.type == OperationType.INSERT
        assert op.position == 5
        assert op.content == "hello"
        assert op.user_id == "user1"
        assert op.operation_id is not None
        assert op.timestamp > 0
    
    def test_operation_id_generation(self):
        """Test automatic operation ID generation"""
        op1 = Operation(OperationType.INSERT, 0, "a", user_id="user1")
        op2 = Operation(OperationType.INSERT, 0, "b", user_id="user1")
        
        assert op1.operation_id != op2.operation_id

class TestDelta:
    """Test Delta class"""
    
    def test_delta_creation(self):
        """Test delta object creation"""
        ops = [
            Operation(OperationType.INSERT, 0, "hello", user_id="user1"),
            Operation(OperationType.INSERT, 5, " world", user_id="user1")
        ]
        
        delta = Delta(
            operations=ops,
            base_version=0,
            result_version=1,
            document_id="doc1",
            user_id="user1",
            session_id="session1",
            timestamp=time.time()
        )
        
        assert len(delta.operations) == 2
        assert delta.base_version == 0
        assert delta.result_version == 1
        assert delta.delta_id is not None

class TestOperationalTransform:
    """Test OT engine functionality"""
    
    def setup_method(self):
        self.ot = OperationalTransform()
    
    def test_create_document(self):
        """Test document creation"""
        doc = self.ot.create_document("doc1", "initial content", {"author": "user1"})
        
        assert doc.document_id == "doc1"
        assert doc.content == "initial content"
        assert doc.version == 0
        assert doc.metadata["author"] == "user1"
        assert "doc1" in self.ot.documents
    
    def test_get_document(self):
        """Test document retrieval"""
        self.ot.create_document("doc1", "content")
        
        doc = self.ot.get_document("doc1")
        assert doc is not None
        assert doc.document_id == "doc1"
        
        non_existent = self.ot.get_document("nonexistent")
        assert non_existent is None
    
    def test_apply_delta_direct(self):
        """Test direct delta application (no conflicts)"""
        doc = self.ot.create_document("doc1", "hello")
        
        ops = [Operation(OperationType.INSERT, 5, " world", user_id="user1")]
        delta = Delta(
            operations=ops,
            base_version=0,
            result_version=1,
            document_id="doc1",
            user_id="user1",
            session_id="session1",
            timestamp=time.time()
        )
        
        success, error, updated_doc = self.ot.apply_delta("doc1", delta)
        
        assert success is True
        assert error == ""
        assert updated_doc.content == "hello world"
        assert updated_doc.version == 1
    
    def test_apply_insert_operation(self):
        """Test insert operation"""
        self.ot.create_document("doc1", "hello")
        
        ops = [Operation(OperationType.INSERT, 0, "Hi ", user_id="user1")]
        delta = Delta(
            operations=ops,
            base_version=0,
            result_version=1,
            document_id="doc1",
            user_id="user1",
            session_id="session1",
            timestamp=time.time()
        )
        
        success, error, updated_doc = self.ot.apply_delta("doc1", delta)
        
        assert success is True
        assert updated_doc.content == "Hi hello"
    
    def test_apply_delete_operation(self):
        """Test delete operation"""
        self.ot.create_document("doc1", "hello world")
        
        ops = [Operation(OperationType.DELETE, 5, length=6, user_id="user1")]  # Delete " world"
        delta = Delta(
            operations=ops,
            base_version=0,
            result_version=1,
            document_id="doc1",
            user_id="user1",
            session_id="session1",
            timestamp=time.time()
        )
        
        success, error, updated_doc = self.ot.apply_delta("doc1", delta)
        
        assert success is True
        assert updated_doc.content == "hello"
    
    def test_transform_insert_vs_insert(self):
        """Test transformation of two insert operations"""
        op1 = Operation(OperationType.INSERT, 5, "ABC", user_id="user1")
        op2 = Operation(OperationType.INSERT, 3, "XYZ", user_id="user2")
        
        transformed = self.ot._transform_operation(op1, op2)
        
        # op1 should be adjusted because op2 comes before it
        assert transformed.position == 8  # 5 + 3 (length of "XYZ")
        assert transformed.content == "ABC"
    
    def test_transform_insert_vs_delete(self):
        """Test transformation of insert vs delete operations"""
        op1 = Operation(OperationType.INSERT, 5, "ABC", user_id="user1")
        op2 = Operation(OperationType.DELETE, 2, length=3, user_id="user2")
        
        transformed = self.ot._transform_operation(op1, op2)
        
        # op1 position should be adjusted for the deletion
        assert transformed.position == 2  # 5 - 3 (deletion length)
        assert transformed.content == "ABC"
    
    def test_transform_delete_vs_insert(self):
        """Test transformation of delete vs insert operations"""
        op1 = Operation(OperationType.DELETE, 5, length=3, user_id="user1")
        op2 = Operation(OperationType.INSERT, 2, "XYZ", user_id="user2")
        
        transformed = self.ot._transform_operation(op1, op2)
        
        # op1 position should be adjusted for the insertion
        assert transformed.position == 8  # 5 + 3 (length of "XYZ")
        assert transformed.length == 3
    
    def test_transform_delete_vs_delete_before(self):
        """Test transformation of two delete operations (second before first)"""
        op1 = Operation(OperationType.DELETE, 10, length=5, user_id="user1")
        op2 = Operation(OperationType.DELETE, 5, length=3, user_id="user2")
        
        transformed = self.ot._transform_operation(op1, op2)
        
        # op1 position should be adjusted
        assert transformed.position == 7  # 10 - 3 (op2 length)
        assert transformed.length == 5
    
    def test_transform_delete_vs_delete_overlapping(self):
        """Test transformation of overlapping delete operations"""
        op1 = Operation(OperationType.DELETE, 5, length=5, user_id="user1")  # Delete pos 5-10
        op2 = Operation(OperationType.DELETE, 7, length=3, user_id="user2")  # Delete pos 7-10
        
        transformed = self.ot._transform_operation(op1, op2)
        
        # op1 should be adjusted for overlap
        assert transformed.position == 5
        assert transformed.length == 2  # 5 - 3 (overlap)
    
    def test_create_delta_from_diff(self):
        """Test creating delta from content difference"""
        old_content = "hello world"
        new_content = "hello beautiful world"
        
        delta = self.ot.create_delta_from_diff(
            document_id="doc1",
            old_content=old_content,
            new_content=new_content,
            user_id="user1",
            session_id="session1",
            base_version=0
        )
        
        assert len(delta.operations) > 0
        assert delta.document_id == "doc1"
        assert delta.user_id == "user1"
    
    def test_get_document_at_version(self):
        """Test reconstructing document at specific version"""
        # Create document and apply several deltas
        self.ot.create_document("doc1", "")
        
        # Apply first delta
        ops1 = [Operation(OperationType.INSERT, 0, "hello", user_id="user1")]
        delta1 = Delta(
            operations=ops1,
            base_version=0,
            result_version=1,
            document_id="doc1",
            user_id="user1",
            session_id="session1",
            timestamp=time.time()
        )
        self.ot.apply_delta("doc1", delta1)
        
        # Apply second delta
        ops2 = [Operation(OperationType.INSERT, 5, " world", user_id="user1")]
        delta2 = Delta(
            operations=ops2,
            base_version=1,
            result_version=2,
            document_id="doc1",
            user_id="user1",
            session_id="session1",
            timestamp=time.time()
        )
        self.ot.apply_delta("doc1", delta2)
        
        # Test version reconstruction
        version_0 = self.ot.get_document_at_version("doc1", 0)
        version_1 = self.ot.get_document_at_version("doc1", 1)
        version_2 = self.ot.get_document_at_version("doc1", 2)
        
        assert version_0 == ""
        assert version_1 == "hello"
        assert version_2 == "hello world"
    
    def test_get_operation_stats(self):
        """Test operation statistics"""
        doc = self.ot.create_document("doc1", "hello")
        
        ops = [
            Operation(OperationType.INSERT, 5, " world", user_id="user1"),
            Operation(OperationType.DELETE, 0, length=1, user_id="user1")
        ]
        delta = Delta(
            operations=ops,
            base_version=0,
            result_version=1,
            document_id="doc1",
            user_id="user1",
            session_id="session1",
            timestamp=time.time()
        )
        self.ot.apply_delta("doc1", delta)
        
        stats = self.ot.get_operation_stats("doc1")
        
        assert stats["document_id"] == "doc1"
        assert stats["version"] == 1
        assert stats["total_deltas"] == 1
        assert stats["total_operations"] == 2
        assert "insert" in stats["operations_by_type"]
        assert "delete" in stats["operations_by_type"]
        assert "user1" in stats["operations_by_user"]

class TestDocumentSync:
    """Test document synchronization"""
    
    def setup_method(self):
        self.sync = DocumentSync()
    
    def test_update_cursor(self):
        """Test cursor position update"""
        self.sync.update_cursor("doc1", "user1", 10, {"line": 1, "column": 10})
        
        presence = self.sync.get_document_presence("doc1")
        assert "user1" in presence["cursors"]
        assert presence["cursors"]["user1"]["position"] == 10
        assert presence["cursors"]["user1"]["metadata"]["line"] == 1
    
    def test_update_selection(self):
        """Test selection range update"""
        self.sync.update_selection("doc1", "user1", 5, 15, {"type": "highlight"})
        
        presence = self.sync.get_document_presence("doc1")
        assert "user1" in presence["selections"]
        assert presence["selections"]["user1"]["start"] == 5
        assert presence["selections"]["user1"]["end"] == 15
    
    def test_get_document_presence(self):
        """Test getting document presence information"""
        self.sync.update_cursor("doc1", "user1", 10)
        self.sync.update_selection("doc1", "user2", 5, 15)
        
        presence = self.sync.get_document_presence("doc1")
        
        assert len(presence["cursors"]) == 1
        assert len(presence["selections"]) == 1
        assert "user1" in presence["active_users"]
        assert "user2" in presence["active_users"]
    
    def test_cleanup_stale_presence(self):
        """Test cleanup of stale presence data"""
        # Add presence data with old timestamp
        old_time = time.time() - 400  # 400 seconds ago
        self.sync.cursors["doc1"] = {
            "user1": {"position": 10, "timestamp": old_time, "metadata": {}}
        }
        self.sync.selections["doc1"] = {
            "user1": {"start": 5, "end": 15, "timestamp": old_time, "metadata": {}}
        }
        
        # Cleanup with 300 second timeout
        self.sync.cleanup_stale_presence("doc1", 300)
        
        presence = self.sync.get_document_presence("doc1")
        assert len(presence["cursors"]) == 0
        assert len(presence["selections"]) == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])