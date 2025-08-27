"""
Operational Transformation Engine for Real-time Collaborative Editing
Handles conflict resolution, document synchronization, and version control
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import uuid4
import copy

class OperationType(Enum):
    """Types of operations supported"""
    INSERT = "insert"
    DELETE = "delete"
    RETAIN = "retain"
    FORMAT = "format"
    MOVE = "move"

@dataclass
class Operation:
    """Single operation in a transformation"""
    type: OperationType
    position: int
    content: str = ""
    length: int = 0
    attributes: Dict[str, Any] = None
    user_id: str = ""
    timestamp: float = 0
    operation_id: str = ""
    
    def __post_init__(self):
        if not self.operation_id:
            self.operation_id = str(uuid4())
        if not self.timestamp:
            self.timestamp = time.time()
        if self.attributes is None:
            self.attributes = {}

@dataclass
class Delta:
    """Collection of operations representing a document change"""
    operations: List[Operation]
    base_version: int
    result_version: int
    document_id: str
    user_id: str
    session_id: str
    timestamp: float
    delta_id: str = ""
    
    def __post_init__(self):
        if not self.delta_id:
            self.delta_id = str(uuid4())

@dataclass
class Document:
    """Document state with version history"""
    document_id: str
    content: str
    version: int
    deltas: List[Delta]
    metadata: Dict[str, Any]
    created_at: float
    updated_at: float
    
    def __post_init__(self):
        if not self.deltas:
            self.deltas = []

class OperationalTransform:
    """Core OT engine for conflict resolution"""
    
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self._locks: Dict[str, bool] = {}
    
    def create_document(self, document_id: str, initial_content: str = "",
                       metadata: Dict[str, Any] = None) -> Document:
        """Create a new document"""
        doc = Document(
            document_id=document_id,
            content=initial_content,
            version=0,
            deltas=[],
            metadata=metadata or {},
            created_at=time.time(),
            updated_at=time.time()
        )
        
        self.documents[document_id] = doc
        self._locks[document_id] = False
        return doc
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID"""
        return self.documents.get(document_id)
    
    def apply_delta(self, document_id: str, delta: Delta) -> Tuple[bool, str, Document]:
        """Apply a delta to a document with conflict resolution"""
        if document_id not in self.documents:
            return False, "Document not found", None
        
        if self._locks.get(document_id, False):
            return False, "Document locked", None
        
        self._locks[document_id] = True
        
        try:
            doc = self.documents[document_id]
            
            # Check if delta is based on current version
            if delta.base_version == doc.version:
                # Direct application
                success, error, new_content = self._apply_operations(doc.content, delta.operations)
                if success:
                    doc.content = new_content
                    doc.version += 1
                    doc.updated_at = time.time()
                    delta.result_version = doc.version
                    doc.deltas.append(copy.deepcopy(delta))
                    return True, "", doc
                else:
                    return False, error, doc
            
            # Need to transform against missed deltas
            transformed_delta = self._transform_delta_against_history(doc, delta)
            if not transformed_delta:
                return False, "Transformation failed", doc
            
            # Apply transformed delta
            success, error, new_content = self._apply_operations(doc.content, transformed_delta.operations)
            if success:
                doc.content = new_content
                doc.version += 1
                doc.updated_at = time.time()
                transformed_delta.result_version = doc.version
                doc.deltas.append(transformed_delta)
                return True, "", doc
            else:
                return False, error, doc
                
        finally:
            self._locks[document_id] = False
    
    def _transform_delta_against_history(self, doc: Document, delta: Delta) -> Optional[Delta]:
        """Transform a delta against missed operations"""
        if delta.base_version >= doc.version:
            return delta
        
        # Get deltas that occurred after the base version
        missed_deltas = [d for d in doc.deltas if d.result_version > delta.base_version]
        
        transformed_ops = delta.operations[:]
        
        # Transform against each missed delta
        for missed_delta in missed_deltas:
            new_ops = []
            for op in transformed_ops:
                for missed_op in missed_delta.operations:
                    op = self._transform_operation(op, missed_op)
                new_ops.append(op)
            transformed_ops = new_ops
        
        # Create new delta with transformed operations
        return Delta(
            operations=transformed_ops,
            base_version=doc.version,
            result_version=doc.version + 1,
            document_id=delta.document_id,
            user_id=delta.user_id,
            session_id=delta.session_id,
            timestamp=time.time(),
            delta_id=str(uuid4())
        )
    
    def _transform_operation(self, op1: Operation, op2: Operation) -> Operation:
        """Transform one operation against another (OT core algorithm)"""
        # Create copy to avoid modifying original
        transformed_op = copy.deepcopy(op1)
        
        if op1.type == OperationType.INSERT and op2.type == OperationType.INSERT:
            # Both insertions
            if op2.position <= op1.position:
                # op2 comes before op1, adjust op1 position
                transformed_op.position += len(op2.content)
            # else: op1 comes before op2, no change needed
            
        elif op1.type == OperationType.INSERT and op2.type == OperationType.DELETE:
            # Insert vs Delete
            if op2.position < op1.position:
                # Deletion before insertion
                transformed_op.position -= min(op2.length, op1.position - op2.position)
            elif op2.position <= op1.position + len(op1.content):
                # Deletion overlaps with insertion - complex case
                # For simplicity, keep insertion position
                pass
            
        elif op1.type == OperationType.DELETE and op2.type == OperationType.INSERT:
            # Delete vs Insert
            if op2.position <= op1.position:
                # Insertion before deletion
                transformed_op.position += len(op2.content)
            
        elif op1.type == OperationType.DELETE and op2.type == OperationType.DELETE:
            # Both deletions
            if op2.position < op1.position:
                # op2 before op1
                overlap = max(0, min(op1.position + op1.length, op2.position + op2.length) - op1.position)
                transformed_op.position = max(op2.position, op1.position - op2.length)
                transformed_op.length = max(0, op1.length - overlap)
            elif op2.position < op1.position + op1.length:
                # Overlapping deletions
                if op2.position + op2.length <= op1.position + op1.length:
                    # op2 contained within op1
                    transformed_op.length -= op2.length
                else:
                    # Partial overlap
                    transformed_op.length = op2.position - op1.position
        
        return transformed_op
    
    def _apply_operations(self, content: str, operations: List[Operation]) -> Tuple[bool, str, str]:
        """Apply a list of operations to content"""
        try:
            result = list(content)
            offset = 0
            
            # Sort operations by position to apply in order
            sorted_ops = sorted(operations, key=lambda op: op.position)
            
            for op in sorted_ops:
                pos = op.position + offset
                
                if op.type == OperationType.INSERT:
                    result[pos:pos] = list(op.content)
                    offset += len(op.content)
                    
                elif op.type == OperationType.DELETE:
                    if pos < len(result) and pos + op.length <= len(result):
                        del result[pos:pos + op.length]
                        offset -= op.length
                    else:
                        return False, f"Delete operation out of bounds: pos={pos}, length={op.length}", ""
                
                elif op.type == OperationType.RETAIN:
                    # Retain operations don't change content, just position
                    offset += op.length
            
            return True, "", "".join(result)
            
        except Exception as e:
            return False, f"Error applying operations: {str(e)}", ""
    
    def get_document_at_version(self, document_id: str, version: int) -> Optional[str]:
        """Reconstruct document content at a specific version"""
        doc = self.documents.get(document_id)
        if not doc:
            return None
        
        if version == 0:
            # Find initial content from first delta or empty
            return ""
        
        if version > doc.version:
            return None
        
        # Start with empty content and apply deltas up to version
        content = ""
        applied_deltas = [d for d in doc.deltas if d.result_version <= version]
        applied_deltas.sort(key=lambda d: d.result_version)
        
        for delta in applied_deltas:
            success, _, new_content = self._apply_operations(content, delta.operations)
            if success:
                content = new_content
        
        return content
    
    def create_delta_from_diff(self, document_id: str, old_content: str, 
                              new_content: str, user_id: str, session_id: str,
                              base_version: int) -> Delta:
        """Create a delta from content difference"""
        operations = self._diff_to_operations(old_content, new_content)
        
        return Delta(
            operations=operations,
            base_version=base_version,
            result_version=base_version + 1,
            document_id=document_id,
            user_id=user_id,
            session_id=session_id,
            timestamp=time.time()
        )
    
    def _diff_to_operations(self, old_content: str, new_content: str) -> List[Operation]:
        """Convert content diff to operations (simplified Myers algorithm)"""
        operations = []
        
        # Simple character-by-character diff
        old_chars = list(old_content)
        new_chars = list(new_content)
        
        # Use dynamic programming for LCS
        lcs_matrix = self._calculate_lcs_matrix(old_chars, new_chars)
        
        # Backtrack to find operations
        operations = self._backtrack_operations(old_chars, new_chars, lcs_matrix)
        
        return operations
    
    def _calculate_lcs_matrix(self, seq1: List[str], seq2: List[str]) -> List[List[int]]:
        """Calculate Longest Common Subsequence matrix"""
        m, n = len(seq1), len(seq2)
        matrix = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    matrix[i][j] = matrix[i-1][j-1] + 1
                else:
                    matrix[i][j] = max(matrix[i-1][j], matrix[i][j-1])
        
        return matrix
    
    def _backtrack_operations(self, old_chars: List[str], new_chars: List[str],
                            lcs_matrix: List[List[int]]) -> List[Operation]:
        """Backtrack LCS matrix to generate operations"""
        operations = []
        i, j = len(old_chars), len(new_chars)
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and old_chars[i-1] == new_chars[j-1]:
                # Characters match, no operation needed
                i -= 1
                j -= 1
            elif j > 0 and (i == 0 or lcs_matrix[i][j-1] >= lcs_matrix[i-1][j]):
                # Insert operation
                operations.append(Operation(
                    type=OperationType.INSERT,
                    position=i,
                    content=new_chars[j-1],
                    timestamp=time.time()
                ))
                j -= 1
            elif i > 0:
                # Delete operation
                operations.append(Operation(
                    type=OperationType.DELETE,
                    position=i-1,
                    length=1,
                    timestamp=time.time()
                ))
                i -= 1
        
        # Reverse to get operations in correct order
        operations.reverse()
        return operations
    
    def merge_deltas(self, document_id: str, deltas: List[Delta]) -> Optional[Delta]:
        """Merge multiple deltas into one (for optimization)"""
        if not deltas:
            return None
        
        if len(deltas) == 1:
            return deltas[0]
        
        # Sort deltas by timestamp
        sorted_deltas = sorted(deltas, key=lambda d: d.timestamp)
        
        # Merge operations
        merged_operations = []
        for delta in sorted_deltas:
            merged_operations.extend(delta.operations)
        
        # Create merged delta
        return Delta(
            operations=merged_operations,
            base_version=sorted_deltas[0].base_version,
            result_version=sorted_deltas[-1].result_version,
            document_id=document_id,
            user_id="system",
            session_id="merge",
            timestamp=time.time()
        )
    
    def get_operation_stats(self, document_id: str) -> Dict[str, Any]:
        """Get statistics about operations on a document"""
        doc = self.documents.get(document_id)
        if not doc:
            return {}
        
        total_ops = sum(len(delta.operations) for delta in doc.deltas)
        ops_by_type = {}
        ops_by_user = {}
        
        for delta in doc.deltas:
            user_id = delta.user_id
            if user_id not in ops_by_user:
                ops_by_user[user_id] = 0
            ops_by_user[user_id] += len(delta.operations)
            
            for op in delta.operations:
                op_type = op.type.value
                if op_type not in ops_by_type:
                    ops_by_type[op_type] = 0
                ops_by_type[op_type] += 1
        
        return {
            "document_id": document_id,
            "version": doc.version,
            "total_deltas": len(doc.deltas),
            "total_operations": total_ops,
            "operations_by_type": ops_by_type,
            "operations_by_user": ops_by_user,
            "document_length": len(doc.content),
            "created_at": doc.created_at,
            "updated_at": doc.updated_at
        }

class DocumentSync:
    """Document synchronization manager"""
    
    def __init__(self):
        self.ot_engine = OperationalTransform()
        self.cursors: Dict[str, Dict[str, Any]] = {}  # document_id -> user_id -> cursor_info
        self.selections: Dict[str, Dict[str, Any]] = {}  # document_id -> user_id -> selection_info
    
    def update_cursor(self, document_id: str, user_id: str, position: int, 
                     metadata: Dict[str, Any] = None):
        """Update user cursor position"""
        if document_id not in self.cursors:
            self.cursors[document_id] = {}
        
        self.cursors[document_id][user_id] = {
            "position": position,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
    
    def update_selection(self, document_id: str, user_id: str, start: int, 
                        end: int, metadata: Dict[str, Any] = None):
        """Update user selection"""
        if document_id not in self.selections:
            self.selections[document_id] = {}
        
        self.selections[document_id][user_id] = {
            "start": start,
            "end": end,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
    
    def get_document_presence(self, document_id: str) -> Dict[str, Any]:
        """Get all user presence info for a document"""
        return {
            "cursors": self.cursors.get(document_id, {}),
            "selections": self.selections.get(document_id, {}),
            "active_users": list(set(
                list(self.cursors.get(document_id, {}).keys()) +
                list(self.selections.get(document_id, {}).keys())
            ))
        }
    
    def cleanup_stale_presence(self, document_id: str, timeout: int = 300):
        """Remove stale cursor and selection data"""
        current_time = time.time()
        
        # Clean cursors
        if document_id in self.cursors:
            stale_users = [
                user_id for user_id, info in self.cursors[document_id].items()
                if current_time - info["timestamp"] > timeout
            ]
            for user_id in stale_users:
                del self.cursors[document_id][user_id]
        
        # Clean selections
        if document_id in self.selections:
            stale_users = [
                user_id for user_id, info in self.selections[document_id].items()
                if current_time - info["timestamp"] > timeout
            ]
            for user_id in stale_users:
                del self.selections[document_id][user_id]

# Global instances
ot_engine = OperationalTransform()
doc_sync = DocumentSync()