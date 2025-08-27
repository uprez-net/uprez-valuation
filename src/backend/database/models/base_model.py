"""
Base Model Classes
Common model functionality for all database entities
"""
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from sqlalchemy import Column, DateTime, String, Boolean, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import validates
from sqlalchemy.sql import func

from ..base import Base


class TimestampMixin:
    """Mixin for timestamp columns"""
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class SoftDeleteMixin:
    """Mixin for soft delete functionality"""
    
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    
    def soft_delete(self):
        """Mark record as deleted"""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
    
    def restore(self):
        """Restore soft deleted record"""
        self.is_deleted = False
        self.deleted_at = None


class AuditMixin:
    """Mixin for audit trail functionality"""
    
    created_by = Column(UUID(as_uuid=True), nullable=True)
    updated_by = Column(UUID(as_uuid=True), nullable=True)
    version = Column(String(50), default="1.0", nullable=False)
    metadata = Column(JSON, default={}, nullable=False)
    
    def increment_version(self):
        """Increment version number"""
        try:
            major, minor = map(int, self.version.split('.'))
            self.version = f"{major}.{minor + 1}"
        except (ValueError, AttributeError):
            self.version = "1.1"


class BaseModel(Base, TimestampMixin, SoftDeleteMixin, AuditMixin):
    """Base model class with common functionality"""
    
    __abstract__ = True
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    
    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name from class name"""
        # Convert CamelCase to snake_case
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', cls.__name__)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    
    def to_dict(self, exclude_fields: Optional[set] = None) -> Dict[str, Any]:
        """Convert model to dictionary"""
        exclude_fields = exclude_fields or set()
        result = {}
        
        for column in self.__table__.columns:
            if column.name not in exclude_fields:
                value = getattr(self, column.name)
                if isinstance(value, datetime):
                    value = value.isoformat()
                elif isinstance(value, uuid.UUID):
                    value = str(value)
                result[column.name] = value
        
        return result
    
    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update model from dictionary"""
        for key, value in data.items():
            if hasattr(self, key) and key not in ['id', 'created_at']:
                setattr(self, key, value)
    
    @validates('metadata')
    def validate_metadata(self, key, value):
        """Validate metadata is a dictionary"""
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("metadata must be a dictionary")
        return value
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.id})>"


class VersionedModel(BaseModel):
    """Model with versioning support"""
    
    __abstract__ = True
    
    version_number = Column(String(20), default="1.0.0", nullable=False)
    parent_version_id = Column(UUID(as_uuid=True), nullable=True)
    is_current_version = Column(Boolean, default=True, nullable=False)
    change_summary = Column(Text, nullable=True)
    
    def create_new_version(self, change_summary: Optional[str] = None):
        """Create a new version of the model"""
        # Mark current version as not current
        self.is_current_version = False
        
        # Create new version
        new_version = self.__class__()
        for column in self.__table__.columns:
            if column.name not in ['id', 'created_at', 'updated_at', 'version_number', 'parent_version_id']:
                setattr(new_version, column.name, getattr(self, column.name))
        
        # Set version info
        new_version.parent_version_id = self.id
        new_version.change_summary = change_summary
        
        # Increment version number
        try:
            major, minor, patch = map(int, self.version_number.split('.'))
            new_version.version_number = f"{major}.{minor}.{patch + 1}"
        except (ValueError, AttributeError):
            new_version.version_number = "1.0.1"
        
        return new_version


class CacheableModel:
    """Mixin for models that can be cached"""
    
    @property
    def cache_key(self) -> str:
        """Generate cache key for the model"""
        return f"{self.__class__.__name__}:{self.id}"
    
    @property
    def cache_ttl(self) -> int:
        """Cache TTL in seconds"""
        return 3600  # 1 hour default