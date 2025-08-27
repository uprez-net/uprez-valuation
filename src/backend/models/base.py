"""
Base model classes with common functionality
"""

from datetime import datetime
from typing import Any, Dict
from sqlalchemy import Column, Integer, DateTime, Boolean, String, JSON
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import as_declarative
from pydantic import BaseModel
import uuid


@as_declarative()
class Base:
    """Base model class with common fields and methods"""
    
    # Generate table names automatically
    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower() + 's'
    
    # Common fields
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Metadata field for flexible data storage
    metadata = Column(JSON, default=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary"""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update model instance from dictionary"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create model instance from dictionary"""
        return cls(**data)


class TimestampMixin:
    """Mixin for timestamp fields"""
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class SoftDeleteMixin:
    """Mixin for soft delete functionality"""
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime, nullable=True)
    
    def soft_delete(self):
        """Mark record as deleted"""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
    
    def restore(self):
        """Restore soft-deleted record"""
        self.is_deleted = False
        self.deleted_at = None


class AuditMixin:
    """Mixin for audit trail functionality"""
    created_by = Column(String, nullable=True)
    updated_by = Column(String, nullable=True)
    version = Column(Integer, default=1, nullable=False)
    
    def increment_version(self, user_id: str = None):
        """Increment version and update audit fields"""
        self.version += 1
        self.updated_by = user_id
        self.updated_at = datetime.utcnow()


class UUIDMixin:
    """Mixin for UUID fields"""
    uuid = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, index=True)


# Pydantic base models for API schemas
class BaseSchema(BaseModel):
    """Base Pydantic schema with common configuration"""
    
    class Config:
        orm_mode = True
        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class TimestampSchema(BaseSchema):
    """Schema with timestamp fields"""
    created_at: datetime
    updated_at: datetime


class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = 1
    size: int = 20
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size


class PaginationResponse(BaseModel):
    """Pagination response metadata"""
    page: int
    size: int
    total: int
    pages: int
    has_next: bool
    has_prev: bool


class SortParams(BaseModel):
    """Sorting parameters"""
    sort_by: str = "id"
    sort_order: str = "asc"  # asc or desc


class FilterParams(BaseModel):
    """Base filter parameters"""
    is_active: bool = True


class BulkOperationResult(BaseModel):
    """Result of bulk operations"""
    success_count: int
    error_count: int
    errors: list = []
    total_processed: int