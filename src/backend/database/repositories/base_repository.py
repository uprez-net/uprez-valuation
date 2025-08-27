"""
Base Repository Pattern
Generic repository with common CRUD operations
"""
import uuid
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
from sqlalchemy import and_, or_, desc, asc, func, text
from sqlalchemy.orm import Session, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.exc import IntegrityError, NoResultFound

from ..models.base_model import BaseModel

ModelType = TypeVar("ModelType", bound=BaseModel)
CreateSchemaType = TypeVar("CreateSchemaType")
UpdateSchemaType = TypeVar("UpdateSchemaType")


class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Base repository class with common CRUD operations"""
    
    def __init__(self, model: Type[ModelType]):
        self.model = model
    
    # Synchronous methods
    def get(self, db: Session, id: Union[uuid.UUID, str]) -> Optional[ModelType]:
        """Get a single record by ID"""
        try:
            return db.query(self.model).filter(self.model.id == id).first()
        except Exception:
            return None
    
    def get_multi(
        self, 
        db: Session, 
        *, 
        skip: int = 0, 
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        order_direction: str = "asc"
    ) -> List[ModelType]:
        """Get multiple records with filtering and pagination"""
        query = db.query(self.model)
        
        # Apply filters
        if filters:
            query = self._apply_filters(query, filters)
        
        # Apply ordering
        if order_by:
            order_field = getattr(self.model, order_by, None)
            if order_field:
                if order_direction.lower() == "desc":
                    query = query.order_by(desc(order_field))
                else:
                    query = query.order_by(asc(order_field))
        
        # Apply pagination
        query = query.offset(skip).limit(limit)
        
        return query.all()
    
    def create(self, db: Session, *, obj_in: CreateSchemaType) -> ModelType:
        """Create a new record"""
        if isinstance(obj_in, dict):
            obj_data = obj_in
        else:
            obj_data = obj_in.dict(exclude_unset=True)
        
        db_obj = self.model(**obj_data)
        db.add(db_obj)
        try:
            db.commit()
            db.refresh(db_obj)
            return db_obj
        except IntegrityError as e:
            db.rollback()
            raise ValueError(f"Integrity constraint violation: {str(e)}")
    
    def update(
        self, 
        db: Session, 
        *, 
        db_obj: ModelType, 
        obj_in: Union[UpdateSchemaType, Dict[str, Any]]
    ) -> ModelType:
        """Update an existing record"""
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        
        try:
            db.commit()
            db.refresh(db_obj)
            return db_obj
        except IntegrityError as e:
            db.rollback()
            raise ValueError(f"Integrity constraint violation: {str(e)}")
    
    def delete(self, db: Session, *, id: Union[uuid.UUID, str]) -> Optional[ModelType]:
        """Delete a record"""
        db_obj = self.get(db, id=id)
        if db_obj:
            db.delete(db_obj)
            db.commit()
            return db_obj
        return None
    
    def soft_delete(self, db: Session, *, id: Union[uuid.UUID, str]) -> Optional[ModelType]:
        """Soft delete a record"""
        db_obj = self.get(db, id=id)
        if db_obj and hasattr(db_obj, 'soft_delete'):
            db_obj.soft_delete()
            db.commit()
            db.refresh(db_obj)
            return db_obj
        return None
    
    def count(self, db: Session, *, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count records with optional filters"""
        query = db.query(func.count(self.model.id))
        
        if filters:
            query = self._apply_filters(query, filters)
        
        return query.scalar()
    
    def exists(self, db: Session, *, id: Union[uuid.UUID, str]) -> bool:
        """Check if record exists"""
        return db.query(self.model.id).filter(self.model.id == id).first() is not None
    
    def search(
        self, 
        db: Session, 
        *, 
        search_term: str, 
        search_fields: List[str],
        skip: int = 0,
        limit: int = 100
    ) -> List[ModelType]:
        """Full-text search across specified fields"""
        query = db.query(self.model)
        
        # Build search conditions
        search_conditions = []
        for field in search_fields:
            if hasattr(self.model, field):
                field_obj = getattr(self.model, field)
                search_conditions.append(field_obj.ilike(f"%{search_term}%"))
        
        if search_conditions:
            query = query.filter(or_(*search_conditions))
        
        return query.offset(skip).limit(limit).all()
    
    # Async methods
    async def get_async(self, db: AsyncSession, id: Union[uuid.UUID, str]) -> Optional[ModelType]:
        """Get a single record by ID (async)"""
        try:
            result = await db.execute(select(self.model).where(self.model.id == id))
            return result.scalar_one_or_none()
        except Exception:
            return None
    
    async def get_multi_async(
        self, 
        db: AsyncSession, 
        *, 
        skip: int = 0, 
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        order_direction: str = "asc"
    ) -> List[ModelType]:
        """Get multiple records with filtering and pagination (async)"""
        query = select(self.model)
        
        # Apply filters
        if filters:
            query = self._apply_filters_async(query, filters)
        
        # Apply ordering
        if order_by:
            order_field = getattr(self.model, order_by, None)
            if order_field:
                if order_direction.lower() == "desc":
                    query = query.order_by(desc(order_field))
                else:
                    query = query.order_by(asc(order_field))
        
        # Apply pagination
        query = query.offset(skip).limit(limit)
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def create_async(self, db: AsyncSession, *, obj_in: CreateSchemaType) -> ModelType:
        """Create a new record (async)"""
        if isinstance(obj_in, dict):
            obj_data = obj_in
        else:
            obj_data = obj_in.dict(exclude_unset=True)
        
        db_obj = self.model(**obj_data)
        db.add(db_obj)
        try:
            await db.commit()
            await db.refresh(db_obj)
            return db_obj
        except IntegrityError as e:
            await db.rollback()
            raise ValueError(f"Integrity constraint violation: {str(e)}")
    
    async def count_async(self, db: AsyncSession, *, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count records with optional filters (async)"""
        query = select(func.count(self.model.id))
        
        if filters:
            query = self._apply_filters_async(query, filters)
        
        result = await db.execute(query)
        return result.scalar()
    
    def _apply_filters(self, query: Query, filters: Dict[str, Any]) -> Query:
        """Apply filters to query"""
        for field, value in filters.items():
            if hasattr(self.model, field):
                field_obj = getattr(self.model, field)
                if isinstance(value, dict):
                    # Handle operators like {'gte': 100, 'lte': 200}
                    for operator, operand in value.items():
                        if operator == 'gte':
                            query = query.filter(field_obj >= operand)
                        elif operator == 'lte':
                            query = query.filter(field_obj <= operand)
                        elif operator == 'gt':
                            query = query.filter(field_obj > operand)
                        elif operator == 'lt':
                            query = query.filter(field_obj < operand)
                        elif operator == 'ne':
                            query = query.filter(field_obj != operand)
                        elif operator == 'in':
                            query = query.filter(field_obj.in_(operand))
                        elif operator == 'like':
                            query = query.filter(field_obj.like(operand))
                        elif operator == 'ilike':
                            query = query.filter(field_obj.ilike(operand))
                elif isinstance(value, list):
                    query = query.filter(field_obj.in_(value))
                else:
                    query = query.filter(field_obj == value)
        
        return query
    
    def _apply_filters_async(self, query, filters: Dict[str, Any]):
        """Apply filters to async query"""
        for field, value in filters.items():
            if hasattr(self.model, field):
                field_obj = getattr(self.model, field)
                if isinstance(value, dict):
                    # Handle operators
                    for operator, operand in value.items():
                        if operator == 'gte':
                            query = query.where(field_obj >= operand)
                        elif operator == 'lte':
                            query = query.where(field_obj <= operand)
                        elif operator == 'gt':
                            query = query.where(field_obj > operand)
                        elif operator == 'lt':
                            query = query.where(field_obj < operand)
                        elif operator == 'ne':
                            query = query.where(field_obj != operand)
                        elif operator == 'in':
                            query = query.where(field_obj.in_(operand))
                        elif operator == 'like':
                            query = query.where(field_obj.like(operand))
                        elif operator == 'ilike':
                            query = query.where(field_obj.ilike(operand))
                elif isinstance(value, list):
                    query = query.where(field_obj.in_(value))
                else:
                    query = query.where(field_obj == value)
        
        return query


class CacheableRepository(BaseRepository[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Repository with caching capabilities"""
    
    def __init__(self, model: Type[ModelType], cache_client=None):
        super().__init__(model)
        self.cache_client = cache_client
    
    def get_cached(self, db: Session, id: Union[uuid.UUID, str]) -> Optional[ModelType]:
        """Get record with caching"""
        if not self.cache_client:
            return self.get(db, id)
        
        cache_key = f"{self.model.__name__}:{id}"
        
        # Try to get from cache first
        cached_obj = self.cache_client.get(cache_key)
        if cached_obj:
            return cached_obj
        
        # Get from database
        db_obj = self.get(db, id)
        if db_obj:
            # Cache the result
            ttl = getattr(db_obj, 'cache_ttl', 3600)
            self.cache_client.setex(cache_key, ttl, db_obj)
        
        return db_obj
    
    def invalidate_cache(self, id: Union[uuid.UUID, str]):
        """Invalidate cache for specific record"""
        if self.cache_client:
            cache_key = f"{self.model.__name__}:{id}"
            self.cache_client.delete(cache_key)