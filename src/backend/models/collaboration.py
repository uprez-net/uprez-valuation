"""
Collaboration and project management models
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Enum, JSON
from sqlalchemy.orm import relationship
import enum

from .base import Base, TimestampMixin, SoftDeleteMixin, AuditMixin, UUIDMixin


class ProjectStatus(enum.Enum):
    """Project status enumeration"""
    ACTIVE = "active"
    COMPLETED = "completed"
    ON_HOLD = "on_hold"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"


class ProjectRole(enum.Enum):
    """Project member roles"""
    OWNER = "owner"
    ADMIN = "admin"
    ANALYST = "analyst"
    REVIEWER = "reviewer"
    VIEWER = "viewer"


class ActivityType(enum.Enum):
    """Activity log types"""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    SHARED = "shared"
    COMMENTED = "commented"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPORTED = "exported"
    IMPORTED = "imported"


class Project(Base, TimestampMixin, SoftDeleteMixin, AuditMixin, UUIDMixin):
    """Collaboration project for valuation work"""
    
    __tablename__ = "projects"
    
    # Basic project information
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(Enum(ProjectStatus), default=ProjectStatus.ACTIVE, nullable=False)
    
    # Project details
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Project timeline
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    deadline = Column(DateTime, nullable=True)
    
    # Project settings
    is_public = Column(Boolean, default=False, nullable=False)
    allow_comments = Column(Boolean, default=True, nullable=False)
    allow_external_sharing = Column(Boolean, default=False, nullable=False)
    
    # Progress tracking
    progress_percentage = Column(Integer, default=0, nullable=False)
    last_activity = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Project metadata
    tags = Column(JSON, default=list, nullable=True)  # Array of tags
    priority = Column(String(20), default="medium", nullable=False)  # low, medium, high, critical
    budget = Column(Integer, nullable=True)  # Budget in cents
    
    # Settings
    settings = Column(JSON, default=dict, nullable=True)
    notifications_enabled = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    owner = relationship("User")
    company = relationship("Company")
    members = relationship("ProjectMember", back_populates="project", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="project", cascade="all, delete-orphan")
    activity_logs = relationship("ActivityLog", back_populates="project", cascade="all, delete-orphan")
    
    @property
    def member_count(self) -> int:
        """Get total number of project members"""
        return len(self.members)
    
    @property
    def is_overdue(self) -> bool:
        """Check if project is overdue"""
        return self.deadline and self.deadline < datetime.utcnow() and self.status == ProjectStatus.ACTIVE
    
    def add_member(self, user_id: int, role: ProjectRole = ProjectRole.VIEWER):
        """Add member to project"""
        existing_member = next((m for m in self.members if m.user_id == user_id), None)
        if not existing_member:
            member = ProjectMember(
                project_id=self.id,
                user_id=user_id,
                role=role
            )
            self.members.append(member)
            return member
        return existing_member
    
    def remove_member(self, user_id: int):
        """Remove member from project"""
        self.members = [m for m in self.members if m.user_id != user_id]
    
    def update_progress(self, percentage: int):
        """Update project progress"""
        self.progress_percentage = max(0, min(100, percentage))
        self.last_activity = datetime.utcnow()
    
    def get_member_role(self, user_id: int) -> Optional[ProjectRole]:
        """Get user's role in project"""
        member = next((m for m in self.members if m.user_id == user_id), None)
        return member.role if member else None


class ProjectMember(Base, TimestampMixin, AuditMixin):
    """Project membership and roles"""
    
    __tablename__ = "project_members"
    
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Member details
    role = Column(Enum(ProjectRole), default=ProjectRole.VIEWER, nullable=False)
    invited_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    invited_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    joined_at = Column(DateTime, nullable=True)
    
    # Member status
    is_active = Column(Boolean, default=True, nullable=False)
    last_seen = Column(DateTime, nullable=True)
    
    # Permissions
    can_edit = Column(Boolean, default=False, nullable=False)
    can_comment = Column(Boolean, default=True, nullable=False)
    can_share = Column(Boolean, default=False, nullable=False)
    can_export = Column(Boolean, default=False, nullable=False)
    
    # Notification preferences
    email_notifications = Column(Boolean, default=True, nullable=False)
    in_app_notifications = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    project = relationship("Project", back_populates="members")
    user = relationship("User", back_populates="projects", foreign_keys=[user_id])
    inviter = relationship("User", foreign_keys=[invited_by])
    
    def update_permissions_by_role(self):
        """Update permissions based on role"""
        if self.role == ProjectRole.OWNER:
            self.can_edit = True
            self.can_comment = True
            self.can_share = True
            self.can_export = True
        elif self.role == ProjectRole.ADMIN:
            self.can_edit = True
            self.can_comment = True
            self.can_share = True
            self.can_export = True
        elif self.role == ProjectRole.ANALYST:
            self.can_edit = True
            self.can_comment = True
            self.can_share = False
            self.can_export = True
        elif self.role == ProjectRole.REVIEWER:
            self.can_edit = False
            self.can_comment = True
            self.can_share = False
            self.can_export = True
        else:  # VIEWER
            self.can_edit = False
            self.can_comment = False
            self.can_share = False
            self.can_export = False


class Comment(Base, TimestampMixin, SoftDeleteMixin, AuditMixin):
    """Comments on projects and valuations"""
    
    __tablename__ = "comments"
    
    # Comment relationships
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    valuation_result_id = Column(Integer, ForeignKey("valuation_results.id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    parent_id = Column(Integer, ForeignKey("comments.id"), nullable=True)  # For threaded comments
    
    # Comment content
    content = Column(Text, nullable=False)
    content_type = Column(String(20), default="text", nullable=False)  # text, markdown, html
    
    # Comment metadata
    is_internal = Column(Boolean, default=False, nullable=False)  # Internal team comments
    is_resolved = Column(Boolean, default=False, nullable=False)
    resolved_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    
    # Mentions and tags
    mentions = Column(JSON, default=list, nullable=True)  # Array of mentioned user IDs
    tags = Column(JSON, default=list, nullable=True)  # Array of tags
    
    # Attachments
    attachments = Column(JSON, default=list, nullable=True)  # Array of attachment metadata
    
    # Reactions
    reactions = Column(JSON, default=dict, nullable=True)  # Emoji reactions count
    
    # Relationships
    project = relationship("Project", back_populates="comments")
    user = relationship("User", back_populates="comments")
    resolver = relationship("User", foreign_keys=[resolved_by])
    replies = relationship("Comment", backref="parent", remote_side="Comment.id")
    
    @property
    def reply_count(self) -> int:
        """Get number of replies"""
        return len(self.replies)
    
    def add_reaction(self, emoji: str, user_id: int):
        """Add reaction to comment"""
        if not self.reactions:
            self.reactions = {}
        
        if emoji not in self.reactions:
            self.reactions[emoji] = []
        
        if user_id not in self.reactions[emoji]:
            self.reactions[emoji].append(user_id)
    
    def remove_reaction(self, emoji: str, user_id: int):
        """Remove reaction from comment"""
        if self.reactions and emoji in self.reactions:
            if user_id in self.reactions[emoji]:
                self.reactions[emoji].remove(user_id)
                if not self.reactions[emoji]:
                    del self.reactions[emoji]
    
    def resolve(self, resolver_id: int):
        """Mark comment as resolved"""
        self.is_resolved = True
        self.resolved_by = resolver_id
        self.resolved_at = datetime.utcnow()


class ActivityLog(Base, TimestampMixin):
    """Activity log for tracking user actions"""
    
    __tablename__ = "activity_logs"
    
    # Activity relationships
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    
    # Activity details
    action = Column(Enum(ActivityType), nullable=False)
    resource_type = Column(String(50), nullable=False)  # project, valuation, comment, etc.
    resource_id = Column(Integer, nullable=True)
    
    # Activity content
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Activity metadata
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    session_id = Column(String(100), nullable=True)
    
    # Additional data
    metadata = Column(JSON, default=dict, nullable=True)
    changes = Column(JSON, default=dict, nullable=True)  # Before/after values
    
    # Relationships
    user = relationship("User", back_populates="activity_logs")
    project = relationship("Project", back_populates="activity_logs")
    
    @classmethod
    def log_activity(cls, user_id: int, action: ActivityType, resource_type: str, 
                    title: str, resource_id: int = None, project_id: int = None,
                    description: str = None, metadata: dict = None):
        """Create activity log entry"""
        return cls(
            user_id=user_id,
            project_id=project_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            title=title,
            description=description,
            metadata=metadata or {}
        )


class ProjectTemplate(Base, TimestampMixin, AuditMixin, UUIDMixin):
    """Project templates for quick setup"""
    
    __tablename__ = "project_templates"
    
    # Template details
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(100), nullable=True)  # IPO, M&A, Equity Research, etc.
    
    # Template configuration
    template_config = Column(JSON, default=dict, nullable=False)
    default_settings = Column(JSON, default=dict, nullable=False)
    checklist_items = Column(JSON, default=list, nullable=True)
    
    # Template metadata
    is_public = Column(Boolean, default=False, nullable=False)
    usage_count = Column(Integer, default=0, nullable=False)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Industry specific
    industry = Column(String(100), nullable=True)
    complexity_level = Column(String(20), nullable=True)  # basic, intermediate, advanced
    
    # Relationships
    creator = relationship("User")
    
    def increment_usage(self):
        """Increment template usage counter"""
        self.usage_count += 1