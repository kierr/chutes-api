"""
ORM and Pydantic models for user model aliases.
"""

import re
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, field_validator
from sqlalchemy import Column, String, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import JSONB
from api.database import Base


class ModelAlias(Base):
    __tablename__ = "model_aliases"

    user_id = Column(String, ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True)
    alias = Column(String(64), primary_key=True)
    chute_ids = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


_ALIAS_PATTERN = re.compile(r"^[\x21-\x39\x3B-\x7E]+$")


class ModelAliasCreate(BaseModel):
    alias: str
    chute_ids: list[str]

    @field_validator("alias")
    @classmethod
    def validate_alias(cls, v: str) -> str:
        if not 1 <= len(v) <= 64:
            raise ValueError("alias must be 1-64 characters")
        if not _ALIAS_PATTERN.match(v):
            raise ValueError("alias must be ASCII printable (no spaces or colons)")
        v = v.lower()
        if ":latency" in v or ":throughput" in v:
            raise ValueError("alias must not contain ':latency' or ':throughput'")
        return v

    @field_validator("chute_ids")
    @classmethod
    def validate_chute_ids(cls, v: list[str]) -> list[str]:
        # De-duplicate while preserving order to avoid repeated fallback attempts.
        deduped = list(dict.fromkeys(v))
        if not 1 <= len(deduped) <= 20:
            raise ValueError("chute_ids must have 1-20 items")
        return deduped


class ModelAliasResponse(BaseModel):
    alias: str
    chute_ids: list[str]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}
