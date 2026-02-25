"""
Nexus Trading System - Revoked Tokens Model
Persistent token revocation for security
"""

from sqlalchemy import Column, Integer, String, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import hashlib

Base = declarative_base()

class RevokedToken(Base):
    """Revoked token storage for persistent token revocation"""
    __tablename__ = "revoked_tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    token_hash = Column(String(64), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False, index=True)
    revoked_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    reason = Column(String(255), nullable=True)
    
    @staticmethod
    def hash_token(token: str) -> str:
        """Generate SHA256 hash of token"""
        return hashlib.sha256(token.encode()).hexdigest()
