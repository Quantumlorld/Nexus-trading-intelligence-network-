"""
Nexus Trading System - Authentication Lockout System
Failed login attempt tracking and account lockout enforcement
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from sqlalchemy.orm import Session
from database.models import User, UserLockout
from database.session import get_database_session

logger = logging.getLogger(__name__)

class LoginLockoutManager:
    """Manages failed login attempts and account lockouts"""
    
    def __init__(self, max_attempts: int = 5, lockout_minutes: int = 15):
        self.max_attempts = max_attempts
        self.lockout_minutes = lockout_minutes
        self.logger = logging.getLogger(__name__)
    
    def record_failed_attempt(self, db: Session, user_id: int, ip_address: str = None) -> bool:
        """Record a failed login attempt with atomic transaction"""
        try:
            # Begin transaction
            # Get user lockout record with row lock
            lockout = db.query(UserLockout).filter(
                UserLockout.user_id == user_id
            ).with_for_update().first()
            
            if not lockout:
                # Create new lockout record
                lockout = UserLockout(
                    user_id=user_id,
                    failed_attempts=1,
                    last_attempt_at=datetime.utcnow(),
                    ip_address=ip_address
                )
                db.add(lockout)
            else:
                # Update existing record
                lockout.failed_attempts += 1
                lockout.last_attempt_at = datetime.utcnow()
                if ip_address:
                    lockout.ip_address = ip_address
            
            # Check if user should be locked out
            if lockout.failed_attempts >= self.max_attempts:
                lockout.locked_until = datetime.utcnow() + timedelta(minutes=self.lockout_minutes)
                lockout.is_locked = True
                self.logger.warning(f"User {user_id} locked out due to {lockout.failed_attempts} failed attempts")
            
            # Commit transaction - atomic operation
            db.commit()
            
            return lockout.is_locked
            
        except Exception as e:
            self.logger.error(f"Error recording failed attempt: {e}")
            db.rollback()
            return False
    
    def is_user_locked(self, db: Session, user_id: int) -> bool:
        """Check if user is currently locked out"""
        try:
            lockout = db.query(UserLockout).filter(UserLockout.user_id == user_id).first()
            
            if not lockout:
                return False
            
            # Check if lockout has expired
            if lockout.is_locked and lockout.locked_until:
                if datetime.utcnow() > lockout.locked_until:
                    # Lockout expired, reset it
                    lockout.is_locked = False
                    lockout.failed_attempts = 0
                    lockout.locked_until = None
                    db.commit()
                    return False
            
            return lockout.is_locked
            
        except Exception as e:
            self.logger.error(f"Error checking lockout status: {e}")
            return False
    
    def reset_failed_attempts(self, db: Session, user_id: int):
        """Reset failed attempts on successful login"""
        try:
            lockout = db.query(UserLockout).filter(UserLockout.user_id == user_id).first()
            
            if lockout:
                lockout.failed_attempts = 0
                lockout.is_locked = False
                lockout.locked_until = None
                db.commit()
                self.logger.info(f"Reset failed attempts for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Error resetting failed attempts: {e}")
    
    def get_lockout_info(self, db: Session, user_id: int) -> Optional[Dict]:
        """Get lockout information for a user"""
        try:
            lockout = db.query(UserLockout).filter(UserLockout.user_id == user_id).first()
            
            if not lockout:
                return None
            
            return {
                "user_id": lockout.user_id,
                "failed_attempts": lockout.failed_attempts,
                "last_attempt_at": lockout.last_attempt_at.isoformat() if lockout.last_attempt_at else None,
                "is_locked": lockout.is_locked,
                "locked_until": lockout.locked_until.isoformat() if lockout.locked_until else None,
                "ip_address": lockout.ip_address
            }
            
        except Exception as e:
            self.logger.error(f"Error getting lockout info: {e}")
            return None

# Global lockout manager instance
lockout_manager = LoginLockoutManager()
