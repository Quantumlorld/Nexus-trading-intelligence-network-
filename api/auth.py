"""
Nexus Trading System - Authentication Module
JWT authentication, user management, and security utilities
"""

from datetime import datetime, timedelta
from typing import Optional, Union, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import secrets
import logging
import re
from dataclasses import dataclass

from config.settings import settings
from database.session import get_database_session
from database.models import User, UserRole, UserStatus
from database.schemas import TokenData
from .auth_lockout import lockout_manager

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer()

# Token blacklist removed - now using persistent storage

@dataclass
class PasswordValidation:
    """Password validation rules"""
    min_length: int = 8
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digit: bool = True
    require_special: bool = True
    max_length: int = 128

class AuthenticationError(Exception):
    """Custom authentication error"""
    pass

class PasswordValidator:
    """Password validation and security"""
    
    @staticmethod
    def validate_password(password: str) -> tuple[bool, list[str]]:
        """Validate password against security rules"""
        errors = []
        rules = PasswordValidation()
        
        # Length validation
        if len(password) < rules.min_length:
            errors.append(f"Password must be at least {rules.min_length} characters long")
        
        if len(password) > rules.max_length:
            errors.append(f"Password must not exceed {rules.max_length} characters")
        
        # Uppercase validation
        if rules.require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        # Lowercase validation
        if rules.require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        # Digit validation
        if rules.require_digit and not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")
        
        # Special character validation
        if rules.require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        # Common password patterns
        common_patterns = [
            r'123456', r'password', r'qwerty', r'admin', r'letmein',
            r'welcome', r'monkey', r'dragon', r'master', r'sunshine'
        ]
        
        for pattern in common_patterns:
            if re.search(pattern, password, re.IGNORECASE):
                errors.append("Password contains common patterns that are not allowed")
                break
        
        return len(errors) == 0, errors


class PasswordManager:
    """Password hashing and verification"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password with proper rounds"""
        return pwd_context.hash(password, rounds=settings.BCRYPT_ROUNDS)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def create_secure_password() -> str:
        """Generate a secure random password"""
        import string
        import random
        
        # Ensure password meets all requirements
        while True:
            password = ''.join(random.choices(
                string.ascii_letters + string.digits + '!@#$%^&*',
                k=12
            ))
            is_valid, _ = PasswordValidator.validate_password(password)
            if is_valid:
                return password


class TokenManager:
    """JWT token management with persistent revocation"""
    
    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def create_refresh_token(data: Dict[str, Any]) -> str:
        """Create refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str, token_type: str = "access") -> Optional[TokenData]:
        """Verify and decode token with persistent blacklist check"""
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            
            # Check token type
            if payload.get("type") != token_type:
                return None
            
            # Check if token is revoked in database
            token_hash = RevokedToken.hash_token(token)
            from database.session import get_database_session
            
            with next(get_database_session()) as db:
                revoked = db.query(RevokedToken).filter(
                    RevokedToken.token_hash == token_hash,
                    RevokedToken.expires_at > datetime.utcnow()
                ).first()
                
                if revoked:
                    return None
            
            username: str = payload.get("sub")
            if username is None:
                return None
            
            token_data = TokenData(username=username)
            return token_data
            
        except JWTError:
            return None
    
    @staticmethod
    def revoke_token(token: str, reason: str = "") -> bool:
        """Revoke a token by adding to persistent blacklist"""
        try:
            # Decode token to get expiration
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            exp = datetime.fromtimestamp(payload["exp"])
            
            # Add to persistent blacklist
            from database.session import get_database_session
            from database.models import RevokedToken
            
            with next(get_database_session()) as db:
                revoked_token = RevokedToken(
                    token_hash=RevokedToken.hash_token(token),
                    expires_at=exp,
                    reason=reason
                )
                db.add(revoked_token)
                db.commit()
            
            return True
            
        except JWTError:
            return False
    
    @staticmethod
    def cleanup_expired_tokens():
        """Remove expired tokens from persistent blacklist"""
        from database.session import get_database_session
        from database.models import RevokedToken
        
        with next(get_database_session()) as db:
            expired = db.query(RevokedToken).filter(
                RevokedToken.expires_at <= datetime.utcnow()
            ).delete()
            db.commit()
    
    @staticmethod
    def generate_password(length: int = 12) -> str:
        """Generate a random password"""
        return secrets.token_urlsafe(length)


class TokenManager:
    """JWT token management"""
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str) -> Optional[TokenData]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id: int = payload.get("sub")
            username: str = payload.get("username")
            role: str = payload.get("role")
            
            if user_id is None or username is None:
                return None
            
            token_data = TokenData(user_id=user_id, username=username, role=role)
            return token_data
        
        except JWTError:
            return None
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate API key"""
        return f"nx_{secrets.token_urlsafe(32)}"
    
    @staticmethod
    def generate_api_secret() -> str:
        """Generate API secret"""
        return secrets.token_urlsafe(48)


class UserManager:
    """User management utilities"""
    
    @staticmethod
    def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
        """Authenticate user with credentials and lockout enforcement"""
        user = db.query(User).filter(User.username == username).first()
        
        if not user:
            return None
        
        if not user.is_active:
            return None
        
        # Check if user is locked out
        if lockout_manager.is_user_locked(db, user.id):
            logger.warning(f"Login attempt blocked for locked user: {username}")
            return None
        
        # Verify password
        if not PasswordManager.verify_password(password, user.hashed_password):
            # Record failed attempt
            lockout_manager.record_failed_attempt(db, user.id)
            return None
        
        # Successful login - reset failed attempts
        lockout_manager.reset_failed_attempts(db, user.id)
        
        return user
    
    @staticmethod
    def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        db: Session = Depends(get_database_session)
    ) -> User:
        """Get current authenticated user"""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        token_data = TokenManager.verify_token(credentials.credentials, "access")
        
        if token_data is None:
            raise credentials_exception
        
        user = db.query(User).filter(User.username == token_data.username).first()
        
        if user is None:
            raise credentials_exception
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user"
            )
        
        return user
    
    @staticmethod
    def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
        """Get current active user"""
        if not current_user.is_active:
            raise HTTPException(status_code=400, detail="Inactive user")
        return current_user
    
    @staticmethod
    def create_user(db: Session, username: str, email: str, password: str, role: UserRole = UserRole.USER) -> User:
        """Create new user with validation"""
        # Validate password
        is_valid, errors = PasswordValidator.validate_password(password)
        if not is_valid:
            raise ValueError(f"Password validation failed: {'; '.join(errors)}")
        
        # Check if user exists
        if db.query(User).filter(User.username == username).first():
            raise ValueError("Username already exists")
        
        if db.query(User).filter(User.email == email).first():
            raise ValueError("Email already exists")
        
        # Create user
        hashed_password = PasswordManager.hash_password(password)
        db_user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            role=role,
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        return db_user


# Dependency functions
def get_current_user() -> User:
    """Get current authenticated user dependency"""
    return UserManager.get_current_user()


def get_current_active_user() -> User:
    """Get current active user dependency"""
    return UserManager.get_current_active_user()


# Login and token creation functions
def create_user_tokens(user: User) -> Dict[str, str]:
    """Create access and refresh tokens for user"""
    access_data = {"sub": user.username, "user_id": user.id, "role": user.role.value}
    refresh_data = {"sub": user.username, "user_id": user.id}
    
    access_token = TokenManager.create_access_token(access_data)
    refresh_token = TokenManager.create_refresh_token(refresh_data)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


def authenticate_and_create_tokens(db: Session, username: str, password: str) -> Optional[Dict[str, str]]:
    """Authenticate user and create tokens"""
    user = UserManager.authenticate_user(db, username, password)
    
    if not user:
        return None
    
    return create_user_tokens(user)


class SecurityUtils:
    """Security utilities"""
    
    @staticmethod
    def generate_session_id() -> str:
        """Generate a unique session ID"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def validate_password_strength(password: str) -> tuple[bool, str]:
        """Validate password strength"""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter"
        
        if not any(c.islower() for c in password):
            return False, "Password must contain at least one lowercase letter"
        
        if not any(c.isdigit() for c in password):
            return False, "Password must contain at least one digit"
        
        return True, "Password is strong"
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def sanitize_input(input_string: str) -> str:
        """Sanitize user input"""
        import html
        return html.escape(input_string)
    
    @staticmethod
    def log_security_event(event_type: str, user_id: Optional[int] = None, 
                           details: Optional[str] = None, ip_address: Optional[str] = None):
        """Log security events"""
        logging.warning(f"SECURITY EVENT: {event_type} - User: {user_id} - IP: {ip_address} - Details: {details}")


# Authentication dependencies
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security),
                     db: Session = Depends(get_database_session)) -> User:
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        token_data = TokenManager.verify_token(token)
        
        if token_data is None:
            raise credentials_exception
        
        user = UserManager.get_user_by_id(db, token_data.user_id)
        if user is None:
            raise credentials_exception
        
        return user
    
    except JWTError:
        raise credentials_exception


def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    if current_user.status not in [UserStatus.ACTIVE, UserStatus.TRIAL]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is not active"
        )
    return current_user


def get_current_verified_user(current_user: User = Depends(get_current_active_user)) -> User:
    """Get current verified user"""
    if not UserManager.is_user_authorized(current_user, UserRole.VERIFIED):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    return current_user


def get_current_elite_user(current_user: User = Depends(get_current_active_user)) -> User:
    """Get current elite user"""
    if not UserManager.is_user_authorized(current_user, UserRole.ELITE):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    return current_user


def get_current_admin_user(current_user: User = Depends(get_current_active_user)) -> User:
    """Get current admin user"""
    if not UserManager.is_user_authorized(current_user, UserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


# API key authentication
def get_user_from_api_key(api_key: str, db: Session = Depends(get_database_session)) -> Optional[User]:
    """Get user from API key"""
    user = UserManager.get_user_by_api_key(db, api_key)
    if not user:
        return None
    
    if user.status not in [UserStatus.ACTIVE, UserStatus.TRIAL]:
        return None
    
    return user


# Rate limiting
class RateLimiter:
    """Simple rate limiter for API endpoints"""
    
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, key: str, limit: int, window: int = 60) -> bool:
        """Check if request is allowed"""
        now = datetime.utcnow()
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Remove old requests
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if (now - req_time).total_seconds() < window
        ]
        
        # Check limit
        if len(self.requests[key]) >= limit:
            return False
        
        # Add current request
        self.requests[key].append(now)
        return True


# Global rate limiter
rate_limiter = RateLimiter()


# Rate limiting decorator
def rate_limit(limit: int, window: int = 60):
    """Rate limiting decorator"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get client IP or user ID for rate limiting
            key = args[0] if args else "anonymous"
            
            if not rate_limiter.is_allowed(key, limit, window):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Security middleware
class SecurityMiddleware:
    """Security middleware for API"""
    
    def __init__(self, app):
        self.app = app
    
    def __call__(self, environ, start_response):
        # Add security headers
        def custom_start_response(status, headers, exc_info=None):
            headers.append(('X-Content-Type-Options', 'nosniff'))
            headers.append(('X-Frame-Options', 'DENY'))
            headers.append(('X-XSS-Protection', '1; mode=block'))
            headers.append(('Strict-Transport-Security', 'max-age=31536000; includeSubDomains'))
            return start_response(status, headers, exc_info)
        
        return self.app(environ, custom_start_response)
