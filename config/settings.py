"""
Nexus Trading System - Settings Configuration
Environment-based configuration management
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings from environment variables"""
    
    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    DB_HOST: str = os.getenv("DB_HOST")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME")
    DB_USER: str = os.getenv("DB_USER")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD")
    
    # JWT Configuration
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_DEBUG: bool = os.getenv("API_DEBUG", "false").lower() == "true"
    API_RELOAD: bool = os.getenv("API_RELOAD", "false").lower() == "true"
    
    # Trading Configuration
    MAX_DAILY_LOSS: float = float(os.getenv("MAX_DAILY_LOSS", "9.99"))
    MAX_RISK_PERCENT: float = float(os.getenv("MAX_RISK_PERCENT", "1.0"))
    DEFAULT_SL_POINTS: int = int(os.getenv("DEFAULT_SL_POINTS", "300"))
    DEFAULT_TP_POINTS: int = int(os.getenv("DEFAULT_TP_POINTS", "990"))
    
    # MetaTrader 5 Configuration
    MT5_LOGIN: int = int(os.getenv("MT5_LOGIN", "12345678"))
    MT5_PASSWORD: str = os.getenv("MT5_PASSWORD")
    MT5_SERVER: str = os.getenv("MT5_SERVER", "MetaQuotes-Demo")
    
    # Exchange API Keys
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY")
    BINANCE_SECRET_KEY: str = os.getenv("BINANCE_SECRET_KEY")
    
    # Email Configuration
    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    EMAIL_USERNAME: str = os.getenv("EMAIL_USERNAME")
    EMAIL_PASSWORD: str = os.getenv("EMAIL_PASSWORD")
    
    # AWS S3 Configuration
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME")
    
    # Stripe Configuration
    STRIPE_SECRET_KEY: str = os.getenv("STRIPE_SECRET_KEY")
    STRIPE_WEBHOOK_SECRET: str = os.getenv("STRIPE_WEBHOOK_SECRET")
    
    # Monitoring Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_MONITORING: bool = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
    
    # Security Configuration
    BCRYPT_ROUNDS: int = int(os.getenv("BCRYPT_ROUNDS", "12"))
    SESSION_TIMEOUT_MINUTES: int = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
    MAX_LOGIN_ATTEMPTS: int = int(os.getenv("MAX_LOGIN_ATTEMPTS", "5"))
    ACCOUNT_LOCKOUT_MINUTES: int = int(os.getenv("ACCOUNT_LOCKOUT_MINUTES", "15"))
    
    @classmethod
    def validate(cls) -> bool:
        """Validate critical settings - fail fast if missing"""
        critical_vars = [
            "SECRET_KEY",
            "DATABASE_URL",
            "DB_PASSWORD"
        ]
        
        for var in critical_vars:
            value = getattr(cls, var)
            if not value or value.strip() == "":
                raise RuntimeError(f"CRITICAL: {var} must be set in environment variables")
            
            # Check for placeholder values
            placeholder_patterns = [
                "demo_", "test_", "example_", "change_in_production", 
                "12345678", "abcdef", "your_", "replace_"
            ]
            
            if any(pattern in value.lower() for pattern in placeholder_patterns):
                if var == "SECRET_KEY":
                    raise RuntimeError(f"CRITICAL: {var} cannot use placeholder value: {value}")
                else:
                    print(f"WARNING: {var} appears to use placeholder value: {value}")
        
        # Validate database URL format
        if not cls.DATABASE_URL.startswith(("postgresql://", "postgres://")):
            raise RuntimeError("CRITICAL: DATABASE_URL must be a PostgreSQL connection string")
        
        # Validate secret key strength
        if len(cls.SECRET_KEY) < 32:
            raise RuntimeError("CRITICAL: SECRET_KEY must be at least 32 characters long")
        
        return True
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get complete database URL"""
        return cls.DATABASE_URL
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production"""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"

# Global settings instance
try:
    settings = Settings()
    settings.validate()
except RuntimeError as e:
    print(f"Configuration Error: {e}")
    print("Please update your .env file with proper values.")
    raise SystemExit(1)
