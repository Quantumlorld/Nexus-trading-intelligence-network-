"""
Nexus Trading System - Database Session Management
SQLAlchemy session setup and database connection management
"""

from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import os
from typing import Generator
import logging

from config.settings import settings
from .models import Base

# Database configuration from environment
DATABASE_URL = settings.get_database_url()

# Create engine with PostgreSQL only
engine = create_engine(
    DATABASE_URL,
    echo=settings.LOG_LEVEL == "DEBUG",
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,  # Recycle connections every hour
    connect_args={
        "connect_timeout": 10,
        "command_timeout": 30,
    }
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine,
    expire_on_commit=False
)

# Create tables
def create_tables():
    """Create all database tables with proper indexing"""
    Base.metadata.create_all(bind=engine)
    
    # Create additional indexes for performance
    with engine.connect() as conn:
        # User-related indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active)"))
        
        # Trade-related indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_user_id ON trades(user_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_user_timestamp ON trades(user_id, timestamp)"))
        
        # Signal-related indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_signals_user_id ON signals(user_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy_id)"))
        
        # Performance-related indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_performance_user_id ON user_performance(user_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_performance_date ON user_performance(date)"))
        
        # Adaptive weight indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_adaptive_weights_strategy ON adaptive_weights(strategy_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_adaptive_weights_updated ON adaptive_weights(updated_at)"))
        
        conn.commit()
    
    logging.info("Database tables and indexes created successfully")

def drop_tables():
    """Drop all database tables (for testing)"""
    Base.metadata.drop_all(bind=engine)
    logging.info("Database tables dropped")

def get_database_session() -> Generator[Session, None, None]:
    """
    Dependency to get database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db() -> Generator[Session, None, None]:
    """Alias for get_database_session for backward compatibility"""
    return get_database_session()

class DatabaseManager:
    """Database connection manager with health monitoring"""
    
    def __init__(self, database_url: str = DATABASE_URL):
        self.database_url = database_url
        self.engine = engine
        self.SessionLocal = SessionLocal
        
    def create_session(self) -> Session:
        """Create a new database session"""
        return self.SessionLocal()
    
    def close_session(self, session: Session):
        """Close database session"""
        session.close()
    
    def health_check(self) -> dict:
        """Comprehensive database health check"""
        try:
            session = self.create_session()
            
            # Basic connectivity test
            result = session.execute(text("SELECT 1"))
            basic_check = result.scalar() == 1
            
            # Table existence check
            tables_to_check = ['users', 'trades', 'signals', 'user_performance']
            table_checks = {}
            
            for table in tables_to_check:
                try:
                    result = session.execute(text(f"SELECT COUNT(*) FROM {table} LIMIT 1"))
                    table_checks[table] = result.scalar() >= 0
                except Exception:
                    table_checks[table] = False
            
            # Index check
            index_check = True
            try:
                result = session.execute(text("""
                    SELECT COUNT(*) FROM information_schema.indexes 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('users', 'trades', 'signals', 'user_performance')
                """))
                index_count = result.scalar()
                index_check = index_count > 0
            except Exception:
                index_check = False
            
            session.close()
            
            return {
                "healthy": basic_check and all(table_checks.values()),
                "basic_connectivity": basic_check,
                "tables": table_checks,
                "indexes": index_check,
                "database_type": "postgresql",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Database health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_connection_info(self) -> dict:
        """Get database connection information"""
        health = self.health_check()
        
        info = {
            "database_url": self.database_url.split('@')[-1] if '@' in self.database_url else self.database_url,  # Hide credentials
            "driver": self.engine.driver,
            "dialect": self.engine.dialect.name,
            "pool_size": getattr(self.engine.pool, 'size', 'N/A') if hasattr(self.engine, 'pool') else 'N/A',
            "health": health,
            "settings": {
                "echo": self.engine.echo,
                "pool_pre_ping": getattr(self.engine.pool, 'pre_ping', False) if hasattr(self.engine, 'pool') else False
            }
        }
        
        return info
    
    def get_performance_stats(self) -> dict:
        """Get database performance statistics"""
        try:
            session = self.create_session()
            
            stats = {}
            
            if "postgresql" in self.database_url:
                # PostgreSQL performance stats
                try:
                    result = session.execute(text("""
                        SELECT 
                            schemaname,
                            tablename,
                            n_tup_ins as inserts,
                            n_tup_upd as updates,
                            n_tup_del as deletes,
                            n_live_tup as live_tuples,
                            n_dead_tup as dead_tuples
                        FROM pg_stat_user_tables
                        WHERE schemaname = 'public'
                    """))
                    
                    table_stats = {}
                    for row in result:
                        table_stats[row.tablename] = {
                            "inserts": row.inserts,
                            "updates": row.updates,
                            "deletes": row.deletes,
                            "live_tuples": row.live_tuples,
                            "dead_tuples": row.dead_tuples
                        }
                    
                    stats["table_stats"] = table_stats
                    
                    # Connection stats
                    result = session.execute(text("SELECT count(*) FROM pg_stat_activity"))
                    stats["active_connections"] = result.scalar()
                    
                except Exception as e:
                    stats["error"] = str(e)
            
            session.close()
            return stats
            
        except Exception as e:
            return {"error": str(e)}

# Global database manager instance
db_manager = DatabaseManager()

# Dependency for FastAPI
def get_database_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency to get database session
    """
    session = db_manager.create_session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        db_manager.close_session(session)

# Database initialization
def init_database():
    """Initialize database with tables and basic data"""
    create_tables()
    logging.info("Database initialized successfully")

# Database cleanup
def cleanup_database():
    """Clean up database resources"""
    try:
        if hasattr(engine, 'dispose'):
            engine.dispose()
        logging.info("Database cleaned up successfully")
    except Exception as e:
        logging.error(f"Database cleanup failed: {e}")

# Database migration utilities
class MigrationManager:
    """Database migration manager"""
    
    def __init__(self, engine):
        self.engine = engine
    
    def run_migration(self, migration_sql: str, migration_name: str):
        """Run a database migration"""
        try:
            with self.engine.connect() as conn:
                conn.execute(migration_sql)
                conn.commit()
            logging.info(f"Migration '{migration_name}' completed successfully")
            return True
        except Exception as e:
            logging.error(f"Migration '{migration_name}' failed: {e}")
            return False
    
    def create_migration_table(self):
        """Create migration tracking table"""
        migration_sql = """
        CREATE TABLE IF NOT EXISTS migrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(255) NOT NULL UNIQUE,
            executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        return self.run_migration(migration_sql, "create_migration_table")
    
    def is_migration_executed(self, migration_name: str) -> bool:
        """Check if migration has been executed"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    "SELECT name FROM migrations WHERE name = ?",
                    (migration_name,)
                )
                return result.fetchone() is not None
        except Exception as e:
            logging.error(f"Error checking migration '{migration_name}': {e}")
            return False
    
    def mark_migration_executed(self, migration_name: str):
        """Mark migration as executed"""
        migration_sql = """
        INSERT INTO migrations (name) VALUES (?)
        """
        return self.run_migration(migration_sql, f"mark_migration_{migration_name}")

# Global migration manager
migration_manager = MigrationManager(engine)
