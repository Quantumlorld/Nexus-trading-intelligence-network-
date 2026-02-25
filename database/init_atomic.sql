-- Nexus Trading System - Atomic Database Initialization
-- Creates tables for persistent token revocation and atomic risk enforcement

-- Revoked Tokens Table
CREATE TABLE IF NOT EXISTS revoked_tokens (
    id SERIAL PRIMARY KEY,
    token_hash VARCHAR(64) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    revoked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    reason VARCHAR(255)
);

-- Indexes for revoked tokens
CREATE INDEX IF NOT EXISTS idx_revoked_tokens_hash ON revoked_tokens(token_hash);
CREATE INDEX IF NOT EXISTS idx_revoked_tokens_expires ON revoked_tokens(expires_at);

-- User Daily Stats Table for Atomic Risk Enforcement
CREATE TABLE IF NOT EXISTS user_daily_stats (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    trade_count INTEGER DEFAULT 0 NOT NULL,
    daily_loss DECIMAL(15,2) DEFAULT 0.00 NOT NULL,
    daily_pnl DECIMAL(15,2) DEFAULT 0.00 NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    UNIQUE(user_id, date)
);

-- Indexes for user daily stats
CREATE INDEX IF NOT EXISTS idx_user_daily_stats_user_id ON user_daily_stats(user_id);
CREATE INDEX IF NOT EXISTS idx_user_daily_stats_date ON user_daily_stats(date);
CREATE INDEX IF NOT EXISTS idx_user_daily_stats_user_date ON user_daily_stats(user_id, date);

-- User Lockouts Table (enhanced for atomic operations)
CREATE TABLE IF NOT EXISTS user_lockouts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    failed_attempts INTEGER DEFAULT 0 NOT NULL,
    last_attempt_at TIMESTAMP NOT NULL,
    is_locked BOOLEAN DEFAULT FALSE NOT NULL,
    locked_until TIMESTAMP,
    ip_address INET,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Indexes for user lockouts
CREATE INDEX IF NOT EXISTS idx_user_lockouts_user_id ON user_lockouts(user_id);
CREATE INDEX IF NOT EXISTS idx_user_lockouts_locked_until ON user_lockouts(locked_until);

-- Function to clean up expired revoked tokens
CREATE OR REPLACE FUNCTION cleanup_expired_tokens()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM revoked_tokens WHERE expires_at < CURRENT_TIMESTAMP;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to user_lockouts table
CREATE TRIGGER update_user_lockouts_updated_at
    BEFORE UPDATE ON user_lockouts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON revoked_tokens TO nexus_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON user_daily_stats TO nexus_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON user_lockouts TO nexus_user;

-- Grant usage on sequences
-- GRANT USAGE ON SEQUENCE revoked_tokens_id_seq TO nexus_user;
-- GRANT USAGE ON SEQUENCE user_daily_stats_id_seq TO nexus_user;
-- GRANT USAGE ON SEQUENCE user_lockouts_id_seq TO nexus_user;
