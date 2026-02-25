-- Nexus Trading System - Database Initialization Script
-- PostgreSQL initialization with proper extensions and indexes

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

-- Trade-related indexes
CREATE INDEX IF NOT EXISTS idx_trades_user_id ON trades(user_id);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_user_timestamp ON trades(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);

-- Signal-related indexes
CREATE INDEX IF NOT EXISTS idx_signals_user_id ON signals(user_id);
CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy_id);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);

-- Performance-related indexes
CREATE INDEX IF NOT EXISTS idx_performance_user_id ON user_performance(user_id);
CREATE INDEX IF NOT EXISTS idx_performance_date ON user_performance(date);
CREATE INDEX IF NOT EXISTS idx_performance_period ON user_performance(period_start, period_end);

-- Adaptive weight indexes
CREATE INDEX IF NOT EXISTS idx_adaptive_weights_strategy ON adaptive_weights(strategy_id);
CREATE INDEX IF NOT EXISTS idx_adaptive_weights_updated ON adaptive_weights(updated_at);
CREATE INDEX IF NOT EXISTS idx_adaptive_weights_asset ON adaptive_weights(asset);

-- System log indexes
CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level);
CREATE INDEX IF NOT EXISTS idx_system_logs_category ON system_logs(category);
CREATE INDEX IF NOT EXISTS idx_system_logs_created_at ON system_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_system_logs_user_id ON system_logs(user_id);

-- Alert system indexes
CREATE INDEX IF NOT EXISTS idx_alerts_user_id ON alerts(user_id);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);

-- Monitoring indexes
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics(metric_type);

-- Create functions for automated cleanup
CREATE OR REPLACE FUNCTION cleanup_old_logs() RETURNS void AS $$
BEGIN
    -- Delete logs older than 90 days
    DELETE FROM system_logs WHERE created_at < NOW() - INTERVAL '90 days';
    
    -- Delete old metrics older than 1 year
    DELETE FROM metrics WHERE timestamp < NOW() - INTERVAL '1 year';
    
    -- Delete resolved alerts older than 6 months
    DELETE FROM alerts WHERE resolved = true AND resolved_at < NOW() - INTERVAL '6 months';
    
    RAISE NOTICE 'Cleanup completed: %', NOW();
END;
$$ LANGUAGE plpgsql;

-- Create trigger for user trade counting
CREATE OR REPLACE FUNCTION update_user_trade_stats() RETURNS trigger AS $$
BEGIN
    -- Update user's daily trade count (would be implemented with proper logic)
    IF TG_OP = 'INSERT' THEN
        -- Logic to track daily trades per user
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for trade P&L calculation
CREATE OR REPLACE FUNCTION calculate_trade_pnl() RETURNS trigger AS $$
BEGIN
    -- Calculate P&L when trade is closed
    IF NEW.exit_time IS NOT NULL AND OLD.exit_time IS NULL THEN
        -- P&L calculation logic would go here
        NEW.pnl = CASE 
            WHEN NEW.action = 'BUY' THEN (NEW.exit_price - NEW.entry_price) * NEW.quantity
            WHEN NEW.action = 'SELL' THEN (NEW.entry_price - NEW.exit_price) * NEW.quantity
            ELSE 0
        END;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers
-- CREATE TRIGGER trigger_update_user_trade_stats
--     AFTER INSERT ON trades
--     FOR EACH ROW
--     EXECUTE FUNCTION update_user_trade_stats();

-- CREATE TRIGGER trigger_calculate_trade_pnl
--     BEFORE UPDATE ON trades
--     FOR EACH ROW
--     EXECUTE FUNCTION calculate_trade_pnl();

-- Create view for user trading statistics
CREATE OR REPLACE VIEW user_trading_stats AS
SELECT 
    u.id as user_id,
    u.username,
    COUNT(t.id) as total_trades,
    COUNT(CASE WHEN t.pnl > 0 THEN 1 END) as winning_trades,
    COUNT(CASE WHEN t.pnl <= 0 THEN 1 END) as losing_trades,
    COALESCE(SUM(t.pnl), 0) as total_pnl,
    COALESCE(AVG(t.pnl), 0) as avg_pnl,
    COALESCE(MAX(t.pnl), 0) as max_win,
    COALESCE(MIN(t.pnl), 0) as max_loss,
    CASE 
        WHEN COUNT(t.id) > 0 THEN ROUND(COUNT(CASE WHEN t.pnl > 0 THEN 1 END) * 100.0 / COUNT(t.id), 2)
        ELSE 0
    END as win_rate
FROM users u
LEFT JOIN trades t ON u.id = t.user_id
GROUP BY u.id, u.username;

-- Create view for daily trading summary
CREATE OR REPLACE VIEW daily_trading_summary AS
SELECT 
    DATE(t.entry_time) as trade_date,
    COUNT(t.id) as total_trades,
    COUNT(DISTINCT t.user_id) as active_traders,
    COALESCE(SUM(t.pnl), 0) as daily_pnl,
    COUNT(CASE WHEN t.pnl > 0 THEN 1 END) as winning_trades,
    COUNT(CASE WHEN t.pnl <= 0 THEN 1 END) as losing_trades,
    CASE 
        WHEN COUNT(t.id) > 0 THEN ROUND(COUNT(CASE WHEN t.pnl > 0 THEN 1 END) * 100.0 / COUNT(t.id), 2)
        ELSE 0
    END as daily_win_rate
FROM trades t
GROUP BY DATE(t.entry_time)
ORDER BY trade_date DESC;

-- Create view for system health metrics
CREATE OR REPLACE VIEW system_health_metrics AS
SELECT 
    'users' as metric,
    COUNT(*) as value,
    COUNT(CASE WHEN is_active = true THEN 1 END) as active_users
FROM users
UNION ALL
SELECT 
    'trades' as metric,
    COUNT(*) as value,
    COUNT(CASE WHEN status = 'FILLED' THEN 1 END) as completed_trades
FROM trades
WHERE DATE(entry_time) = CURRENT_DATE
UNION ALL
SELECT 
    'alerts' as metric,
    COUNT(*) as value,
    COUNT(CASE WHEN resolved = false THEN 1 END) as active_alerts
FROM alerts
WHERE created_at >= CURRENT_DATE - INTERVAL '7 days';

-- Set up automated cleanup job (requires pg_cron extension)
-- SELECT cron.schedule('cleanup-old-logs', '0 2 * * *', 'SELECT cleanup_old_logs();');

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO nexus_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO nexus_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO nexus_user;

-- Initialize with sample data (optional)
-- INSERT INTO users (username, email, hashed_password, role, is_active, created_at)
-- VALUES 
--     ('admin', 'admin@nexus.com', '$2b$12$hashed_password', 'ADMIN', true, NOW()),
--     ('demo', 'demo@nexus.com', '$2b$12$hashed_password', 'USER', true, NOW());

COMMIT;
