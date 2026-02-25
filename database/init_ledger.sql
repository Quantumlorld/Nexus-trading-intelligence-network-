-- Nexus Trading System - Ledger Database Initialization
-- Production-ready schema for broker-safe trading

-- Trade Ledger Table
CREATE TABLE IF NOT EXISTS trade_ledger (
    id SERIAL PRIMARY KEY,
    trade_uuid VARCHAR(36) UNIQUE NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL,
    order_type VARCHAR(10) NOT NULL,
    requested_quantity DECIMAL(15,8) NOT NULL,
    filled_quantity DECIMAL(15,8) DEFAULT 0.0 NOT NULL,
    entry_price DECIMAL(15,8) NOT NULL,
    execution_price DECIMAL(15,8),
    avg_execution_price DECIMAL(15,8),
    stop_loss DECIMAL(15,8),
    take_profit DECIMAL(15,8),
    slippage DECIMAL(10,6) DEFAULT 0.0 NOT NULL,
    potential_loss DECIMAL(15,2) NOT NULL,
    actual_loss DECIMAL(15,2),
    status VARCHAR(20) DEFAULT 'pending' NOT NULL,
    broker_order_id VARCHAR(100),
    broker_position_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    submitted_at TIMESTAMP,
    executed_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    metadata JSONB,
    error_message TEXT
);

-- Indexes for trade_ledger
CREATE INDEX IF NOT EXISTS idx_trade_ledger_user_id ON trade_ledger(user_id);
CREATE INDEX IF NOT EXISTS idx_trade_ledger_status ON trade_ledger(status);
CREATE INDEX IF NOT EXISTS idx_trade_ledger_symbol ON trade_ledger(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_ledger_created_at ON trade_ledger(created_at);
CREATE INDEX IF NOT EXISTS idx_trade_ledger_user_status ON trade_ledger(user_id, status);
CREATE INDEX IF NOT EXISTS idx_trade_ledger_symbol_status ON trade_ledger(symbol, status);
CREATE INDEX IF NOT EXISTS idx_trade_ledger_uuid ON trade_ledger(trade_uuid);

-- Broker Positions Table
CREATE TABLE IF NOT EXISTS broker_positions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(15,8) NOT NULL,
    avg_price DECIMAL(15,8) NOT NULL,
    current_price DECIMAL(15,8),
    unrealized_pnl DECIMAL(15,2) DEFAULT 0.0 NOT NULL,
    margin_used DECIMAL(15,2) DEFAULT 0.0 NOT NULL,
    margin_free DECIMAL(15,2) NOT NULL,
    last_reconciled TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    reconciliation_count INTEGER DEFAULT 0 NOT NULL,
    last_discrepancy TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    broker_position_id VARCHAR(100),
    broker_account VARCHAR(50)
);

-- Indexes for broker_positions
CREATE INDEX IF NOT EXISTS idx_broker_positions_user_id ON broker_positions(user_id);
CREATE INDEX IF NOT EXISTS idx_broker_positions_symbol ON broker_positions(symbol);
CREATE INDEX IF NOT EXISTS idx_broker_positions_user_symbol ON broker_positions(user_id, symbol);
CREATE INDEX IF NOT EXISTS idx_broker_positions_updated ON broker_positions(updated_at);

-- Reconciliation Log Table
CREATE TABLE IF NOT EXISTS reconciliation_log (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    ledger_quantity DECIMAL(15,8) NOT NULL,
    broker_quantity DECIMAL(15,8) NOT NULL,
    quantity_discrepancy DECIMAL(15,8) NOT NULL,
    ledger_price DECIMAL(15,8),
    broker_price DECIMAL(15,8),
    price_discrepancy DECIMAL(15,8),
    action VARCHAR(50) NOT NULL,
    action_details TEXT,
    risk_impact VARCHAR(20) NOT NULL,
    trading_stopped BOOLEAN DEFAULT FALSE NOT NULL,
    reconciliation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    resolved_at TIMESTAMP,
    broker_data JSONB,
    ledger_data JSONB,
    error_details TEXT
);

-- Indexes for reconciliation_log
CREATE INDEX IF NOT EXISTS idx_reconciliation_log_user_id ON reconciliation_log(user_id);
CREATE INDEX IF NOT EXISTS idx_reconciliation_log_symbol ON reconciliation_log(symbol);
CREATE INDEX IF NOT EXISTS idx_reconciliation_log_time ON reconciliation_log(reconciliation_time);
CREATE INDEX IF NOT EXISTS idx_reconciliation_log_action ON reconciliation_log(action);
CREATE INDEX IF NOT EXISTS idx_reconciliation_log_risk ON reconciliation_log(risk_impact);
CREATE INDEX IF NOT EXISTS idx_reconciliation_log_user_time ON reconciliation_log(user_id, reconciliation_time);

-- Risk Adjustments Table
CREATE TABLE IF NOT EXISTS risk_adjustments (
    id SERIAL PRIMARY KEY,
    trade_ledger_id INTEGER NOT NULL REFERENCES trade_ledger(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    adjustment_type VARCHAR(50) NOT NULL,
    original_quantity DECIMAL(15,8) NOT NULL,
    adjusted_quantity DECIMAL(15,8) NOT NULL,
    original_loss DECIMAL(15,2) NOT NULL,
    adjusted_loss DECIMAL(15,2) NOT NULL,
    daily_loss_before DECIMAL(15,2) NOT NULL,
    daily_loss_after DECIMAL(15,2) NOT NULL,
    risk_score_before DECIMAL(10,2) NOT NULL,
    risk_score_after DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    reason TEXT,
    metadata JSONB
);

-- Indexes for risk_adjustments
CREATE INDEX IF NOT EXISTS idx_risk_adjustments_trade_id ON risk_adjustments(trade_ledger_id);
CREATE INDEX IF NOT EXISTS idx_risk_adjustments_user_id ON risk_adjustments(user_id);
CREATE INDEX IF NOT EXISTS idx_risk_adjustments_type ON risk_adjustments(adjustment_type);
CREATE INDEX IF NOT EXISTS idx_risk_adjustments_created_at ON risk_adjustments(created_at);

-- Trading Control Table
CREATE TABLE IF NOT EXISTS trading_control (
    id SERIAL PRIMARY KEY,
    trading_enabled BOOLEAN DEFAULT TRUE NOT NULL,
    reconciliation_enabled BOOLEAN DEFAULT TRUE NOT NULL,
    stop_reason VARCHAR(100),
    stop_threshold_exceeded BOOLEAN DEFAULT FALSE NOT NULL,
    consecutive_failures INTEGER DEFAULT 0 NOT NULL,
    max_discrepancy_threshold DECIMAL(15,2) DEFAULT 1000.0 NOT NULL,
    max_consecutive_failures INTEGER DEFAULT 3 NOT NULL,
    slippage_threshold DECIMAL(10,6) DEFAULT 0.5 NOT NULL,
    stopped_at TIMESTAMP,
    last_check TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    stop_details JSONB
);

-- Indexes for trading_control
CREATE INDEX IF NOT EXISTS idx_trading_control_enabled ON trading_control(trading_enabled);
CREATE INDEX IF NOT EXISTS idx_trading_control_updated ON trading_control(updated_at);

-- Trigger to automatically update updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers to tables with updated_at columns
CREATE TRIGGER update_trade_ledger_updated_at
    BEFORE UPDATE ON trade_ledger
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_broker_positions_updated_at
    BEFORE UPDATE ON broker_positions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_trading_control_updated_at
    BEFORE UPDATE ON trading_control
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to clean up old reconciliation logs
CREATE OR REPLACE FUNCTION cleanup_old_reconciliation_logs(days_to_keep INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM reconciliation_log 
    WHERE reconciliation_time < CURRENT_TIMESTAMP - INTERVAL '%s days' % days_to_keep;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get reconciliation statistics
CREATE OR REPLACE FUNCTION get_reconciliation_stats(
    user_id_param INTEGER DEFAULT NULL,
    hours_back INTEGER DEFAULT 24
)
RETURNS TABLE(
    total_discrepancies BIGINT,
    high_risk_discrepancies BIGINT,
    medium_risk_discrepancies BIGINT,
    low_risk_discrepancies BIGINT,
    trading_stopped_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT as total_discrepancies,
        COUNT(*) FILTER (WHERE risk_impact = 'HIGH')::BIGINT as high_risk_discrepancies,
        COUNT(*) FILTER (WHERE risk_impact = 'MEDIUM')::BIGINT as medium_risk_discrepancies,
        COUNT(*) FILTER (WHERE risk_impact = 'LOW')::BIGINT as low_risk_discrepancies,
        COUNT(*) FILTER (WHERE trading_stopped = TRUE)::BIGINT as trading_stopped_count
    FROM reconciliation_log
    WHERE (user_id_param IS NULL OR user_id = user_id_param)
    AND reconciliation_time >= CURRENT_TIMESTAMP - INTERVAL '%s hours' % hours_back;
END;
$$ LANGUAGE plpgsql;

-- Function to get user position summary
CREATE OR REPLACE FUNCTION get_user_position_summary(user_id_param INTEGER)
RETURNS TABLE(
    symbol VARCHAR(20),
    quantity DECIMAL(15,8),
    avg_price DECIMAL(15,8),
    current_price DECIMAL(15,8),
    unrealized_pnl DECIMAL(15,2),
    last_reconciled TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        bp.symbol,
        bp.quantity,
        bp.avg_price,
        bp.current_price,
        bp.unrealized_pnl,
        bp.last_reconciled
    FROM broker_positions bp
    WHERE bp.user_id = user_id_param
    AND bp.quantity != 0
    ORDER BY bp.symbol;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON trade_ledger TO nexus_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON broker_positions TO nexus_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON reconciliation_log TO nexus_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON risk_adjustments TO nexus_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON trading_control TO nexus_user;

-- Grant usage on sequences
-- GRANT USAGE ON SEQUENCE trade_ledger_id_seq TO nexus_user;
-- GRANT USAGE ON SEQUENCE broker_positions_id_seq TO nexus_user;
-- GRANT USAGE ON SEQUENCE reconciliation_log_id_seq TO nexus_user;
-- GRANT USAGE ON SEQUENCE risk_adjustments_id_seq TO nexus_user;
-- GRANT USAGE ON SEQUENCE trading_control_id_seq TO nexus_user;

-- Grant execute permissions on functions
-- GRANT EXECUTE ON FUNCTION update_updated_at_column() TO nexus_user;
-- GRANT EXECUTE ON FUNCTION cleanup_old_reconciliation_logs(INTEGER) TO nexus_user;
-- GRANT EXECUTE ON FUNCTION get_reconciliation_stats(INTEGER, INTEGER) TO nexus_user;
-- GRANT EXECUTE ON FUNCTION get_user_position_summary(INTEGER) TO nexus_user;
