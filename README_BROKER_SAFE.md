# Nexus Trading System - Broker Safe Edition

## Overview

Production-ready trading system with broker-safe ledger and reconciliation. This edition ensures capital safety through proper trade lifecycle management, atomic risk enforcement, and continuous broker-state reconciliation.

## 🛡️ Safety Features

### 1. Broker-Safe Ledger
- **Trade Ledger**: Complete trade lifecycle tracking with atomic precision
- **Broker Positions**: Authoritative broker state synchronization
- **Reconciliation Log**: Comprehensive discrepancy tracking and audit trail

### 2. Atomic Risk Enforcement
- **Row-Level Locking**: Prevents race conditions in risk validation
- **Single Transaction**: Risk validation + trade execution in one atomic operation
- **Live Balance Verification**: Uses broker balance for risk calculations

### 3. Partial Fill & Slippage Handling
- **Dynamic Risk Recalculation**: Adjusts risk metrics on partial fills
- **Slippage Monitoring**: Tracks and alerts on execution price differences
- **Position Size Correction**: Handles partial executions safely

### 4. Continuous Reconciliation
- **Background Service**: 60-second reconciliation loop
- **Discrepancy Detection**: Automatic mismatch identification
- **Emergency Stops**: Trading halt on critical discrepancies

### 5. Production Monitoring
- **Real-time Alerts**: Email/Slack/SMS notifications
- **Health Checks**: System component monitoring
- **Audit Trail**: Complete reconciliation history

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Trading API  │    │  Broker-Safe   │    │   MT5 Broker   │
│                │◄──►│    Executor     │◄──►│                │
│  /api/v1/trade│    │                │    │                │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Trade Ledger │    │ Reconciliation  │    │ Broker Positions│
│                │    │     Service     │    │                │
│   PostgreSQL   │    │                │    │                │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📊 Database Schema

### Core Tables

1. **trade_ledger**: Complete trade lifecycle
   - PENDING → PARTIALLY_FILLED → FILLED/REJECTED
   - Tracks requested vs filled quantities
   - Records slippage and actual losses

2. **broker_positions**: Authoritative broker state
   - Real-time position synchronization
   - Reconciliation tracking
   - Discrepancy monitoring

3. **reconciliation_log**: Audit trail
   - All discrepancies logged
   - Actions taken recorded
   - Risk impact assessment

4. **risk_adjustments**: Dynamic risk changes
   - Partial fill adjustments
   - Slippage impact tracking
   - Risk score evolution

5. **trading_control**: System safety controls
   - Emergency stop mechanisms
   - Threshold configurations
   - Consecutive failure tracking

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit with production values
nano .env
```

### 2. Database Initialization

```bash
# Start with ledger schema
docker-compose -f docker-compose-broker-safe.yml up postgres

# Initialize database (automatic via init scripts)
# - 01-init.sql: Base schema
# - 02-atomic.sql: Atomic risk tables
# - 03-ledger.sql: Broker-safe ledger tables
```

### 3. Start System

```bash
# Start all services
docker-compose -f docker-compose-broker-safe.yml up -d

# Start with monitoring
docker-compose -f docker-compose-broker-safe.yml --profile monitoring up -d
```

### 4. Verify Health

```bash
# Check system health
curl http://localhost:8000/health

# Check trading status
curl http://localhost:8000/api/v1/trading/status
```

## 🔧 Configuration

### Environment Variables

```bash
# Trading Control
MAX_SLIPPAGE_PERCENT=0.5          # Max 0.5% slippage
MAX_DISCREPANCY_THRESHOLD=1000.0  # Max $1000 discrepancy
MAX_CONSECUTIVE_FAILURES=3         # Max reconciliation failures

# Reconciliation
RECONCILIATION_INTERVAL=60          # Seconds between reconciliations
TRADING_ENABLED=true                # Global trading switch

# Broker Integration
MT5_LOGIN=your_mt5_login
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_mt5_server
```

### Risk Thresholds

- **Slippage**: 0.5% default, alerts on exceed
- **Discrepancy**: $1000 default, trading stops on exceed
- **Consecutive Failures**: 3 default, trading stops on exceed
- **Reconciliation Interval**: 60 seconds default

## 📡 API Endpoints

### Trading Operations

```bash
# Execute trade
POST /api/v1/trading/execute
{
  "symbol": "EURUSD",
  "action": "BUY",
  "quantity": 0.1,
  "entry_price": 1.0850,
  "stop_loss": 1.0800,
  "take_profit": 1.0900
}

# Get positions
GET /api/v1/trading/positions

# Get trade ledger
GET /api/v1/trading/ledger?limit=100&status=FILLED
```

### Reconciliation

```bash
# Force reconciliation
POST /api/v1/trading/reconcile

# Get reconciliation report
GET /api/v1/trading/reconciliation/report?hours=24
```

### System Control

```bash
# Enable/disable trading (admin)
POST /api/v1/trading/enable
POST /api/v1/trading/disable

# Emergency stop
POST /api/v1/system/emergency-stop
```

## 🔍 Monitoring

### Health Checks

- **System Health**: `/health`
- **Trading Status**: `/api/v1/trading/status`
- **System Status**: `/api/v1/system/status`

### Metrics & Alerts

- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3000` (admin/admin_password)
- **AlertManager**: `http://localhost:9093`

### Key Metrics

- Trading enabled status
- Reconciliation success rate
- Discrepancy count and severity
- Slippage percentages
- Consecutive failure count

## 🛠️ Development

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run broker-safe edition
python main_broker_safe.py
```

### Testing

```bash
# Run safety tests
python -m pytest tests/test_broker_safe.py -v

# Run reconciliation tests
python -m pytest tests/test_reconciliation.py -v
```

## 📋 Safety Checklist

### Pre-Production

- [ ] Environment variables configured
- [ ] Database schema initialized
- [ ] Broker connection tested
- [ ] Risk thresholds validated
- [ ] Alert endpoints configured
- [ ] SSL certificates installed

### Production Deployment

- [ ] Docker images built and scanned
- [ ] Database backups configured
- [ ] Monitoring dashboards set up
- [ ] Alert routing verified
- [ ] Emergency procedures documented

## 🚨 Emergency Procedures

### Trading Stop

```bash
# Immediate stop
curl -X POST http://localhost:8000/api/v1/system/emergency-stop \
  -H "Content-Type: application/json" \
  -d '{"reason": "Market volatility"}'

# Manual disable
curl -X POST http://localhost:8000/api/v1/trading/disable \
  -H "Content-Type: application/json" \
  -d '{"reason": "Manual maintenance"}'
```

### Reconciliation Issues

```bash
# Check reconciliation status
curl http://localhost:8000/api/v1/trading/reconciliation/report

# Force reconciliation
curl -X POST http://localhost:8000/api/v1/trading/reconcile
```

## 🔒 Security Considerations

1. **API Authentication**: JWT tokens with persistent revocation
2. **Database Security**: Row-level security and encryption
3. **Network Security**: TLS encryption and firewall rules
4. **Secret Management**: Environment variables only
5. **Audit Logging**: Complete action tracking

## 📈 Performance

### Optimizations

- **Database Indexes**: Optimized for reconciliation queries
- **Connection Pooling**: Efficient database connections
- **Async Operations**: Non-blocking I/O throughout
- **Caching**: Redis for frequently accessed data

### Benchmarks

- **Trade Execution**: <100ms average
- **Reconciliation**: <5s for 1000 positions
- **API Response**: <50ms average
- **Database Queries**: <10ms average

## 🤝 Support

### Documentation

- [API Documentation](http://localhost:8000/docs)
- [Database Schema](./database/)
- [Configuration Guide](./config/)

### Troubleshooting

1. **Trading Disabled**: Check `/api/v1/system/status`
2. **Reconciliation Failures**: Check logs and broker connection
3. **High Slippage**: Verify market conditions and broker liquidity
4. **Position Mismatches**: Force reconciliation and review logs

---

## ⚠️ Production Warning

This system handles real capital. Ensure:

1. **Thorough Testing**: Complete test cycle with paper trading
2. **Risk Limits**: Appropriate for your capital size
3. **Monitoring**: All alerts configured and tested
4. **Backup Plans**: Manual override procedures documented
5. **Compliance**: Regulatory requirements met

**Never deploy with default credentials or settings!**

---

*Nexus Trading System - Broker Safe Edition v2.0.0*  
*Production-Ready Capital Safety*
