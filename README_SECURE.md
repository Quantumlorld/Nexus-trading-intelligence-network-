# Nexus Trading System - Secure Deployment Guide

## Overview
This guide covers the secure deployment of the Nexus Trading System with all security hardening and risk enforcement features implemented.

## Security Features Implemented

### ✅ Phase 1: Secret & Auth Hardening
- **Environment Variables**: All secrets moved to `.env` file
- **JWT Security**: 15-minute access tokens with refresh tokens
- **Token Revocation**: Blacklist system for compromised tokens
- **Password Security**: Strong validation with bcrypt (12 rounds)
- **No Hardcoded Secrets**: All credentials externalized

### ✅ Phase 2: Database Migration
- **PostgreSQL**: Production-ready database with connection pooling
- **Indexing Strategy**: Optimized indexes for performance
- **Health Monitoring**: Comprehensive database health checks
- **Migration Support**: Alembic-ready for schema migrations

### ✅ Phase 3: Risk Engine Enforcement
- **Daily Loss Cap**: Strict $9.99 maximum daily loss per user
- **Position Limits**: 1% maximum risk per trade
- **Timeframe Limits**: 
  - 9H: Max 2 trades/day
  - 6H: Max 2 trades/day  
  - 3H: Max 1 trade/day
- **Dynamic SL/TP**: Automatic stop loss and take profit management
- **Duplicate Prevention**: Blocks duplicate signals within 1 hour

### ✅ Phase 4: Execution Safety
- **Emergency Kill Switch**: Global trading disable capability
- **Per-User Controls**: Individual trading disable flags
- **Comprehensive Logging**: Every trade decision logged
- **Error Handling**: API failure retries and network drop handling
- **Order Validation**: Multi-layer order validation system

### ✅ Phase 5: Docker Deployment
- **Container Security**: Non-root user, minimal attack surface
- **Health Checks**: Comprehensive health monitoring
- **Volume Management**: Proper data persistence
- **Network Isolation**: Secure network configuration

## Quick Start

### Prerequisites
- Docker & Docker Compose
- PostgreSQL 15+
- Python 3.11+

### 1. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your secure values
nano .env
```

### 2. Critical Security Settings
Update these values in `.env`:
```bash
# Change this immediately!
SECRET_KEY=your_super_secret_unique_key_here

# Database credentials
DB_PASSWORD=your_secure_database_password

# JWT settings
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7

# Trading limits
MAX_DAILY_LOSS=9.99
MAX_RISK_PERCENT=1.0
```

### 3. Deploy with Docker
```bash
# Start all services
docker-compose up --build

# Check health
curl http://localhost:8000/health
```

### 4. Verify Security
```bash
# Check system status
curl http://localhost:8000/status

# Test authentication
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"your_password"}'
```

## Security Configuration

### Authentication
- **Token Expiry**: 15 minutes (access), 7 days (refresh)
- **Password Requirements**: 8+ chars, uppercase, lowercase, digit, special
- **Failed Login Lockout**: 5 attempts, 15 minute lockout
- **Token Blacklisting**: Immediate revocation capability

### Risk Management
- **Hard Limits**: Enforced at database and application level
- **Real-time Monitoring**: Continuous risk assessment
- **Emergency Controls**: Instant trading disable capability
- **Audit Trail**: Complete logging of all risk decisions

### Database Security
- **Connection Pooling**: 10 base connections, 20 max overflow
- **Index Optimization**: Performance and security indexes
- **Health Monitoring**: Real-time database health checks
- **Data Encryption**: Encrypted connections recommended

## Monitoring & Alerts

### Health Endpoints
- `/health` - System health status
- `/status` - Detailed system information
- `/api/v1/metrics` - Performance metrics

### Risk Monitoring
- Daily loss tracking per user
- Real-time position monitoring
- Emergency stop alerts
- Duplicate trade detection

### Logging
- Structured JSON logging
- Security event logging
- Trade execution logging
- Error tracking and alerting

## Production Deployment

### Security Checklist
- [ ] Change default SECRET_KEY
- [ ] Set strong database passwords
- [ ] Configure proper CORS origins
- [ ] Enable SSL/TLS termination
- [ ] Set up backup strategy
- [ ] Configure monitoring alerts
- [ ] Test emergency procedures

### Docker Production
```bash
# Production profile with monitoring
docker-compose --profile monitoring --profile production up --build

# Scale backend if needed
docker-compose up --scale nexus-backend=3
```

### Environment Variables
```bash
# Production environment
ENVIRONMENT=production
API_DEBUG=false
LOG_LEVEL=WARNING

# Security
BCRYPT_ROUNDS=12
SESSION_TIMEOUT_MINUTES=30
MAX_LOGIN_ATTEMPTS=5
```

## Emergency Procedures

### Stop All Trading
```bash
curl -X POST http://localhost:8000/admin/emergency-stop \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -d '{"reason":"Market volatility"}'
```

### Resume Trading
```bash
curl -X POST http://localhost:8000/admin/emergency-resume \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

### Check System Status
```bash
curl http://localhost:8000/status | jq .
```

## Troubleshooting

### Common Issues

#### Database Connection Failed
```bash
# Check PostgreSQL status
docker-compose logs postgres

# Verify connection string
echo $DATABASE_URL
```

#### Authentication Issues
```bash
# Check JWT settings
grep -E "(SECRET_KEY|ACCESS_TOKEN_EXPIRE)" .env

# Test token refresh
curl -X POST http://localhost:8000/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token":"YOUR_REFRESH_TOKEN"}'
```

#### Risk Engine Blocking Trades
```bash
# Check user risk status
curl http://localhost:8000/api/v1/users/{user_id}/risk-status \
  -H "Authorization: Bearer YOUR_TOKEN"

# Check emergency stop status
curl http://localhost:8000/status | jq .risk_engine
```

## Security Best Practices

### Regular Maintenance
1. **Rotate Secrets**: Monthly SECRET_KEY rotation
2. **Update Dependencies**: Keep packages updated
3. **Backup Database**: Daily automated backups
4. **Review Logs**: Security log analysis
5. **Test Recovery**: Emergency procedure testing

### Monitoring Setup
1. **Prometheus**: Metrics collection
2. **Grafana**: Visualization dashboards
3. **Alertmanager**: Alert routing
4. **Log Aggregation**: Centralized logging

### Access Control
1. **Principle of Least Privilege**: Minimal required permissions
2. **Multi-factor Authentication**: For admin access
3. **IP Whitelisting**: Restrict admin access
4. **Session Management**: Proper timeout handling

## Support

### Security Issues
Report security vulnerabilities to the security team immediately.

### Documentation
- API Documentation: `http://localhost:8000/docs`
- System Status: `http://localhost:8000/status`
- Health Check: `http://localhost:8000/health`

### Emergency Contact
- System Administrator: admin@nexus.com
- Security Team: security@nexus.com

---

**⚠️ WARNING**: This system handles real financial data. Ensure all security measures are properly configured before production deployment.
