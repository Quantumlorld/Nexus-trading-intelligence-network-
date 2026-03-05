# MetaTrader 5 Integration Guide

## 🎯 CURRENT STATUS
- **Demo System**: Simulated broker connection
- **No Real Trading**: Safe for testing
- **Mock Data**: Generated candle data

## 🔧 TO ADD REAL MT5 CONNECTION

### 1. Install MetaTrader 5 Terminal
- Download MT5 from your broker
- Install on Windows machine
- Enable "Allow DLL imports"

### 2. MT5 Python API
```bash
pip install MetaTrader5
```

### 3. Broker Configuration
```python
import MetaTrader5 as mt5

# Initialize MT5
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()

# Login to account
account = 12345678
password = "your_password"
server = "your_broker_server"

authorized = mt5.login(account, password=password, server=server)
if authorized:
    print("Connected to MT5 account #{}".format(account))
else:
    print("failed to connect at account #{}, error code: {}".format(account, mt5.last_error()))
```

### 4. Real Trading Functions
- Place real orders
- Get live market data
- Monitor account balance
- Execute trades automatically

## ⚠️ IMPORTANT WARNINGS
- **REAL MONEY RISK**: Only use demo accounts first
- **BROKER API**: Need broker-specific settings
- **REGULATORY**: Trading regulations apply
- **TESTING**: Always test with demo accounts

## 🎯 NEXT STEPS
1. Get demo MT5 account from broker
2. Install MT5 terminal
3. Test with paper money first
4. Configure broker-specific settings
5. Implement risk management

## 📞 CURRENT SYSTEM FEATURES
- ✅ **Simulation Mode**: Safe testing
- ✅ **Broker Failure Testing**: Resilience features
- ✅ **9H Candles**: Custom timeframes
- ✅ **Real-time Dashboard**: Live monitoring
- ✅ **Trading Controls**: Enable/disable safety

**The current system is designed for SAFE DEMO and TESTING purposes!**
