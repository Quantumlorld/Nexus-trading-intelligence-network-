from mt5_integration import get_mt5_connector

# Test with default MetaTrader 5
connector = get_mt5_connector()
print(f"Using terminal path: {connector.terminal_path}")

# Initialize
result = connector.initialize()
print(f"Initialize result: {result}")
print(f"Connection status: {connector.connection_status}")
print(f"Last error: {connector.last_error}")

if result:
    # Try login
    login_result = connector.login(5047475068, "Z*FiJx0n", "MetaQuotes-Demo")
    print(f"Login result: {login_result}")
    print(f"Connection status: {connector.connection_status}")
    
    # Get account info
    health = connector.mt5_health()
    print(f"Account health: {health}")
    
    connector.shutdown()
