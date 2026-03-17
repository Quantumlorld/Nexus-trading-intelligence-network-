from mt5_integration import get_mt5_connector

print("🔍 Testing MT5 Connection...")
conn = get_mt5_connector()
print(f"Using terminal: {conn.terminal_path}")

result = conn.initialize()
print(f"Connection result: {result}")
print(f"Status: {conn.connection_status}")
print(f"Error: {conn.last_error}")

if result:
    print("✅ MT5 CONNECTED!")
    
    # Try login
    login_result = conn.login(5047475068, "Z*FiJx0n", "MetaQuotes-Demo")
    print(f"Login result: {login_result}")
    
    if login_result:
        account = conn.get_account_info()
        print(f"✅ Account: {account}")
        
    conn.shutdown()
else:
    print("❌ MT5 Connection failed")
