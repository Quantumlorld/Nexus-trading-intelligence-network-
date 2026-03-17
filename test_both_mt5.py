from mt5_integration import MT5Connector

print("🔍 Testing BOTH MT5 Paths...")

# Test 1: MetaTrader 5 from Public Desktop
print("\n1️⃣ Testing MetaTrader 5 (Public Desktop)...")
conn1 = MT5Connector(terminal_path=r"C:\Users\Public\Desktop\MetaTrader 5.lnk")
result1 = conn1.initialize()
print(f"Result: {result1}")
print(f"Status: {conn1.connection_status}")
print(f"Error: {conn1.last_error}")

# Test 2: XM Global from Public Desktop  
print("\n2️⃣ Testing XM Global (Public Desktop)...")
conn2 = MT5Connector(terminal_path=r"C:\Users\Public\Desktop\XM Global MT5.lnk")
result2 = conn2.initialize()
print(f"Result: {result2}")
print(f"Status: {conn2.connection_status}")
print(f"Error: {conn2.last_error}")

# Test 3: XM from OneDrive Desktop
print("\n3️⃣ Testing XM (OneDrive Desktop)...")
conn3 = MT5Connector(terminal_path=r"C:\Users\prime\OneDrive\Desktop\XM.lnk")
result3 = conn3.initialize()
print(f"Result: {result3}")
print(f"Status: {conn3.connection_status}")
print(f"Error: {conn3.last_error}")

# Test 4: Direct XM Global path
print("\n4️⃣ Testing XM Global (Direct Path)...")
conn4 = MT5Connector(terminal_path=r"C:\Program Files\XM Global MT5\terminal64.exe")
result4 = conn4.initialize()
print(f"Result: {result4}")
print(f"Status: {conn4.connection_status}")
print(f"Error: {conn4.last_error}")

# Find working connection
connections = [
    ("MetaTrader 5 (Public)", conn1, result1),
    ("XM Global (Public)", conn2, result2), 
    ("XM (OneDrive)", conn3, result3),
    ("XM Global (Direct)", conn4, result4)
]

print("\n🎯 RESULTS SUMMARY:")
print("=" * 50)
for name, conn, result in connections:
    status = "✅ WORKING" if result else "❌ FAILED"
    print(f"{name}: {status}")
    
    if result:
        print(f"🚀 SUCCESS! Using {name}")
        # Try login with your demo account
        login_result = conn.login(5047475068, "Z*FiJx0n", "MetaQuotes-Demo")
        print(f"Login result: {login_result}")
        
        if login_result:
            account = conn.get_account_info()
            print(f"✅ Account connected: {account}")
        else:
            print(f"❌ Login failed: {conn.last_error}")
        
        conn.shutdown()
        break
    else:
        conn.shutdown()

print("\n🎯 If all failed, MetaTrader 5 needs to be configured for Python API access")
