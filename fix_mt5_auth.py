"""
MT5 AUTHORIZATION FIX
Connect to MT5 with proper authorization
"""

import MetaTrader5 as mt5
import time

print("🔧 MT5 AUTHORIZATION FIX")
print("=" * 40)

# Try connecting without login first
print("\n1️⃣ Initialize MT5 without login...")
result = mt5.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe")

if result:
    print("✅ MT5 Initialized successfully!")
    
    # Check if already logged in
    account_info = mt5.account_info()
    if account_info:
        print(f"✅ Already logged in: {account_info.login}")
        print(f"💰 Balance: ${account_info.balance}")
        print(f"🏢 Server: {account_info.server}")
        
        # Test trading capabilities
        terminal_info = mt5.terminal_info()
        print(f"🔧 Trade allowed: {terminal_info.trade_allowed}")
        print(f"🔧 DLLs allowed: {terminal_info.dlls_allowed}")
        
        mt5.shutdown()
        print("🎉 MT5 READY FOR TRADING!")
        
    else:
        print("❌ Not logged in, attempting login...")
        
        # Try login with your credentials
        login_result = mt5.login(5047475068, "Z*FiJx0n", "MetaQuotes-Demo")
        
        if login_result:
            account_info = mt5.account_info()
            print(f"✅ Login successful: {account_info.login}")
            print(f"💰 Balance: ${account_info.balance}")
            mt5.shutdown()
        else:
            print(f"❌ Login failed: {mt5.last_error()}")
            mt5.shutdown()
else:
    print(f"❌ MT5 initialization failed: {mt5.last_error()}")

print("\n🎯 NEXT: Test with Nexus backend...")
