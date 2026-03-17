"""
TEST XM GLOBAL TERMINAL
Try XM Global instead of MetaTrader 5
"""

import MetaTrader5 as mt5

print("🔧 TESTING XM GLOBAL TERMINAL")
print("=" * 40)

# Test XM Global terminal
xm_path = r"C:\Program Files\XM Global MT5\terminal64.exe"

print(f"\n🚀 Trying XM Global: {xm_path}")
result = mt5.initialize(path=xm_path)

if result:
    print("✅ XM Global Initialized successfully!")
    
    # Check account info
    account_info = mt5.account_info()
    if account_info:
        print(f"✅ Account: {account_info.login}")
        print(f"💰 Balance: ${account_info.balance}")
        print(f"🏢 Server: {account_info.server}")
        
        # Try login with your demo account
        if account_info.login != 5047475068:
            print("🔄 Logging into your demo account...")
            login_result = mt5.login(5047475068, "Z*FiJx0n", "MetaQuotes-Demo")
            
            if login_result:
                account_info = mt5.account_info()
                print(f"✅ Demo login successful: {account_info.login}")
                print(f"💰 Demo Balance: ${account_info.balance}")
            else:
                print(f"❌ Demo login failed: {mt5.last_error()}")
        
        # Check trading capabilities
        terminal_info = mt5.terminal_info()
        print(f"🔧 Trade allowed: {terminal_info.trade_allowed}")
        
        mt5.shutdown()
        print("🎉 XM GLOBAL READY FOR TRADING!")
        
    else:
        print("❌ No account info available")
        mt5.shutdown()
        
else:
    print(f"❌ XM Global failed: {mt5.last_error()}")

print("\n🎯 If XM Global works, we'll use it for demo trading!")
