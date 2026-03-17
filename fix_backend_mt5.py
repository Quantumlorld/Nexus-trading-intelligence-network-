"""
FIX BACKEND MT5 CONNECTION
Use the working method from our successful test
"""

import subprocess
import time
import MetaTrader5 as mt5

def fix_backend_mt5():
    """Fix backend MT5 connection using working method"""
    
    print("🔧 FIXING BACKEND MT5 CONNECTION")
    print("=" * 50)
    
    # Step 1: Kill existing MT5
    print("1️⃣ Killing existing MT5...")
    subprocess.run(["taskkill", "/F", "/IM", "terminal64.exe"], capture_output=True)
    time.sleep(3)
    
    # Step 2: Launch MT5 normally
    print("2️⃣ Launching MT5...")
    mt5_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    subprocess.Popen([mt5_path])
    time.sleep(15)  # Wait for full startup
    
    # Step 3: Initialize with explicit path (WORKING METHOD!)
    print("3️⃣ Initializing MT5...")
    result = mt5.initialize(path=mt5_path)
    
    if result:
        print("✅ MT5 Connected!")
        
        # Step 4: Login to demo account
        print("4️⃣ Logging into demo account...")
        login_result = mt5.login(5047475068, "Z*FiJx0n", "MetaQuotes-Demo")
        
        if login_result:
            account = mt5.account_info()
            print(f"🎉 SUCCESS! Account: {account.login}")
            print(f"💰 Balance: ${account.balance}")
            print(f"🏢 Server: {account.server}")
            
            # Test trading
            print("5️⃣ Testing trading capabilities...")
            terminal = mt5.terminal_info()
            print(f"🔧 Trading allowed: {terminal.trade_allowed}")
            
            # Keep connection alive for backend
            print("✅ MT5 READY FOR BACKEND!")
            print("🚀 Now try frontend MT5 connection...")
            
            # Keep running
            try:
                while True:
                    time.sleep(10)
                    # Check connection status
                    account = mt5.account_info()
                    if account:
                        print(f"📊 Still connected - Balance: ${account.balance}")
                    else:
                        print("❌ Connection lost")
                        break
            except KeyboardInterrupt:
                print("🛑 Stopping MT5 connection test")
            
            mt5.shutdown()
            return True
        else:
            print(f"❌ Login failed: {mt5.last_error()}")
            
    else:
        print(f"❌ Connection failed: {mt5.last_error()}")
    
    mt5.shutdown()
    return False

if __name__ == "__main__":
    success = fix_backend_mt5()
    
    if success:
        print("\n🎉 BACKEND MT5 CONNECTION FIXED!")
        print("🌐 Now try frontend MT5 connection")
    else:
        print("\n❌ Backend MT5 connection failed")
