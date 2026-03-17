"""
WORKING MT5 CONNECTION - FINAL SOLUTION
Use the exact method that worked in fix_mt5_ipc.py
"""

import subprocess
import time
import MetaTrader5 as mt5

def connect_working_mt5():
    """Connect using the exact method that worked"""
    
    print("🎯 WORKING MT5 CONNECTION")
    print("=" * 40)
    
    # Step 1: Kill existing processes
    print("1️⃣ Killing existing MT5...")
    subprocess.run(["taskkill", "/F", "/IM", "terminal64.exe"], capture_output=True)
    time.sleep(3)
    
    # Step 2: Launch MT5 normally (no special parameters)
    print("2️⃣ Launching MT5...")
    mt5_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    subprocess.Popen([mt5_path])
    time.sleep(15)  # Wait for full startup
    
    # Step 3: Initialize with EXPLICIT PATH (this was the key!)
    print("3️⃣ Initializing with explicit path...")
    result = mt5.initialize(path=mt5_path)
    
    if result:
        print("✅ MT5 Connected!")
        
        # Step 4: Login to your demo account
        print("4️⃣ Logging into demo account...")
        login_result = mt5.login(5047475068, "Z*FiJx0n", "MetaQuotes-Demo")
        
        if login_result:
            account = mt5.account_info()
            print(f"🎉 SUCCESS! Account: {account.login}")
            print(f"💰 Balance: ${account.balance}")
            print(f"🏢 Server: {account.server}")
            
            # Check trading permissions
            terminal = mt5.terminal_info()
            print(f"🔧 Trading allowed: {terminal.trade_allowed}")
            
            return True
        else:
            print(f"❌ Login failed: {mt5.last_error()}")
            
    else:
        print(f"❌ Connection failed: {mt5.last_error()}")
    
    mt5.shutdown()
    return False

if __name__ == "__main__":
    success = connect_working_mt5()
    
    if success:
        print("\n🚀 READY FOR NEXUS DEMO TRADING!")
        print("📊 Backend can now connect to MT5!")
    else:
        print("\n❌ Connection failed")
