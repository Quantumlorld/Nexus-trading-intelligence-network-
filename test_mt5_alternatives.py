import MetaTrader5 as mt5
import time
import subprocess
import os

def test_different_approaches():
    print("🔍 Testing different MT5 connection approaches...")
    
    # Approach 1: Launch terminal and wait
    print("\n1️⃣ Testing: Launch terminal + wait + connect")
    terminal_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    
    try:
        # Kill existing terminals
        subprocess.run(["taskkill", "/F", "/IM", "terminal64.exe"], capture_output=True)
        time.sleep(2)
        
        # Launch terminal
        subprocess.Popen([terminal_path])
        time.sleep(10)  # Wait longer for full startup
        
        # Try connection
        result = mt5.initialize(timeout=60000)  # 60 second timeout
        print(f"   Result: {result}")
        print(f"   Error: {mt5.last_error()}")
        
        if result:
            print("   ✅ SUCCESS! Terminal connected!")
            
            # Try login
            login_result = mt5.login(5047475068, "Z*FiJx0n", "MetaQuotes-Demo")
            print(f"   Login result: {login_result}")
            print(f"   Login error: {mt5.last_error()}")
            
            if login_result:
                account = mt5.account_info()
                print(f"   ✅ Account: {account.login} - Balance: ${account.balance}")
                
            mt5.shutdown()
            return True
            
        mt5.shutdown()
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Approach 2: Try different terminal path
    print("\n2️⃣ Testing: Alternative terminal paths")
    paths_to_try = [
        r"C:\Program Files\MetaTrader 5\terminal64.exe",
        r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
        r"C:\Users\prime\AppData\Roaming\MetaQuotes\Terminal\terminal64.exe"
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            print(f"   Found terminal at: {path}")
            try:
                result = mt5.initialize(path=path, timeout=30000)
                print(f"   Connection result: {result}")
                print(f"   Error: {mt5.last_error()}")
                
                if result:
                    print("   ✅ SUCCESS with this path!")
                    mt5.shutdown()
                    return True
                    
            except Exception as e:
                print(f"   Error with this path: {e}")
    
    # Approach 3: Check MT5 version and compatibility
    print("\n3️⃣ Testing: MT5 version check")
    try:
        print(f"   MT5 version: {mt5.__version__}")
        print(f"   MT5 package info: {mt5}")
    except Exception as e:
        print(f"   Version check error: {e}")
    
    print("\n❌ All approaches failed")
    return False

if __name__ == "__main__":
    test_different_approaches()
