"""
MT5 IPC Timeout - COMPREHENSIVE FIX GUIDE
Step-by-step solution to resolve MT5 connection issues
"""

import os
import subprocess
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_mt5_ipc_timeout():
    """Comprehensive fix for MT5 IPC timeout issues"""
    
    print("🔧 MT5 IPC TIMEOUT - COMPREHENSIVE FIX")
    print("=" * 60)
    
    # STEP 1: Kill all MT5 processes
    print("\n1️⃣ Killing all MT5 processes...")
    try:
        subprocess.run(["taskkill", "/F", "/IM", "terminal64.exe"], capture_output=True)
        print("✅ MT5 processes killed")
        time.sleep(3)
    except Exception as e:
        print(f"⚠️ Could not kill MT5: {e}")
    
    # STEP 2: Check MT5 installation paths
    print("\n2️⃣ Checking MT5 installations...")
    paths_to_check = [
        r"C:\Program Files\MetaTrader 5\terminal64.exe",
        r"C:\Program Files\XM Global MT5\terminal64.exe"
    ]
    
    working_paths = []
    for path in paths_to_check:
        if os.path.exists(path):
            working_paths.append(path)
            print(f"✅ Found: {path}")
        else:
            print(f"❌ Missing: {path}")
    
    if not working_paths:
        print("❌ NO MT5 INSTALLATIONS FOUND!")
        return False
    
    # STEP 3: Launch MT5 with proper parameters
    print("\n3️⃣ Launching MT5 with API parameters...")
    mt5_path = working_paths[0]  # Use first available
    
    # Launch MT5 with specific parameters for API access
    cmd = [
        mt5_path,
        "/portable",  # Run in portable mode
        "/skipupdate",  # Skip updates
        "/config:MT5_API_Config"  # Custom config
    ]
    
    try:
        print(f"🚀 Launching: {mt5_path}")
        process = subprocess.Popen(cmd)
        print("✅ MT5 launched, waiting for initialization...")
        time.sleep(15)  # Wait longer for full startup
        
    except Exception as e:
        print(f"❌ Failed to launch MT5: {e}")
        return False
    
    # STEP 4: Test connection with different approaches
    print("\n4️⃣ Testing connection approaches...")
    
    approaches = [
        "Direct initialization",
        "With explicit path", 
        "With extended timeout",
        "With terminal restart"
    ]
    
    for i, approach in enumerate(approaches, 1):
        print(f"\n   Approach {i}: {approach}")
        
        try:
            import MetaTrader5 as mt5
            
            if i == 1:  # Direct
                result = mt5.initialize()
            elif i == 2:  # With path
                result = mt5.initialize(path=mt5_path)
            elif i == 3:  # With timeout
                result = mt5.initialize(timeout=60000)
            elif i == 4:  # Restart terminal
                mt5.shutdown()
                time.sleep(2)
                result = mt5.initialize()
            
            if result:
                print(f"   ✅ SUCCESS! Approach {i} worked!")
                
                # Get terminal info
                terminal_info = mt5.terminal_info()
                print(f"   📊 Terminal: {terminal_info}")
                
                # Try login
                login_result = mt5.login(5047475068, "Z*FiJx0n", "MetaQuotes-Demo")
                if login_result:
                    account = mt5.account_info()
                    print(f"   💰 Account: {account.login} - Balance: ${account.balance}")
                    print("   🎉 MT5 FULLY CONNECTED!")
                    mt5.shutdown()
                    return True
                else:
                    print(f"   ⚠️ Connection OK but login failed: {mt5.last_error()}")
                
                mt5.shutdown()
            else:
                error = mt5.last_error()
                print(f"   ❌ Failed: {error}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    # STEP 5: Alternative - Use XM Global if available
    if len(working_paths) > 1:
        print("\n5️⃣ Trying XM Global as alternative...")
        xm_path = working_paths[1]
        
        try:
            # Kill current MT5
            subprocess.run(["taskkill", "/F", "/IM", "terminal64.exe"], capture_output=True)
            time.sleep(3)
            
            # Launch XM Global
            subprocess.Popen([xm_path])
            time.sleep(15)
            
            # Test XM Global connection
            result = mt5.initialize(path=xm_path)
            if result:
                print("✅ XM Global connected successfully!")
                return True
                
        except Exception as e:
            print(f"❌ XM Global failed: {e}")
    
    print("\n❌ ALL APPROACHES FAILED")
    print("\n🔧 MANUAL FIXES NEEDED:")
    print("1. Open MetaTrader 5 manually")
    print("2. Go to Tools → Options → Expert Advisors")  
    print("3. Enable 'Allow algorithmic trading'")
    print("4. Check 'Allow DLL imports'")
    print("5. Restart MT5 and try again")
    
    return False

if __name__ == "__main__":
    success = fix_mt5_ipc_timeout()
    
    if success:
        print("\n🎉 MT5 CONNECTION SUCCESSFUL!")
        print("🚀 Ready for demo trading!")
    else:
        print("\n📋 NEXT STEPS:")
        print("1. Configure MT5 manually (see above)")
        print("2. Try the MQL5 WebRequest bridge approach")
        print("3. Use simulated trading for immediate results")
