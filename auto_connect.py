#!/usr/bin/env python3
"""
🚀 NEXUS AUTO-CONNECT SCRIPT
Auto-connects to MT5 and starts demo trading
"""

import requests
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Your MT5 credentials
MT5_ACCOUNT = 103969793
MT5_PASSWORD = "*d8qNgQq"
MT5_SERVER = "MetaQuotes-Demo"

# Backend URL
BACKEND_URL = "http://localhost:8000"

def auto_connect_mt5():
    """Auto-connect to MT5 with your credentials"""
    logger.info("🚀 Starting auto-connection to MT5...")
    
    try:
        # Connect to MT5
        logger.info(f"📡 Connecting to MT5 account {MT5_ACCOUNT}...")
        response = requests.post(
            f"{BACKEND_URL}/admin/mt5-connect",
            params={
                "account": MT5_ACCOUNT,
                "password": MT5_PASSWORD,
                "server": MT5_SERVER
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                logger.info("✅ MT5 connection successful!")
                logger.info(f"📊 Account: {data.get('account')}")
                logger.info(f"💰 Server: {data.get('server')}")
                logger.info(f"🚀 Demo Mode: {data.get('demo_mode')}")
                logger.info(f"📈 Current Phase: {data.get('current_phase')}")
                logger.info(f"🔄 Trade Count: {data.get('trade_count')}")
                return True
            else:
                logger.error(f"❌ MT5 connection failed: {data.get('message')}")
                return False
        else:
            logger.error(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Auto-connection error: {e}")
        return False

def check_status():
    """Check system status"""
    try:
        response = requests.get(f"{BACKEND_URL}/admin/mt5-status")
        if response.status_code == 200:
            data = response.json()
            logger.info("📊 Current Status:")
            logger.info(f"  Connected: {data.get('connected')}")
            logger.info(f"  Demo Mode: {data.get('demo_mode')}")
            logger.info(f"  Trade Count: {data.get('trade_count')}")
            logger.info(f"  Current Phase: {data.get('current_phase')}")
            return data
        else:
            logger.error(f"❌ Status check failed: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"❌ Status check error: {e}")
        return None

def main():
    """Main auto-connect function"""
    logger.info("🎯 NEXUS AUTO-CONNECT STARTING...")
    
    # Check backend is running
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        if response.status_code != 200:
            logger.error("❌ Backend not running! Start backend first.")
            return
        logger.info("✅ Backend is running")
    except:
        logger.error("❌ Cannot reach backend! Start backend first.")
        return
    
    # Auto-connect to MT5
    if auto_connect_mt5():
        logger.info("🎉 Auto-connection successful!")
        
        # Check status
        time.sleep(2)
        status = check_status()
        
        if status and status.get("demo_mode"):
            logger.info("🚀 Demo trading is now ACTIVE!")
            logger.info("📊 Your 500-trade learning journey has begun!")
            logger.info("🌟 The system will now start learning your trading patterns!")
        else:
            logger.error("❌ Demo mode not activated")
    else:
        logger.error("❌ Auto-connection failed")

if __name__ == "__main__":
    main()
