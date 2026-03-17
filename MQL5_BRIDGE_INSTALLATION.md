# MQL5 WebRequest Bridge Installation Guide

## Overview
The MQL5 WebRequest Bridge bypasses Python MT5 IPC issues by using an MQL5 service as a bridge between MetaTrader 5 and the Python backend.

## Files Created
- `mql5_bridge/NEXUS_MT5_Bridge.mq5` - MQL5 service that runs in MT5
- `mt5_bridge_client.py` - Python client for communicating with the bridge
- Bridge API endpoints in `simple_app.py`

## Installation Steps

### 1. Install MQL5 Service in MetaTrader 5

1. Open **default MetaTrader 5** (C:\Program Files\MetaTrader 5\terminal64.exe)
2. Press `F4` to open MetaEditor
3. In MetaEditor, go to `File -> Open Data Folder`
4. Navigate to `MQL5/Services/` folder
5. Copy `NEXUS_MT5_Bridge.mq5` to this folder
6. In MetaEditor, open the copied file
7. Press `F7` to compile the service
8. In MT5 terminal, go to `Tools -> Options -> Expert Advisors`
9. Enable "Allow algorithmic trading" and "Allow DLL imports"
10. Go to `Tools -> Services` (or press `Ctrl+M`)
11. Click "Add" and select "NEXUS_MT5_Bridge"
12. Configure the service:
    - WebServerURL: `http://localhost:8000`
    - WebServerPort: `8000`
    - MagicNumber: `123456`
13. Click "OK" to start the service

### 2. Verify Bridge Installation

1. Check the "Experts" tab in MT5 terminal
2. Look for messages like:
   ```
   [NEXUS MT5 Bridge] Service initialized
   [NEXUS MT5 Bridge] Backend URL: http://localhost:8000:8000
   ```
3. The service should show "Running" status

### 3. Test Bridge Connection

1. Start the Nexus backend:
   ```bash
   python simple_app.py
   ```

2. Test bridge connection:
   ```bash
   curl -X POST http://localhost:8000/mt5/bridge/connect
   ```

3. Expected response:
   ```json
   {
     "success": true,
     "message": "Bridge connected successfully",
     "account_info": {
       "login": YOUR_ACCOUNT_NUMBER,
       "server": "MetaQuotes-Demo",
       "balance": 10000.0,
       "equity": 10000.0
     }
   }
   ```

### 4. Update Frontend to Use Bridge

The frontend automatically uses the bridge when available. The MT5 status endpoint will show:
- Connection method: "Bridge (MQL5 WebRequest)"
- Account info from bridge
- Real-time data from MT5

## Bridge Features

### Supported Commands
- `get_account_info` - Get account information
- `get_symbol_info` - Get symbol information (bid/ask, spread, etc.)
- `place_order` - Place buy/sell orders
- `get_positions` - Get open positions

### API Endpoints
- `GET /mt5/bridge/commands` - MQL5 service checks for commands
- `POST /mt5/bridge/data` - MQL5 service sends responses
- `POST /mt5/bridge/connect` - Initialize bridge connection
- `GET /mt5/bridge/status` - Get bridge status

## Troubleshooting

### Service Not Starting
1. Check MT5 "Tools -> Options -> Expert Advisors"
2. Ensure algorithmic trading is enabled
3. Check "Experts" tab for error messages
4. Verify compilation succeeded in MetaEditor

### Connection Issues
1. Ensure Nexus backend is running on port 8000
2. Check firewall settings for localhost connections
3. Verify WebRequest is enabled in MT5 options

### No Data Received
1. Check MQL5 service is running in MT5
2. Look for error messages in "Experts" tab
3. Test bridge status endpoint: `GET /mt5/bridge/status`

## Security Notes

- The bridge only works with localhost connections
- No credentials are stored in MQL5 code
- All communication is local to your machine
- Magic number prevents conflicts with other EAs

## Performance

- Bridge adds ~100ms latency to MT5 operations
- Suitable for demo trading and testing
- For production, consider direct MT5 API if IPC issues are resolved

## Next Steps

1. Install the MQL5 service following the steps above
2. Test bridge connection
3. Verify frontend shows MT5 connected status
4. Test demo trading functionality
5. Clean up repository for GitHub commit
