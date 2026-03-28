# Nexus Trading Intelligence Network

> Production-ready, broker-safe trading platform with real MT5 execution via MQL5 WebRequest bridge.

## 🎯 What It Does

- **Real MT5 execution**: Queue trades from UI/backend → MQL5 EA executes in MetaTrader 5 → results flow back
- **Resonance validation**: 3H/6H/9H multi-timeframe signal gating (confidence ≥ 80)
- **Live status dashboard**: Frontend shows MT5 connection, demo progress, trade results
- **Secure by default**: All credentials excluded from git; uses `.env` for secrets

## 🏗️ Architecture

```
Frontend (React/Vite)    Backend (FastAPI)          MT5 (MQL5 EA)
─────────────────► /trade ──────────────► /mt5/bridge/commands (FIFO)
◄────────────────── UI ◄───────────────◄ /mt5/bridge/data (heartbeats + order_result)
```

- **Backend**: `/trade` queues real `place_order` commands (not simulated)
- **MQL5 EA**: `NEXUS_WebRequest_Bridge_Full.mq5` polls commands, executes via `CTrade`, posts `order_result`
- **Bridge endpoints**:
  - `GET /mt5/bridge/commands` — returns one command at a time (queue)
  - `POST /mt5/bridge/data` — receives `account_info` (heartbeat) and `order_result`
  - `GET /mt5/bridge/responses` — inspector/debug view

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- MetaTrader 5 (XM Global or any broker)
- WebRequest enabled in MT5: `Tools → Options → Expert Advisors → Allow WebRequest for listed URLs` and add `http://127.0.0.1:8000`

### 1) Clone & Install

```bash
git clone https://github.com/Quantumlorld/Nexus-trading-intelligence-network-.git
cd Nexus-trading-intelligence-network-
```

```bash
# Backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

### 2) Configure Environment

Copy `.env.example` to `.env` and fill in your secrets (never commit `.env`):

```env
# Database (optional, simulated if not used)
DATABASE_URL=postgresql://...

# MT5 Demo (optional, bridge works without)
MT5_LOGIN=12345678
MT5_PASSWORD=demo_password
MT5_SERVER=MetaQuotes-Demo

# API keys (optional)
BINANCE_API_KEY=...
```

### 3) Run Services

```bash
# Backend
python simple_app.py

# Frontend (in another terminal)
cd frontend && npm run dev
```

Visit:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000/docs

### 4) Deploy MQL5 EA

- Open MetaEditor in your MT5 terminal
- Compile `NEXUS_WebRequest_Bridge_Full.mq5`
- Drag the EA onto any chart or enable AutoTrading
- Watch Experts tab for heartbeat logs

## 🧪 Verify End-to-End

1) Enable trading:
   ```bash
   curl -X POST http://127.0.0.1:8000/admin/enable-trading
   ```

2) Trigger a trade (from UI or API):
   ```bash
   python -c "import requests; r=requests.post('http://127.0.0.1:8000/trade', json={'symbol':'EUR/USD','action':'BUY','quantity':0.01,'order_type':'MARKET'}); print(r.json())"
   ```

3) Check status:
   ```bash
   curl http://127.0.0.1:8000/admin/mt5-status
   curl http://127.0.0.1:8000/mt5/bridge/responses
   ```

You should see:
- Backend: `{"status":"QUEUED","command_id":"CMD_..."}`
- MT5 EA: logs showing `>>> Sending JSON: {"command":"place_order"...}` and `<<< Response: {"command":null}`
- Backend: order_result logged and `trade_count` incremented

## 📁 Project Structure

```
├─ frontend/          # React/Vite UI
├─ simple_app.py      # FastAPI backend
├─ NEXUS_WebRequest_Bridge_Full.mq5  # Production EA
├─ .env.example       # Template for secrets
├─ .gitignore        # Blocks credentials and compiled files
└─ README.md
```

## 🔐 Security

- `.env` is gitignored (never commit secrets)
- Frontend never contains hardcoded passwords
- Compiled MT5 files (`.ex5`) are gitignored
- WebRequest URL must be whitelisted in MT5

## 🛠️ Development

### Adding New Commands

1) Define a command in `simple_app.py` and queue via `bridge_commands.append(cmd)`
2) Extend EA `PollAndExecuteCommands()` to handle the new command type
3) POST results back to `/mt5/bridge/data`

### Resonance Logic

Modify `resonance_validate()` in `simple_app.py` to change:
- Timeframe sources
- Confidence thresholds
- Signal combination rules

## 📊 Monitoring

- `/admin/system-status` — overall health
- `/admin/mt5-status` — bridge heartbeat and trade count
- `/mt5/bridge/responses` — raw bridge messages (debug)

## 🤝 Contributing

1) Fork the repo
2) Create a feature branch
3) Keep credentials out of source
4) Submit a PR

## 📜 License

MIT License — see `LICENSE` file.

---

**Built for production use. Real MT5 execution. Broker-safe by design.**
