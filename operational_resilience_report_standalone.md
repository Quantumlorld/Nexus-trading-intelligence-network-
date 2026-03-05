# OPERATIONAL RESILIENCE TEST REPORT
Generated: 2026-03-01T00:25:26.647317

## Emergency Kill Switch
**Success:** True
**Duration:** 0.01s
**Trading State Before:** True
**Trading State After:** True

### System State:
- trading_enabled: True
- metrics_counters: {'trades_total': 24, 'trades_successful': 18, 'trades_failed': 6, 'trades_timeout': 3, 'ledger_update_retries': 1, 'broker_failures': 4, 'db_failures': 4, 'reconciliation_jobs': 0}
- metrics_gauges: {'trading_enabled': 1.0, 'broker_connected': 0.0, 'db_connected': 0.0, 'reconciliation_queue_size': 0.0, 'active_positions': 0.0}

### Log Entries:
```
[2026-03-01T00:25:26.594567] INFO: Trade 1 correctly rejected: Trading is currently disabled by system control
```
```
[2026-03-01T00:25:26.594567] INFO: Trade attempt 2...
```
```
[2026-03-01T00:25:26.594567] INFO: Trade 2 result: {'success': False, 'error': 'Trading is currently disabled by system control', 'trade_uuid': 'test-kill-switch-1-1772324726', 'trading_enabled': False}
```
```
[2026-03-01T00:25:26.594567] INFO: Trade 2 correctly rejected: Trading is currently disabled by system control
```
```
[2026-03-01T00:25:26.596087] INFO: Trade attempt 3...
```
```
[2026-03-01T00:25:26.596087] INFO: Trade 3 result: {'success': False, 'error': 'Trading is currently disabled by system control', 'trade_uuid': 'test-kill-switch-2-1772324726', 'trading_enabled': False}
```
```
[2026-03-01T00:25:26.597105] INFO: Trade 3 correctly rejected: Trading is currently disabled by system control
```
```
[2026-03-01T00:25:26.597105] INFO: Re-enabling trading...
```
```
[2026-03-01T00:25:26.597622] INFO: Enable trading result: True
```
```
[2026-03-01T00:25:26.598634] INFO: Final trading state: True
```

---

## Broker Connectivity Monitor
**Success:** False
**Duration:** 0.01s
**Trading State Before:** True
**Trading State After:** True

### System State:
- trading_enabled: True
- broker_failures: 3
- broker_connected: 1.0

### Log Entries:
```
[2026-03-01T00:25:26.600643] WARNING: Simulating broker failure 1...
```
```
[2026-03-01T00:25:26.601155] INFO: Trading state after failure 1: True
```
```
[2026-03-01T00:25:26.601155] WARNING: Simulating broker failure 2...
```
```
[2026-03-01T00:25:26.601155] INFO: Trading state after failure 2: True
```
```
[2026-03-01T00:25:26.602781] WARNING: Simulating broker failure 3...
```
```
[2026-03-01T00:25:26.603422] INFO: Trading state after failure 3: True
```
```
[2026-03-01T00:25:26.603422] INFO: Trading state after broker failures: True
```
```
[2026-03-01T00:25:26.604433] INFO: Simulating broker recovery...
```
```
[2026-03-01T00:25:26.605444] INFO: Re-enabling trading after broker recovery...
```
```
[2026-03-01T00:25:26.606434] INFO: Final trading state: True
```

---

## Database Health Guard
**Success:** False
**Duration:** 0.01s
**Trading State Before:** True
**Trading State After:** True

### System State:
- trading_enabled: True
- db_failures: 3
- db_connected: 1.0

### Log Entries:
```
[2026-03-01T00:25:26.608434] WARNING: Simulating DB failure 1...
```
```
[2026-03-01T00:25:26.609438] INFO: Trading state after DB failure 1: True
```
```
[2026-03-01T00:25:26.610438] WARNING: Simulating DB failure 2...
```
```
[2026-03-01T00:25:26.610438] INFO: Trading state after DB failure 2: True
```
```
[2026-03-01T00:25:26.611434] WARNING: Simulating DB failure 3...
```
```
[2026-03-01T00:25:26.611434] INFO: Trading state after DB failure 3: True
```
```
[2026-03-01T00:25:26.612436] INFO: Trading state after DB failures: True
```
```
[2026-03-01T00:25:26.612436] INFO: Simulating database recovery...
```
```
[2026-03-01T00:25:26.613067] INFO: Re-enabling trading after DB recovery...
```
```
[2026-03-01T00:25:26.614258] INFO: Final trading state: True
```

---

## Monitoring & Metrics
**Success:** True
**Duration:** 0.02s
**Trading State Before:** True
**Trading State After:** True

### System State:
- trading_enabled: True
- metrics_summary: {'counters': {'trades_total': 24, 'trades_successful': 18, 'trades_failed': 6, 'trades_timeout': 3, 'ledger_update_retries': 1, 'broker_failures': 4, 'db_failures': 4, 'reconciliation_jobs': 0}, 'gauges': {'trading_enabled': 1.0, 'broker_connected': 0.0, 'db_connected': 0.0, 'reconciliation_queue_size': 0.0, 'active_positions': 0.0}}

### Log Entries:
```
[2026-03-01T00:25:26.630991] INFO: Trade 20: Normal execution...
```
```
[2026-03-01T00:25:26.630991] INFO: Simulating broker latency metrics...
```
```
[2026-03-01T00:25:26.633161] INFO: Simulating DB latency metrics...
```
```
[2026-03-01T00:25:26.636037] INFO: Generating Prometheus metrics output...
```
```
[2026-03-01T00:25:26.636037] INFO: === KEY METRICS SUMMARY ===
```
```
[2026-03-01T00:25:26.636037] INFO: Total trades: 23
```
```
[2026-03-01T00:25:26.636037] INFO: Successful trades: 18
```
```
[2026-03-01T00:25:26.637461] INFO: Failed trades: 5
```
```
[2026-03-01T00:25:26.637461] INFO: Timeout trades: 2
```
```
[2026-03-01T00:25:26.637461] INFO: Ledger retries: 1
```

---

## Alert Escalation Logic
**Success:** True
**Duration:** 0.01s
**Trading State Before:** True
**Trading State After:** True

### System State:
- trading_enabled: True
- alert_severity_tested: ['CRITICAL', 'HIGH', 'WARNING']
- broker_failures: 4
- db_failures: 4

### Log Entries:
```
[2026-03-01T00:25:26.642473] WARNING: HIGH alert logged - trading should continue
```
```
[2026-03-01T00:25:26.642473] INFO: Trading state after HIGH alert: True
```
```
[2026-03-01T00:25:26.643512] WARNING: Testing WARNING alert - High slippage detected...
```
```
[2026-03-01T00:25:26.643512] WARNING: WARNING alert logged - trading should continue
```
```
[2026-03-01T00:25:26.643512] INFO: Trading state after WARNING alert: True
```
```
[2026-03-01T00:25:26.644514] WARNING: Testing broker disconnect scenario...
```
```
[2026-03-01T00:25:26.645205] WARNING: Broker disconnect alert sent
```
```
[2026-03-01T00:25:26.645205] WARNING: Testing DB failure scenario...
```
```
[2026-03-01T00:25:26.646216] WARNING: DB failure alert sent
```
```
[2026-03-01T00:25:26.646216] INFO: Final trading state: True
```

---
