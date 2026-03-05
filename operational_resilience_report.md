# OPERATIONAL RESILIENCE TEST REPORT
Generated: 2026-03-01T00:24:39.121471

## Emergency Kill Switch
**Success:** False
**Duration:** 0.41s
**Trading State Before:** False
**Trading State After:** False

### System State:
- error: No module named 'email.message'

### Log Entries:
```
[2026-03-01T00:24:38.568413] INFO: === PHASE 1: GLOBAL EMERGENCY KILL SWITCH TEST ===
```
```
[2026-03-01T00:24:38.956819] ERROR: Failed to get trading state: No module named 'email.message'
```
```
[2026-03-01T00:24:38.957823] INFO: Initial trading state: False
```
```
[2026-03-01T00:24:38.975484] ERROR: Emergency kill switch test failed: No module named 'email.message'
```

---

## Broker Connectivity Monitor
**Success:** False
**Duration:** 0.02s
**Trading State Before:** False
**Trading State After:** False

### System State:
- error: No module named 'email.message'

### Log Entries:
```
[2026-03-01T00:24:38.976801] INFO: === PHASE 2: BROKER CONNECTIVITY MONITOR TEST ===
```
```
[2026-03-01T00:24:38.990609] ERROR: Failed to get trading state: No module named 'email.message'
```
```
[2026-03-01T00:24:38.992256] INFO: Initial trading state: False
```
```
[2026-03-01T00:24:39.001094] ERROR: Broker connectivity test failed: No module named 'email.message'
```

---

## Database Health Guard
**Success:** False
**Duration:** 0.02s
**Trading State Before:** False
**Trading State After:** False

### System State:
- error: No module named 'email.message'

### Log Entries:
```
[2026-03-01T00:24:39.001094] INFO: === PHASE 3: DATABASE HEALTH GUARD TEST ===
```
```
[2026-03-01T00:24:39.010975] ERROR: Failed to get trading state: No module named 'email.message'
```
```
[2026-03-01T00:24:39.012526] INFO: Initial trading state: False
```
```
[2026-03-01T00:24:39.022860] ERROR: Database health test failed: No module named 'email.message'
```

---

## Monitoring & Metrics
**Success:** False
**Duration:** 0.08s
**Trading State Before:** False
**Trading State After:** False

### System State:
- error: name 'asyncio' is not defined

### Log Entries:
```
[2026-03-01T00:24:39.023852] INFO: === PHASE 4: MONITORING & METRICS TEST ===
```
```
[2026-03-01T00:24:39.039351] ERROR: Failed to get trading state: No module named 'email.message'
```
```
[2026-03-01T00:24:39.039351] INFO: Initial trading state: False
```
```
[2026-03-01T00:24:39.097409] ERROR: Monitoring metrics test failed: name 'asyncio' is not defined
```

---

## Alert Escalation Logic
**Success:** False
**Duration:** 0.02s
**Trading State Before:** False
**Trading State After:** False

### System State:
- error: No module named 'email.message'

### Log Entries:
```
[2026-03-01T00:24:39.098921] INFO: === PHASE 5: ALERT ESCALATION LOGIC TEST ===
```
```
[2026-03-01T00:24:39.107824] ERROR: Failed to get trading state: No module named 'email.message'
```
```
[2026-03-01T00:24:39.108830] INFO: Initial trading state: False
```
```
[2026-03-01T00:24:39.119473] ERROR: Alert escalation test failed: No module named 'email.message'
```

---
