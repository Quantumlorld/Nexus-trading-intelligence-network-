import MetaTrader5 as mt5

# Test basic initialization
ok = mt5.initialize(timeout=120000)
print('initialize=', ok, 'err=', mt5.last_error())

if ok:
    ai = mt5.account_info()
    print('account_info=', ai)
    mt5.shutdown()
