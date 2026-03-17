//+------------------------------------------------------------------+
//|                                    NEXUS_Ultra_Simple.mq5 |
//|                                    NEXUS Trading Intelligence Network |
//+------------------------------------------------------------------+
#property copyright "NEXUS Trading Intelligence Network"
#property version   "1.00"
#property strict

int OnInit()
{
   Print("[NEXUS Ultra Simple] Started");
   Print("[NEXUS Ultra Simple] Connected to account");
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   Print("[NEXUS Ultra Simple] Stopped");
}

void OnTick()
{
   static datetime last_print = 0;
   
   if(TimeCurrent() - last_print >= 30) // Print every 30 seconds
   {
      CAccountInfo account;
      string login_str = IntegerToString(account.Login());
      string balance_str = DoubleToString(account.Balance(), 2);
      string message = "[NEXUS] Account: " + login_str + " Balance: $" + balance_str;
      Print(message);
      last_print = TimeCurrent();
   }
}
