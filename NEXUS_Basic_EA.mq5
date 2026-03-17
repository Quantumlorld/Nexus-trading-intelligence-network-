//+------------------------------------------------------------------+
//|                                    NEXUS_Basic_EA.mq5 |
//|                                    NEXUS Trading Intelligence Network |
//+------------------------------------------------------------------+
#property copyright "NEXUS Trading Intelligence Network"
#property version   "1.00"
#property strict

int OnInit()
{
   Print("[NEXUS Basic EA] Started");
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   Print("[NEXUS Basic EA] Stopped");
}

void OnTick()
{
   static datetime last_print = 0;
   
   if(TimeCurrent() - last_print >= 30)
   {
      CAccountInfo account;
      string msg = StringFormat("[NEXUS] Account: %d", account.Login());
      Print(msg);
      last_print = TimeCurrent();
   }
}
