//+------------------------------------------------------------------+
//|                                    NEXUS_Basic_Test_EA.mq5 |
//|                                    NEXUS Trading Intelligence Network |
//+------------------------------------------------------------------+
#property copyright "NEXUS Trading Intelligence Network"
#property version   "1.00"
#property strict

int OnInit()
{
   Print("[NEXUS Basic Test EA] Started - SUCCESS!");
   return(INIT_SUCCEEDED);
}

void OnTick()
{
   static int counter = 0;
   counter++;
   
   if(counter % 1000 == 0) // Every ~1000 ticks
   {
      Print("[NEXUS Basic Test EA] Running - Tick: ", counter);
   }
}
