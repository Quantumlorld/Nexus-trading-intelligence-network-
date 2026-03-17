//+------------------------------------------------------------------+
//|                                    NEXUS_Test_Bridge_EA.mq5 |
//|                                    NEXUS Trading Intelligence Network |
//+------------------------------------------------------------------+
#property copyright "NEXUS Trading Intelligence Network"
#property version   "1.00"
#property strict

input string InpWebServerURL = "http://localhost:8000";

int OnInit()
{
   Print("[NEXUS Test Bridge EA] Initialized");
   return(INIT_SUCCEEDED);
}

void OnTick()
{
   static datetime last_check = 0;
   
   if(TimeCurrent() - last_check >= 10)
   {
      SendTestMessage();
      last_check = TimeCurrent();
   }
}

void SendTestMessage()
{
   string url = InpWebServerURL + "/mt5/bridge/data";
   string json = "{\"command\":\"test\",\"message\":\"EA is running\"}";
   
   string headers;
   string result;
   uchar data[];
   StringToCharArray(json, data);
   
   WebRequest("POST", url, headers, 0, data, result, headers);
}
