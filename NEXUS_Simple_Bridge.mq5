//+------------------------------------------------------------------+
//|                                    NEXUS_Simple_Bridge.mq5 |
//|                                    NEXUS Trading Intelligence Network |
//+------------------------------------------------------------------+
#property copyright "NEXUS Trading Intelligence Network"
#property version   "1.00"
#property strict

input string InpWebServerURL = "http://localhost:8000";

int OnInit()
{
   Print("[NEXUS Simple Bridge] Started");
   Print("[NEXUS Simple Bridge] URL: ", InpWebServerURL);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   Print("[NEXUS Simple Bridge] Stopped");
}

void OnTick()
{
   static datetime last_send = 0;
   
   if(TimeCurrent() - last_send >= 10) // Send every 10 seconds
   {
      SendAccountInfo();
      last_send = TimeCurrent();
   }
}

void SendAccountInfo()
{
   CAccountInfo account;
   
   string json = "{";
   json += "\"command\":\"account_info\",";
   json += "\"login\":" + IntegerToString(account.Login()) + ",";
   json += "\"balance\":" + DoubleToString(account.Balance(), 2) + ",";
   json += "\"server\":\"" + account.Server() + "\"";
   json += "}";
   
   string url = InpWebServerURL + "/mt5/bridge/data";
   string headers = "Content-Type: application/json";
   uchar data[];
   uchar response[];
   string result_headers;
   
   StringToCharArray(json, data);
   
   // Correct WebRequest signature for MQL5
   WebRequest("POST", url, headers, 5000, data, response, result_headers);
}
