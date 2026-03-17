//+------------------------------------------------------------------+
//|                                    NEXUS_Simple_Bridge_EA.mq5 |
//|                                    NEXUS Trading Intelligence Network |
//+------------------------------------------------------------------+
#property copyright "NEXUS Trading Intelligence Network"
#property link      "https://github.com/Quantumlorld/Nexus-trading-intelligence-network"
#property version   "1.00"
#property strict

input string InpWebServerURL = "http://localhost:8000";

int OnInit()
{
   Print("[NEXUS Simple Bridge EA] Initialized");
   Print("[NEXUS Simple Bridge EA] Backend URL: ", InpWebServerURL);
   
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   Print("[NEXUS Simple Bridge EA] Stopped");
}

void OnTick()
{
   static datetime last_check = 0;
   
   if(TimeCurrent() - last_check >= 5)
   {
      SendAccountInfo();
      last_check = TimeCurrent();
   }
}

void SendAccountInfo()
{
   CAccountInfo account;
   
   string json = "{";
   json += "\"command\":\"account_info\",";
   json += "\"login\":" + IntegerToString(account.Login()) + ",";
   json += "\"server\":\"" + account.Server() + "\",";
   json += "\"balance\":" + DoubleToString(account.Balance(), 2);
   json += "}";
   
   SendToBackend(json);
}

void SendToBackend(string json_data)
{
   string url = "127.0.0.1:8000/mt5/bridge/data";
   string headers = "Content-Type: application/json\r\n";
   long timeout = 5000;
   uchar data[];
   uchar result[];
   StringToCharArray(json_data, data);
   
   WebRequest("POST", url, headers, timeout, data, result, headers);
}

void SendTestMessage()
{
   string url = "127.0.0.1:8000/mt5/bridge/data";
   string json = "{\"command\":\"test\",\"message\":\"EA is running\"}";
   
   string headers;
   string result;
   uchar data[];
   StringToCharArray(json, data);
   
   WebRequest("POST", url, headers, 0, data, result, headers);
}
