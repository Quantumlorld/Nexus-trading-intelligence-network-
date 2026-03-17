//+------------------------------------------------------------------+
//|                                           NEXUS_MT5_Bridge_EA.mq5  |
//|                                    NEXUS Trading Intelligence Network |
//|                                               MQL5 WebRequest Bridge EA |
//+------------------------------------------------------------------+
#property copyright "NEXUS Trading Intelligence Network"
#property link      "https://github.com/Quantumlorld/Nexus-trading-intelligence-network"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

input string InpWebServerURL = "http://localhost:8000";
input int    InpWebServerPort = 8000;
input long   InpMagicNumber = 123456;

CTrade trade;

int OnInit()
{
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetMarginMode();
   
   Print("[NEXUS MT5 Bridge EA] Initialized");
   Print("[NEXUS MT5 Bridge EA] Backend URL: ", InpWebServerURL);
   
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   Print("[NEXUS MT5 Bridge EA] Stopped");
}

void OnTick()
{
   static datetime last_check = 0;
   
   if(TimeCurrent() - last_check >= 5)
   {
      CheckBackendCommands();
      last_check = TimeCurrent();
   }
}

void CheckBackendCommands()
{
   string url = InpWebServerURL + "/mt5/bridge/commands";
   string result;
   string headers;
   long timeout = 3000;
   
   if(WebRequest("GET", url, "", headers, timeout, result, 0, headers) == 200)
   {
      ProcessCommands(result);
   }
}

void ProcessCommands(string json_data)
{
   if(StringFind(json_data, "\"command\":\"get_account_info\"") >= 0)
   {
      SendAccountInfo();
   }
   else if(StringFind(json_data, "\"command\":\"get_symbol_info\"") >= 0)
   {
      string symbol = ExtractSymbol(json_data);
      if(symbol != "")
         SendSymbolInfo(symbol);
   }
   else if(StringFind(json_data, "\"command\":\"place_order\"") >= 0)
   {
      ProcessOrderCommand(json_data);
   }
   else if(StringFind(json_data, "\"command\":\"get_positions\"") >= 0)
   {
      SendPositions();
   }
}

void SendAccountInfo()
{
   CAccountInfo account;
   
   string json = "{";
   json += "\"command\":\"account_info\",";
   json += "\"login\":" + IntegerToString(account.Login()) + ",";
   json += "\"server\":\"" + account.Server() + "\",";
   json += "\"balance\":" + DoubleToString(account.Balance(), 2) + ",";
   json += "\"equity\":" + DoubleToString(account.Equity(), 2) + ",";
   json += "\"currency\":\"" + account.Currency() + "\"";
   json += "}";
   
   SendToBackend(json);
}

void SendSymbolInfo(string symbol)
{
   MqlTick tick;
   
   string json = "{";
   json += "\"command\":\"symbol_info\",";
   json += "\"symbol\":\"" + symbol + "\"";
   
   if(SymbolInfoTick(symbol, tick))
   {
      json += ",\"bid\":" + DoubleToString(tick.bid, 5);
      json += ",\"ask\":" + DoubleToString(tick.ask, 5);
   }
   
   json += "}";
   
   SendToBackend(json);
}

void ProcessOrderCommand(string json_data)
{
   string symbol = ExtractStringValue(json_data, "symbol");
   double volume = ExtractDoubleValue(json_data, "volume");
   int order_type = ExtractIntValue(json_data, "order_type");
   
   if(symbol == "" || volume <= 0)
   {
      SendError("Invalid order parameters");
      return;
   }
   
   bool result = false;
   
   if(order_type == 0)
   {
      result = trade.Buy(volume, symbol);
   }
   else if(order_type == 1)
   {
      result = trade.Sell(volume, symbol);
   }
   
   string response = "{";
   response += "\"command\":\"order_result\",";
   response += "\"symbol\":\"" + symbol + "\",";
   response += "\"result\":" + (result ? "true" : "false");
   response += "}";
   
   SendToBackend(response);
}

void SendPositions()
{
   string json = "{";
   json += "\"command\":\"positions\",";
   json += "\"positions\":[";
   
   int total = PositionsTotal();
   for(int i = 0; i < total; i++)
   {
      if(PositionGetSymbol(i))
      {
         if(i > 0) json += ",";
         
         json += "{";
         json += "\"ticket\":" + IntegerToString(PositionGetInteger(POSITION_TICKET));
         json += ",\"symbol\":\"" + PositionGetString(POSITION_SYMBOL) + "\"";
         json += ",\"volume\":" + DoubleToString(PositionGetDouble(POSITION_VOLUME), 2);
         json += "}";
      }
   }
   
   json += "]}";
   
   SendToBackend(json);
}

void SendToBackend(string json_data)
{
   string url = InpWebServerURL + "/mt5/bridge/data";
   string headers = "Content-Type: application/json\r\n";
   long timeout = 5000;
   
   WebRequest("POST", url, headers, timeout, json_data, 0, headers);
}

void SendError(string error_msg)
{
   string json = "{";
   json += "\"command\":\"error\",";
   json += "\"error\":\"" + error_msg + "\"";
   json += "}";
   
   SendToBackend(json);
}

string ExtractSymbol(string json_data)
{
   return ExtractStringValue(json_data, "symbol");
}

string ExtractStringValue(string json_data, string key)
{
   string search_key = "\"" + key + "\":\"";
   int start = StringFind(json_data, search_key);
   if(start >= 0)
   {
      start += StringLen(search_key);
      int end = StringFind(json_data, "\"", start);
      if(end > start)
      {
         return StringSubstr(json_data, start, end - start);
      }
   }
   return "";
}

double ExtractDoubleValue(string json_data, string key)
{
   string search_key = "\"" + key + "\":";
   int start = StringFind(json_data, search_key);
   if(start >= 0)
   {
      start += StringLen(search_key);
      int end = StringFind(json_data, ",", start);
      if(end < 0) end = StringFind(json_data, "}", start);
      if(end > start)
      {
         string value = StringSubstr(json_data, start, end - start);
         return StringToDouble(value);
      }
   }
   return 0;
}

int ExtractIntValue(string json_data, string key)
{
   string search_key = "\"" + key + "\":";
   int start = StringFind(json_data, search_key);
   if(start >= 0)
   {
      start += StringLen(search_key);
      int end = StringFind(json_data, ",", start);
      if(end < 0) end = StringFind(json_data, "}", start);
      if(end > start)
      {
         string value = StringSubstr(json_data, start, end - start);
         return StringToInteger(value);
     
   }
   return 0;
}
