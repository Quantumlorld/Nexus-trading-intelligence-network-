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

input string InpWebServerURL = "http://localhost:8000";  // NEXUS Backend URL
input int    InpWebServerPort = 8000;                    // Backend Port
input long   InpMagicNumber = 123456;                   // Magic Number for trades

CTrade trade;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetMarginMode();
   
   Print("[NEXUS MT5 Bridge EA] Initialized");
   Print("[NEXUS MT5 Bridge EA] Backend URL: ", InpWebServerURL, ":", InpWebServerPort);
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("[NEXUS MT5 Bridge EA] Stopped");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   static datetime last_check = 0;
   
   // Check for commands every 5 seconds
   if(TimeCurrent() - last_check >= 5)
   {
      CheckBackendCommands();
      last_check = TimeCurrent();
   }
}

//+------------------------------------------------------------------+
//| Check for commands from backend                                   |
//+------------------------------------------------------------------+
void CheckBackendCommands()
{
   string url = InpWebServerURL + "/mt5/bridge/commands";
   string result;
   string headers;
   int timeout = 3000; // 3 seconds
   
   // Send GET request to check for commands
   if(WebRequest("GET", url, "", headers, timeout, result, 0, headers) == 200)
   {
      // Parse JSON response for commands
      ProcessCommands(result);
   }
}

//+------------------------------------------------------------------+
//| Process commands from JSON                                        |
//+------------------------------------------------------------------+
void ProcessCommands(string json_data)
{
   // Simple JSON parsing - look for command patterns
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

//+------------------------------------------------------------------+
//| Send account information to backend                               |
//+------------------------------------------------------------------+
void SendAccountInfo()
{
   CAccountInfo account;
   
   string json = "{";
   json += "\"command\":\"account_info\",";
   json += "\"login\":" + IntegerToString(account.Login()) + ",";
   json += "\"server\":\"" + account.Server() + "\",";
   json += "\"balance\":" + DoubleToString(account.Balance(), 2) + ",";
   json += "\"equity\":" + DoubleToString(account.Equity(), 2) + ",";
   json += "\"margin\":" + DoubleToString(account.Margin(), 2) + ",";
   json += "\"free_margin\":" + DoubleToString(account.FreeMargin(), 2) + ",";
   json += "\"profit\":" + DoubleToString(account.Profit(), 2) + ",";
   json += "\"leverage\":" + IntegerToString(account.Leverage()) + ",";
   json += "\"currency\":\"" + account.Currency() + "\"";
   json += "}";
   
   SendToBackend(json);
}

//+------------------------------------------------------------------+
//| Send symbol information to backend                                |
//+------------------------------------------------------------------+
void SendSymbolInfo(string symbol)
{
   MqlTick tick;
   
   string json = "{";
   json += "\"command\":\"symbol_info\",";
   json += "\"symbol\":\"" + symbol + "\",";
   
   // Get current tick
   if(SymbolInfoTick(symbol, tick))
   {
      json += "\"bid\":" + DoubleToString(tick.bid, 5) + ",";
      json += "\"ask\":" + DoubleToString(tick.ask, 5) + ",";
      json += "\"last\":" + DoubleToString(tick.last, 5) + ",";
      json += "\"volume\":" + IntegerToString(tick.volume) + ",";
      json += "\"time\":" + IntegerToString(tick.time) + ",";
   }
   
   json += "}";
   
   SendToBackend(json);
}

//+------------------------------------------------------------------+
//| Process order command                                             |
//+------------------------------------------------------------------+
void ProcessOrderCommand(string json_data)
{
   string symbol = ExtractStringValue(json_data, "symbol");
   double volume = ExtractDoubleValue(json_data, "volume");
   int order_type = ExtractIntValue(json_data, "order_type"); // 0=buy, 1=sell
   double price = ExtractDoubleValue(json_data, "price");
   double sl = ExtractDoubleValue(json_data, "sl");
   double tp = ExtractDoubleValue(json_data, "tp");
   string comment = ExtractStringValue(json_data, "comment");
   
   if(symbol == "" || volume <= 0)
   {
      SendError("Invalid order parameters");
      return;
   }
   
   bool result = false;
   string error_msg = "";
   
   if(order_type == 0) // BUY
   {
      result = trade.Buy(volume, symbol, price, sl, tp, comment);
   }
   else if(order_type == 1) // SELL
   {
      result = trade.Sell(volume, symbol, price, sl, tp, comment);
   }
   
   string response = "{";
   response += "\"command\":\"order_result\",";
   response += "\"symbol\":\"" + symbol + "\",";
   response += "\"volume\":" + DoubleToString(volume, 2) + ",";
   response += "\"order_type\":" + IntegerToString(order_type) + ",";
   response += "\"result\":" + (result ? "true" : "false") + ",";
   response += "\"ticket\":" + IntegerToString(trade.ResultOrder()) + ",";
   response += "\"error\":\"" + (result ? "" : trade.ResultComment()) + "\"";
   response += "}";
   
   SendToBackend(response);
}

//+------------------------------------------------------------------+
//| Send open positions to backend                                   |
//+------------------------------------------------------------------+
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
         json += "\"ticket\":" + IntegerToString(PositionGetInteger(POSITION_TICKET)) + ",";
         json += "\"symbol\":\"" + PositionGetString(POSITION_SYMBOL) + "\",";
         json += "\"type\":" + IntegerToString(PositionGetInteger(POSITION_TYPE)) + ",";
         json += "\"volume\":" + DoubleToString(PositionGetDouble(POSITION_VOLUME), 2) + ",";
         json += "\"price_open\":" + DoubleToString(PositionGetDouble(POSITION_PRICE_OPEN), 5) + ",";
         json += "\"price_current\":" + DoubleToString(PositionGetDouble(POSITION_PRICE_CURRENT), 5) + ",";
         json += "\"profit\":" + DoubleToString(PositionGetDouble(POSITION_PROFIT), 2) + ",";
         json += "\"sl\":" + DoubleToString(PositionGetDouble(POSITION_SL), 5) + ",";
         json += "\"tp\":" + DoubleToString(PositionGetDouble(POSITION_TP), 5) + ",";
         json += "\"magic\":" + IntegerToString(PositionGetInteger(POSITION_MAGIC)) + ",";
         json += "\"time\":" + IntegerToString(PositionGetInteger(POSITION_TIME)) + "";
         json += "}";
      }
   }
   
   json += "]}";
   
   SendToBackend(json);
}

//+------------------------------------------------------------------+
//| Send data to backend                                              |
//+------------------------------------------------------------------+
void SendToBackend(string json_data)
{
   string url = InpWebServerURL + "/mt5/bridge/data";
   string headers = "Content-Type: application/json\r\n";
   int timeout = 5000; // 5 seconds
   
   WebRequest("POST", url, headers, timeout, json_data, 0, headers);
}

//+------------------------------------------------------------------+
//| Send error message to backend                                     |
//+------------------------------------------------------------------+
void SendError(string error_msg)
{
   string json = "{";
   json += "\"command\":\"error\",";
   json += "\"error\":\"" + error_msg + "\"";
   json += "}";
   
   SendToBackend(json);
}

//+------------------------------------------------------------------+
//| Helper functions for JSON parsing                                 |
//+------------------------------------------------------------------+
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
   }
   return 0;
}
