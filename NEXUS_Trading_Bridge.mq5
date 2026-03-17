//+------------------------------------------------------------------+
//|                                    NEXUS_Trading_Bridge.mq5 |
//|                                    NEXUS Trading Intelligence Network |
//|                                               MQL5 WebRequest Bridge |
//+------------------------------------------------------------------+
#property copyright "NEXUS Trading Intelligence Network"
#property link      "https://github.com/Quantumlorld/Nexus-trading-intelligence-network"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- Input Parameters
input string InpWebServerURL = "http://localhost:8000";
input long   InpMagicNumber  = 123456;
input int    InpCheckInterval = 5; // Check every 5 seconds

//--- Global Variables
CTrade trade;
datetime last_check = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetMarginMode();
   
   Print("[NEXUS Trading Bridge] Started");
   Print("[NEXUS Trading Bridge] Web Server: ", InpWebServerURL);
   Print("[NEXUS Trading Bridge] Magic Number: ", InpMagicNumber);
   
   // Send initial connection status
   SendConnectionStatus(true);
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                               |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("[NEXUS Trading Bridge] Stopped");
   SendConnectionStatus(false);
}

//+------------------------------------------------------------------+
//| Expert tick function                                          |
//+------------------------------------------------------------------+
void OnTick()
{
   datetime current_time = TimeCurrent();
   
   // Check for commands every N seconds
   if(current_time - last_check >= InpCheckInterval)
   {
      CheckForCommands();
      last_check = current_time;
   }
}

//+------------------------------------------------------------------+
//| Check for commands from backend                              |
//+------------------------------------------------------------------+
void CheckForCommands()
{
   string url = InpWebServerURL + "/mt5/bridge/commands";
   string result;
   string headers;
   uchar data[];
   uchar response[];
   string result_headers;
   
   // Get pending commands from backend
   if(WebRequest("GET", url, "", headers, 5000, data, 0, response, result_headers) == 200)
   {
      // Parse and execute commands
      string commands = CharArrayToString(response);
      
      if(StringFind(commands, "\"command\":\"get_account_info\"") >= 0)
      {
         SendAccountInfo();
      }
      else if(StringFind(commands, "\"command\":\"place_order\"") >= 0)
      {
         ProcessOrderCommand(commands);
      }
      else if(StringFind(commands, "\"command\":\"get_positions\"") >= 0)
      {
         SendPositions();
      }
   }
}

//+------------------------------------------------------------------+
//| Send account information to backend                           |
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
//| Send connection status                                        |
//+------------------------------------------------------------------+
void SendConnectionStatus(bool connected)
{
   string json = "{";
   json += "\"command\":\"connection_status\",";
   json += "\"connected\":" + (connected ? "true" : "false") + ",";
   json += "\"timestamp\":" + IntegerToString(TimeCurrent());
   json += "}";
   
   SendToBackend(json);
}

//+------------------------------------------------------------------+
//| Process order command                                       |
//+------------------------------------------------------------------+
void ProcessOrderCommand(string json_data)
{
   string symbol = ExtractStringValue(json_data, "symbol");
   double volume = ExtractDoubleValue(json_data, "volume");
   int order_type = ExtractIntValue(json_data, "order_type"); // 0=buy, 1=sell
   
   if(symbol == "" || volume <= 0)
   {
      SendError("Invalid order parameters");
      return;
   }
   
   bool result = false;
   string error_msg = "";
   
   if(order_type == 0) // BUY
   {
      result = trade.Buy(volume, symbol);
      if(!result) error_msg = trade.ResultComment();
   }
   else if(order_type == 1) // SELL
   {
      result = trade.Sell(volume, symbol);
      if(!result) error_msg = trade.ResultComment();
   }
   
   string response = "{";
   response += "\"command\":\"order_result\",";
   response += "\"symbol\":\"" + symbol + "\",";
   response += "\"volume\":" + DoubleToString(volume, 2) + ",";
   response += "\"order_type\":" + IntegerToString(order_type) + ",";
   response += "\"result\":" + (result ? "true" : "false") + ",";
   response += "\"ticket\":" + IntegerToString(trade.ResultOrder()) + ",";
   response += "\"error\":\"" + (result ? "" : error_msg) + "\"";
   response += "}";
   
   SendToBackend(response);
}

//+------------------------------------------------------------------+
//| Send open positions                                        |
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
         json += "\"magic\":" + IntegerToString(PositionGetInteger(POSITION_MAGIC)) + "";
         json += "}";
      }
   }
   
   json += "]}";
   
   SendToBackend(json);
}

//+------------------------------------------------------------------+
//| Send data to backend                                        |
//+------------------------------------------------------------------+
void SendToBackend(string json_data)
{
   string url = InpWebServerURL + "/mt5/bridge/data";
   string headers = "Content-Type: application/json\r\n";
   uchar data[];
   uchar response[];
   string result_headers;
   StringToCharArray(json_data, data);
   
   WebRequest("POST", url, headers, 5000, data, 0, response, result_headers);
}

//+------------------------------------------------------------------+
//| Send error message                                        |
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
//| Helper functions for JSON parsing                              |
//+------------------------------------------------------------------+
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
         long long_value = StringToInteger(value);
         return (int)long_value;
      }
   }
   return 0;
}
