#!/bin/bash

# Nexus Trading System - End-to-End Test
# This script tests all system components

echo "🧪 Testing Nexus Trading System..."
echo "================================="

# Test Backend Health
echo "🔍 Testing Backend Health..."
curl -s http://localhost:8000/health | jq '.' || echo "Backend health check failed"

# Test Metrics Endpoint
echo "📊 Testing Metrics Endpoint..."
curl -s http://localhost:8000/metrics | head -20 || echo "Metrics endpoint failed"

# Test Frontend
echo "🌐 Testing Frontend..."
curl -s -o /dev/null -w "%{http_code}" http://localhost:6000 | grep -q "200" && echo "✅ Frontend responding" || echo "❌ Frontend not responding"

# Test Candle Data
echo "📈 Testing Candle Data..."
curl -s http://localhost:8000/candles?symbol=BTCUSDT&timeframe=1h | jq '.data | length' || echo "Candle data test failed"

# Test Trading Toggle
echo "🔄 Testing Trading Toggle..."
curl -s -X POST http://localhost:8000/admin/enable | jq '.' || echo "Trading toggle failed"

# Test Broker Simulation
echo "⚡ Testing Broker Simulation..."
curl -s -X POST http://localhost:8000/admin/simulate-broker-failure | jq '.' || echo "Broker simulation failed"

# Test Prometheus
echo "📊 Testing Prometheus..."
curl -s -o /dev/null -w "%{http_code}" http://localhost:9090/api/v1/query?query=up | grep -q "200" && echo "✅ Prometheus responding" || echo "❌ Prometheus not responding"

echo ""
echo "🎯 System Test Complete!"
echo "======================"
echo "📊 Open dashboard: http://localhost:6000"
echo "🔧 Open API docs: http://localhost:8000/docs"
echo "📈 Open Prometheus: http://localhost:9090"
echo ""
