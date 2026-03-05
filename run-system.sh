#!/bin/bash

# Nexus Trading System - Full Stack Runner
# This script starts the complete system with Docker Compose

echo "🚀 Starting Nexus Trading System..."
echo "=================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose > /dev/null 2>&1; then
    echo "❌ docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose-full.yml down

# Build and start the system
echo "🔨 Building and starting containers..."
docker-compose -f docker-compose-full.yml up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check Backend
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is healthy"
else
    echo "❌ Backend is not responding"
fi

# Check Frontend
if curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Frontend is healthy"
else
    echo "❌ Frontend is not responding"
fi

# Check Prometheus
if curl -f http://localhost:9090 > /dev/null 2>&1; then
    echo "✅ Prometheus is healthy"
else
    echo "❌ Prometheus is not responding"
fi

echo ""
echo "🎉 Nexus Trading System is running!"
echo "=================================="
echo "📊 Frontend Dashboard: http://localhost:6000"
echo "🔧 Backend API: http://localhost:8000/docs"
echo "📈 Prometheus Metrics: http://localhost:9090"
echo "🗄️  PostgreSQL: localhost:5432"
echo "🔴 Redis: localhost:6379"
echo ""
echo "📋 To view logs: docker-compose -f docker-compose-full.yml logs -f"
echo "🛑 To stop system: docker-compose -f docker-compose-full.yml down"
echo ""
