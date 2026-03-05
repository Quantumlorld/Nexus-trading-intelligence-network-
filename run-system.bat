@echo off
REM Nexus Trading System - Full Stack Runner for Windows
REM This script starts the complete system with Docker Compose

echo 🚀 Starting Nexus Trading System...
echo ==================================

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker first.
    pause
    exit /b 1
)

REM Stop any existing containers
echo 🛑 Stopping existing containers...
docker-compose -f docker-compose-full.yml down

REM Build and start the system
echo 🔨 Building and starting containers...
docker-compose -f docker-compose-full.yml up --build -d

REM Wait for services to be ready
echo ⏳ Waiting for services to be ready...
timeout /t 30 /nobreak >nul

REM Check service health
echo 🔍 Checking service health...

REM Check Backend
curl -f http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Backend is healthy
) else (
    echo ❌ Backend is not responding
)

REM Check Frontend
curl -f http://localhost:6000 >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Frontend is healthy
) else (
    echo ❌ Frontend is not responding
)

REM Check Prometheus
curl -f http://localhost:9090 >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Prometheus is healthy
) else (
    echo ❌ Prometheus is not responding
)

echo.
echo 🎉 Nexus Trading System is running!
echo ==================================
echo 📊 Frontend Dashboard: http://localhost:6000
echo 🔧 Backend API: http://localhost:8000/docs
echo 📈 Prometheus Metrics: http://localhost:9090
echo 🗄️  PostgreSQL: localhost:5432
echo 🔴 Redis: localhost:6379
echo.
echo 📋 To view logs: docker-compose -f docker-compose-full.yml logs -f
echo 🛑 To stop system: docker-compose -f docker-compose-full.yml down
echo.
pause
