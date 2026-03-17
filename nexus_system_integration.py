#!/usr/bin/env python3
"""
🎯 NEXUS SYSTEM INTEGRATION
Senior Quantitative Trading Systems Engineer - Complete System Integration
"""

import sys
import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import requests
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nexus_system_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NexusSystemIntegration:
    """Production-ready Nexus system integration test"""
    
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:5174"
        self.test_results = {
            'backend_status': False,
            'frontend_status': False,
            'mt5_status': False,
            'api_endpoints': [],
            'integration_tests': [],
            'start_time': datetime.now(),
            'end_time': None
        }
        
        logger.info("🎯 Nexus System Integration Initialized")
        logger.info(f"🔧 Backend URL: {self.backend_url}")
        logger.info(f"🌐 Frontend URL: {self.frontend_url}")
    
    def test_backend_health(self) -> Dict:
        """Test backend health endpoint"""
        logger.info("🔍 Testing backend health...")
        
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self.test_results['backend_status'] = True
                
                health_info = {
                    'status': 'healthy',
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds(),
                    'data': data
                }
                
                logger.info("✅ Backend health check passed!")
                logger.info(f"  📊 Status: {data.get('status', 'Unknown')}")
                logger.info(f"  🏦 Broker: {data.get('broker_status', 'Unknown')}")
                logger.info(f"  💾 Database: {data.get('db_status', 'Unknown')}")
                logger.info(f"  🔄 Trading: {data.get('trading_enabled', False)}")
                logger.info(f"  ⏱️ Response Time: {health_info['response_time']:.3f}s")
                
                return health_info
            else:
                logger.error(f"❌ Backend health failed: {response.status_code}")
                return {'status': 'failed', 'status_code': response.status_code}
                
        except Exception as e:
            logger.error(f"❌ Backend health check exception: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def test_api_endpoints(self) -> List[Dict]:
        """Test all API endpoints"""
        logger.info("🔍 Testing API endpoints...")
        
        endpoints = [
            {"name": "Health Check", "method": "GET", "path": "/health"},
            {"name": "System Status", "method": "GET", "path": "/admin/system-status"},
            {"name": "MT5 Status", "method": "GET", "path": "/admin/mt5-status"},
            {"name": "Demo Progress", "method": "GET", "path": "/admin/demo/progress"},
            {"name": "Demo Start", "method": "POST", "path": "/admin/demo/start"},
            {"name": "MT5 Connect", "method": "POST", "path": "/admin/mt5-connect"},
            {"name": "Universal MT5 Status", "method": "GET", "path": "/admin/mt5-universal-status"},
            {"name": "Universal MT5 Connect", "method": "POST", "path": "/admin/mt5-universal-connect"}
        ]
        
        endpoint_results = []
        
        for endpoint in endpoints:
            logger.info(f"🔍 Testing {endpoint['name']}...")
            
            try:
                start_time = time.time()
                
                if endpoint['method'] == 'GET':
                    response = requests.get(f"{self.backend_url}{endpoint['path']}", timeout=10)
                else:
                    response = requests.post(f"{self.backend_url}{endpoint['path']}", timeout=10)
                
                response_time = time.time() - start_time
                
                result = {
                    'name': endpoint['name'],
                    'path': endpoint['path'],
                    'method': endpoint['method'],
                    'status_code': response.status_code,
                    'response_time': response_time,
                    'success': response.status_code == 200,
                    'data': response.json() if response.status_code == 200 else None
                }
                
                endpoint_results.append(result)
                
                if result['success']:
                    logger.info(f"✅ {endpoint['name']}: {response.status_code} ({response_time:.3f}s)")
                else:
                    logger.error(f"❌ {endpoint['name']}: {response.status_code}")
                    
            except Exception as e:
                error_result = {
                    'name': endpoint['name'],
                    'path': endpoint['path'],
                    'method': endpoint['method'],
                    'status_code': 0,
                    'response_time': 0,
                    'success': False,
                    'error': str(e)
                }
                endpoint_results.append(error_result)
                logger.error(f"❌ {endpoint['name']}: Exception - {e}")
        
        self.test_results['api_endpoints'] = endpoint_results
        
        # Calculate endpoint statistics
        successful_endpoints = [e for e in endpoint_results if e['success']]
        total_endpoints = len(endpoint_results)
        
        logger.info(f"📊 API Endpoints: {len(successful_endpoints)}/{total_endpoints} working")
        logger.info(f"  ✅ Working: {len(successful_endpoints)}")
        logger.info(f"  ❌ Failed: {total_endpoints - len(successful_endpoints)}")
        
        return endpoint_results
    
    def test_frontend_access(self) -> Dict:
        """Test frontend accessibility"""
        logger.info("🔍 Testing frontend accessibility...")
        
        try:
            response = requests.get(self.frontend_url, timeout=10)
            
            if response.status_code == 200:
                self.test_results['frontend_status'] = True
                
                frontend_info = {
                    'status': 'accessible',
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds(),
                    'content_length': len(response.content)
                }
                
                logger.info("✅ Frontend accessibility test passed!")
                logger.info(f"  🌐 URL: {self.frontend_url}")
                logger.info(f"  📊 Status Code: {response.status_code}")
                logger.info(f"  ⏱️ Response Time: {frontend_info['response_time']:.3f}s")
                logger.info(f"  📄 Content Length: {frontend_info['content_length']} bytes")
                
                return frontend_info
            else:
                logger.error(f"❌ Frontend not accessible: {response.status_code}")
                return {'status': 'inaccessible', 'status_code': response.status_code}
                
        except Exception as e:
            logger.error(f"❌ Frontend accessibility exception: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def test_mt5_integration(self) -> Dict:
        """Test MT5 integration"""
        logger.info("🔍 Testing MT5 integration...")
        
        try:
            # Test MT5 status
            response = requests.get(f"{self.backend_url}/admin/mt5-status", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self.test_results['mt5_status'] = data.get('connected', False)
                
                mt5_info = {
                    'connected': data.get('connected', False),
                    'account_info': data.get('account_info', {}),
                    'server': data.get('server', ''),
                    'demo_mode': data.get('demo_mode', False),
                    'trade_count': data.get('trade_count', 0),
                    'current_phase': data.get('current_phase', ''),
                    'adaptive_learning': data.get('adaptive_learning', False)
                }
                
                if mt5_info['connected']:
                    logger.info("✅ MT5 integration test passed!")
                    logger.info(f"  👤 Account: {mt5_info['account_info'].get('account', 'Unknown')}")
                    logger.info(f"  🏦 Server: {mt5_info['server']}")
                    logger.info(f"  💰 Balance: ${mt5_info['account_info'].get('balance', 0):.2f}")
                    logger.info(f"  📈 Equity: ${mt5_info['account_info'].get('equity', 0):.2f}")
                    logger.info(f"  🎯 Demo Mode: {mt5_info['demo_mode']}")
                    logger.info(f"  📊 Trade Count: {mt5_info['trade_count']}")
                    logger.info(f"  🧠 Current Phase: {mt5_info['current_phase']}")
                else:
                    logger.warning("⚠️ MT5 not connected")
                    logger.info(f"  📝 Message: {data.get('message', 'Unknown')}")
                
                return mt5_info
            else:
                logger.error(f"❌ MT5 status check failed: {response.status_code}")
                return {'connected': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            logger.error(f"❌ MT5 integration test exception: {e}")
            return {'connected': False, 'error': str(e)}
    
    def test_demo_trading_system(self) -> Dict:
        """Test demo trading system"""
        logger.info("🔍 Testing demo trading system...")
        
        try:
            # Start demo trading
            response = requests.post(f"{self.backend_url}/admin/demo/start", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                demo_info = {
                    'started': data.get('success', False),
                    'message': data.get('message', ''),
                    'plan': data.get('plan', {}),
                    'current_phase': data.get('current_phase', ''),
                    'features': data.get('features', [])
                }
                
                if demo_info['started']:
                    logger.info("✅ Demo trading system test passed!")
                    logger.info(f"  📊 Plan: {demo_info['plan']}")
                    logger.info(f"  🎯 Current Phase: {demo_info['current_phase']}")
                    logger.info(f"  ✨ Features: {demo_info['features']}")
                    
                    # Check progress
                    progress_response = requests.get(f"{self.backend_url}/admin/demo/progress", timeout=10)
                    if progress_response.status_code == 200:
                        progress_data = progress_response.json()
                        demo_info['progress'] = progress_data
                        logger.info(f"  📈 Progress: {progress_data.get('trade_count', 0)} trades")
                else:
                    logger.warning("⚠️ Demo trading start failed")
                    logger.info(f"  📝 Error: {demo_info['message']}")
                
                return demo_info
            else:
                logger.error(f"❌ Demo trading start failed: {response.status_code}")
                return {'started': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            logger.error(f"❌ Demo trading test exception: {e}")
            return {'started': False, 'error': str(e)}
    
    def test_end_to_end_integration(self) -> Dict:
        """Test end-to-end integration"""
        logger.info("🔍 Testing end-to-end integration...")
        
        integration_results = {
            'backend_to_frontend': False,
            'mt5_to_backend': False,
            'demo_to_mt5': False,
            'data_flow': False
        }
        
        try:
            # Test 1: Backend to Frontend
            logger.info("🔍 Testing Backend → Frontend...")
            backend_health = self.test_backend_health()
            frontend_access = self.test_frontend_access()
            
            if backend_health['status'] == 'healthy' and frontend_access['status'] == 'accessible':
                integration_results['backend_to_frontend'] = True
                logger.info("✅ Backend → Frontend: Working")
            
            # Test 2: MT5 to Backend
            logger.info("🔍 Testing MT5 → Backend...")
            mt5_status = self.test_mt5_integration()
            
            if mt5_status.get('connected', False):
                integration_results['mt5_to_backend'] = True
                logger.info("✅ MT5 → Backend: Working")
            
            # Test 3: Demo to MT5
            logger.info("🔍 Testing Demo → MT5...")
            demo_status = self.test_demo_trading_system()
            
            if demo_status.get('started', False):
                integration_results['demo_to_mt5'] = True
                logger.info("✅ Demo → MT5: Working")
            
            # Test 4: Data Flow
            logger.info("🔍 Testing Data Flow...")
            
            # Check if all components are working
            if (integration_results['backend_to_frontend'] and 
                integration_results['mt5_to_backend'] and 
                integration_results['demo_to_mt5']):
                integration_results['data_flow'] = True
                logger.info("✅ Data Flow: Working")
            
        except Exception as e:
            logger.error(f"❌ End-to-end integration test exception: {e}")
        
        return integration_results
    
    def run_comprehensive_integration_test(self) -> Dict:
        """Run comprehensive system integration test"""
        logger.info("🚀 STARTING COMPREHENSIVE SYSTEM INTEGRATION TEST")
        logger.info("=" * 60)
        
        try:
            # Test 1: Backend Health
            logger.info("🔍 PHASE 1: BACKEND HEALTH TEST")
            backend_health = self.test_backend_health()
            
            # Test 2: Frontend Accessibility
            logger.info("🔍 PHASE 2: FRONTEND ACCESSIBILITY TEST")
            frontend_access = self.test_frontend_access()
            
            # Test 3: API Endpoints
            logger.info("🔍 PHASE 3: API ENDPOINTS TEST")
            api_endpoints = self.test_api_endpoints()
            
            # Test 4: MT5 Integration
            logger.info("🔍 PHASE 4: MT5 INTEGRATION TEST")
            mt5_integration = self.test_mt5_integration()
            
            # Test 5: Demo Trading System
            logger.info("🔍 PHASE 5: DEMO TRADING SYSTEM TEST")
            demo_trading = self.test_demo_trading_system()
            
            # Test 6: End-to-End Integration
            logger.info("🔍 PHASE 6: END-TO-END INTEGRATION TEST")
            end_to_end = self.test_end_to_end_integration()
            
            # Complete test results
            self.test_results['end_time'] = datetime.now()
            self.test_results['duration'] = self.test_results['end_time'] - self.test_results['start_time']
            
            # Calculate overall statistics
            working_endpoints = [e for e in api_endpoints if e['success']]
            total_endpoints = len(api_endpoints)
            
            overall_status = {
                'backend_healthy': backend_health['status'] == 'healthy',
                'frontend_accessible': frontend_access['status'] == 'accessible',
                'api_success_rate': (len(working_endpoints) / total_endpoints * 100) if total_endpoints > 0 else 0,
                'mt5_connected': mt5_integration.get('connected', False),
                'demo_trading_active': demo_trading.get('started', False),
                'end_to_end_working': all(end_to_end.values()),
                'overall_score': 0
            }
            
            # Calculate overall score
            score_components = [
                overall_status['backend_healthy'],
                overall_status['frontend_accessible'],
                overall_status['api_success_rate'] >= 80,
                overall_status['mt5_connected'],
                overall_status['demo_trading_active'],
                overall_status['end_to_end_working']
            ]
            
            overall_status['overall_score'] = (sum(score_components) / len(score_components)) * 100
            
            self.test_results['overall_status'] = overall_status
            
            return self.test_results
            
        except KeyboardInterrupt:
            logger.info("⏹️ Test interrupted by user")
        except Exception as e:
            logger.error(f"❌ Test execution failed: {e}")
        
        return self.test_results
    
    def print_integration_results(self, results: Dict):
        """Print comprehensive integration results"""
        logger.info("=" * 60)
        logger.info("📊 COMPREHENSIVE SYSTEM INTEGRATION RESULTS")
        logger.info("=" * 60)
        
        # Basic results
        logger.info(f"🎯 Test Duration: {results['duration']}")
        logger.info(f"🔧 Backend Status: {'✅ Healthy' if results['backend_status'] else '❌ Unhealthy'}")
        logger.info(f"🌐 Frontend Status: {'✅ Accessible' if results['frontend_status'] else '❌ Inaccessible'}")
        logger.info(f"🏦 MT5 Status: {'✅ Connected' if results['mt5_status'] else '❌ Disconnected'}")
        
        # API endpoints
        if results['api_endpoints']:
            working_endpoints = [e for e in results['api_endpoints'] if e['success']]
            total_endpoints = len(results['api_endpoints'])
            
            logger.info(f"🔗 API Endpoints: {len(working_endpoints)}/{total_endpoints} working")
            logger.info(f"  ✅ Working: {len(working_endpoints)}")
            logger.info(f"  ❌ Failed: {total_endpoints - len(working_endpoints)}")
            
            # Show failed endpoints
            failed_endpoints = [e for e in results['api_endpoints'] if not e['success']]
            if failed_endpoints:
                logger.info("❌ Failed Endpoints:")
                for endpoint in failed_endpoints:
                    logger.info(f"  📝 {endpoint['name']}: {endpoint.get('error', 'Unknown error')}")
        
        # Overall status
        if 'overall_status' in results:
            status = results['overall_status']
            logger.info("📊 Overall System Status:")
            logger.info(f"  🔧 Backend: {'✅' if status['backend_healthy'] else '❌'}")
            logger.info(f"  🌐 Frontend: {'✅' if status['frontend_accessible'] else '❌'}")
            logger.info(f"  🔗 API Success Rate: {status['api_success_rate']:.1f}%")
            logger.info(f"  🏦 MT5: {'✅' if status['mt5_connected'] else '❌'}")
            logger.info(f"  🎯 Demo Trading: {'✅' if status['demo_trading_active'] else '❌'}")
            logger.info(f"  🔄 End-to-End: {'✅' if status['end_to_end_working'] else '❌'}")
            logger.info(f"  📊 Overall Score: {status['overall_score']:.1f}%")
        
        # Final assessment
        if 'overall_status' in results:
            score = results['overall_status']['overall_score']
            
            logger.info("=" * 60)
            if score >= 90:
                logger.info("🎉 SYSTEM INTEGRATION: EXCELLENT!")
                logger.info("✅ Nexus Trading System is fully operational!")
            elif score >= 75:
                logger.info("✅ SYSTEM INTEGRATION: VERY GOOD!")
                logger.info("🔧 Minor improvements may be needed")
            elif score >= 50:
                logger.info("⚠️ SYSTEM INTEGRATION: NEEDS IMPROVEMENT!")
                logger.info("🔧 Several components need attention")
            else:
                logger.error("❌ SYSTEM INTEGRATION: CRITICAL ISSUES!")
                logger.error("🔧 Major improvements required")
        
        logger.info("=" * 60)
        
        # Recommendations
        logger.info("🎯 RECOMMENDATIONS:")
        
        if not results['backend_status']:
            logger.info("  🔧 Start backend server: python simple_app.py")
        
        if not results['frontend_status']:
            logger.info("  🌐 Start frontend server: npm run dev")
        
        if not results['mt5_status']:
            logger.info("  🏦 Start MT5 terminal and login to XM Global")
        
        if 'overall_status' in results and results['overall_status']['overall_score'] < 100:
            logger.info("  🔧 Review failed components and fix issues")
        
        logger.info("=" * 60)

def main():
    """Main integration test execution"""
    logger.info("🎯 NEXUS SYSTEM INTEGRATION TEST")
    logger.info("🔧 Senior Quantitative Trading Systems Engineer")
    
    # Create integration test instance
    integration_tester = NexusSystemIntegration()
    
    # Run comprehensive test
    try:
        results = integration_tester.run_comprehensive_integration_test()
        integration_tester.print_integration_results(results)
        
        # Save results to file
        results_file = f"nexus_integration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to: {results_file}")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")

if __name__ == "__main__":
    main()
