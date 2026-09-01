
---

## tests/test_api.py

```python
# tests/test_api.py - API Testing Script
import requests
import json
import time
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Configuration
API_URL = "http://localhost:8000"
TEST_SAMPLES = [
    {
        "name": "Healthy",
        "data": {
            "temperature": 65.0,
            "vibration": 0.8,
            "rpm": 1450.0,
            "pressure": 5.2,
            "load": 75.0,
            "maintenance_history": 30.0,
            "current": 10.5,
            "speed": 1500.0
        },
        "expected_status": "Healthy"
    },
    {
        "name": "Warning",
        "data": {
            "temperature": 78.0,
            "vibration": 2.1,
            "rpm": 1800.0,
            "pressure": 7.5,
            "load": 85.0,
            "maintenance_history": 10.0,
            "current": 18.0,
            "speed": 2000.0
        },
        "expected_status": "Warning"
    },
    {
        "name": "Critical",
        "data": {
            "temperature": 89.0,
            "vibration": 3.8,
            "rpm": 2200.0,
            "pressure": 9.2,
            "load": 95.0,
            "maintenance_history": 5.0,
            "current": 24.0,
            "speed": 2500.0
        },
        "expected_status": "Critical"
    }
]

# ============================================
# Helper Functions
# ============================================

def print_section(title):
    """Print a section header"""
    print("\n" + "="*60)
    print(f"📋 {title}")
    print("="*60)

def print_result(test_name, passed, message=""):
    """Print test result"""
    if passed:
        print(f"   ✅ {test_name}: PASSED")
        if message:
            print(f"      {message}")
    else:
        print(f"   ❌ {test_name}: FAILED")
        if message:
            print(f"      {message}")
    return passed

def wait_for_server():
    """Wait for server to be ready"""
    print("\n⏳ Checking if server is running...")
    for i in range(5):
        try:
            response = requests.get(f"{API_URL}/api/health", timeout=2)
            if response.status_code == 200:
                print("✅ Server is ready")
                return True
        except:
            pass
        print(f"   Waiting... ({i+1}/5)")
        time.sleep(2)
    
    print("❌ Server not responding")
    return False

# ============================================
# Test Functions
# ============================================

def test_health():
    """Test health endpoint"""
    print_section("HEALTH CHECK")
    
    try:
        response = requests.get(f"{API_URL}/api/health", timeout=5)
        passed = response.status_code == 200
        
        if passed:
            data = response.json()
            print(f"   Status: {data.get('status', 'Unknown')}")
            print(f"   Model loaded: {data.get('model_loaded', False)}")
            print(f"   Features: {data.get('features', [])}")
        else:
            print(f"   Error: {response.status_code}")
        
        return print_result("Health Check", passed)
    except Exception as e:
        print(f"   Error: {e}")
        return print_result("Health Check", False)

def test_features():
    """Test features endpoint"""
    print_section("FEATURES ENDPOINT")
    
    try:
        response = requests.get(f"{API_URL}/api/features", timeout=5)
        passed = response.status_code == 200
        
        if passed:
            data = response.json()
            print(f"   Features: {data.get('features', [])}")
            print(f"   Count: {data.get('count', 0)}")
            print(f"   Model type: {data.get('model_type', 'Unknown')}")
        else:
            print(f"   Error: {response.status_code}")
        
        return print_result("Get Features", passed)
    except Exception as e:
        print(f"   Error: {e}")
        return print_result("Get Features", False)

def test_predict():
    """Test predict endpoint"""
    print_section("PREDICTION ENDPOINT")
    
    all_passed = True
    
    for sample in TEST_SAMPLES:
        try:
            response = requests.post(
                f"{API_URL}/api/predict",
                json=sample["data"],
                timeout=10
            )
            
            passed = response.status_code == 200
            
            if passed:
                data = response.json()
                rul = data.get('rul_days', 0)
                status = data.get('status', 'Unknown')
                confidence = data.get('confidence', 0)
                
                print(f"\n   📊 {sample['name']}:")
                print(f"      RUL: {rul:.1f} days")
                print(f"      Status: {status}")
                print(f"      Confidence: {confidence*100:.1f}%")
                print(f"      Expected: {sample['expected_status']}")
                
                # Check if status matches expectation (approximately)
                expected = sample['expected_status']
                if status == expected:
                    print(f"      ✅ Status matches expectation")
                else:
                    print(f"      ⚠️ Status differs (got: {status}, expected: {expected})")
            else:
                print(f"\n   ❌ {sample['name']}: Error {response.status_code}")
                passed = False
            
            all_passed = all_passed and passed
            
        except Exception as e:
            print(f"\n   ❌ {sample['name']}: {e}")
            all_passed = False
    
    return print_result("Prediction Tests", all_passed)

def test_simulate():
    """Test simulation endpoint"""
    print_section("SIMULATION ENDPOINT")
    
    sample = TEST_SAMPLES[0]  # Use healthy sample
    load_increase = 0.3
    
    try:
        data = sample["data"].copy()
        data["load_increase"] = load_increase
        
        response = requests.post(
            f"{API_URL}/api/simulate",
            json=data,
            timeout=10
        )
        
        passed = response.status_code == 200
        
        if passed:
            result = response.json()
            rul = result.get('rul_days', 0)
            original_rul = result.get('original_rul', 0)
            is_sim = result.get('isSimulation', False)
            
            print(f"\n   Original RUL: {original_rul:.1f} days")
            print(f"   Simulated RUL ({load_increase*100:.0f}% load): {rul:.1f} days")
            print(f"   Reduction: {original_rul - rul:.1f} days ({((original_rul - rul)/original_rul*100):.1f}%)")
            print(f"   Simulation flag: {is_sim}")
            
            if is_sim and rul < original_rul:
                print("   ✅ Simulation effect detected")
            else:
                print("   ⚠️ Simulation effect not detected")
        else:
            print(f"   Error: {response.status_code}")
        
        return print_result("Simulation Test", passed)
        
    except Exception as e:
        print(f"   Error: {e}")
        return print_result("Simulation Test", False)

def test_batch_predict():
    """Test batch prediction endpoint"""
    print_section("BATCH PREDICTION ENDPOINT")
    
    samples = [s["data"] for s in TEST_SAMPLES]
    
    try:
        response = requests.post(
            f"{API_URL}/api/batch_predict",
            json={"samples": samples},
            timeout=15
        )
        
        passed = response.status_code == 200
        
        if passed:
            data = response.json()
            results = data.get('results', [])
            
            print(f"\n   Batch size: {len(results)}")
            for i, result in enumerate(results):
                if result.get('status') != 'error':
                    rul = result.get('rul_days', 0)
                    status = result.get('status', 'Unknown')
                    print(f"   Sample {i+1}: RUL={rul:.1f} days, Status={status}")
                else:
                    print(f"   Sample {i+1}: Error - {result.get('message', 'Unknown error')}")
            
            if len(results) == len(samples):
                print("   ✅ All samples processed")
            else:
                print("   ⚠️ Sample count mismatch")
        else:
            print(f"   Error: {response.status_code}")
        
        return print_result("Batch Prediction", passed)
        
    except Exception as e:
        print(f"   Error: {e}")
        return print_result("Batch Prediction", False)

def test_invalid_input():
    """Test invalid input handling"""
    print_section("INVALID INPUT HANDLING")
    
    all_passed = True
    
    # Test missing features
    try:
        response = requests.post(
            f"{API_URL}/api/predict",
            json={"temperature": 65.0},  # Missing features
            timeout=5
        )
        
        passed = response.status_code == 400
        if passed:
            print("   ✅ Missing features correctly rejected")
        else:
            print(f"   ❌ Expected 400, got {response.status_code}")
        all_passed = all_passed and passed
    except Exception as e:
        print(f"   ❌ Error: {e}")
        all_passed = False
    
    # Test invalid data type
    try:
        response = requests.post(
            f"{API_URL}/api/predict",
            json={"temperature": "invalid"},  # String instead of number
            timeout=5
        )
        
        passed = response.status_code == 400 or response.status_code == 500
        if passed:
            print("   ✅ Invalid data type correctly handled")
        else:
            print(f"   ❌ Expected 400/500, got {response.status_code}")
        all_passed = all_passed and passed
    except Exception as e:
        print(f"   ❌ Error: {e}")
        all_passed = False
    
    return print_result("Invalid Input Tests", all_passed)

def test_cors():
    """Test CORS headers"""
    print_section("CORS HEADERS")
    
    try:
        response = requests.options(
            f"{API_URL}/api/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST"
            }
        )
        
        has_cors = 'access-control-allow-origin' in response.headers
        if has_cors:
            print("   ✅ CORS headers present")
        else:
            print("   ⚠️ CORS headers not found")
        
        return print_result("CORS Check", has_cors)
    except Exception as e:
        print(f"   Error: {e}")
        return print_result("CORS Check", False)

def test_performance():
    """Test API performance"""
    print_section("PERFORMANCE TEST")
    
    sample = TEST_SAMPLES[0]["data"]
    times = []
    n_requests = 10
    
    print(f"   Running {n_requests} predictions...")
    
    for i in range(n_requests):
        start = time.time()
        try:
            response = requests.post(
                f"{API_URL}/api/predict",
                json=sample,
                timeout=5
            )
            elapsed = time.time() - start
            times.append(elapsed)
        except:
            times.append(float('inf'))
    
    valid_times = [t for t in times if t < float('inf')]
    
    if valid_times:
        avg_time = sum(valid_times) / len(valid_times)
        max_time = max(valid_times)
        min_time = min(valid_times)
        
        print(f"   Average: {avg_time*1000:.1f} ms")
        print(f"   Min: {min_time*1000:.1f} ms")
        print(f"   Max: {max_time*1000:.1f} ms")
        
        passed = avg_time < 1.0  # Less than 1 second average
        if passed:
            print("   ✅ Performance is good")
        else:
            print("   ⚠️ Performance could be improved")
    else:
        print("   ❌ All requests failed")
        passed = False
    
    return print_result("Performance Test", passed)

# ============================================
# Main Test Runner
# ============================================

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("🚀 BEARING RUL API TESTS")
    print("="*60)
    print(f"🌐 API URL: {API_URL}")
    
    # Check if server is running
    if not wait_for_server():
        print("\n❌ Cannot run tests - server not available")
        print("   Start the server first: python app.py")
        return False
    
    # Run tests
    tests = [
        test_health,
        test_features,
        test_predict,
        test_simulate,
        test_batch_predict,
        test_invalid_input,
        test_cors,
        test_performance
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"   ❌ Test crashed: {e}")
            results.append((test.__name__, False))
    
    # Print summary
    print_section("TEST SUMMARY")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {name.replace('test_', '')}: {status}")
    
    print("\n" + "="*60)
    print(f"📊 Results: {passed}/{total} tests passed")
    print("="*60)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)