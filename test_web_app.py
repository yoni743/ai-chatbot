#!/usr/bin/env python3
"""
Test script for the Flask web application.
Tests all routes and templates to ensure they work correctly.
"""

import requests
import json
import sys
import time
from threading import Thread
import subprocess
import os

def test_web_app():
    """Test the web application endpoints."""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Flask Web Application...")
    print("=" * 50)
    
    # Test 1: Main page
    print("1. Testing main page (/)...")
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            print("✅ Main page loads successfully")
        else:
            print(f"❌ Main page failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Main page error: {e}")
        return False
    
    # Test 2: Favicon
    print("2. Testing favicon (/favicon.ico)...")
    try:
        response = requests.get(f"{base_url}/favicon.ico", timeout=5)
        if response.status_code == 204:
            print("✅ Favicon handled correctly")
        else:
            print(f"⚠️  Favicon returned: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Favicon error: {e}")
    
    # Test 3: Status endpoint
    print("3. Testing status endpoint (/api/status)...")
    try:
        response = requests.get(f"{base_url}/api/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status endpoint works: {data.get('status', 'unknown')}")
        else:
            print(f"❌ Status endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Status endpoint error: {e}")
        return False
    
    # Test 4: Chat endpoint
    print("4. Testing chat endpoint (/api/chat)...")
    try:
        chat_data = {"message": "Hello, how are you?"}
        response = requests.post(
            f"{base_url}/api/chat", 
            json=chat_data, 
            headers={'Content-Type': 'application/json'},
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            if 'response' in data:
                print(f"✅ Chat endpoint works: {data['response'][:50]}...")
            else:
                print(f"❌ Chat response missing 'response' field")
                return False
        else:
            print(f"❌ Chat endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Chat endpoint error: {e}")
        return False
    
    # Test 5: Settings endpoint
    print("5. Testing settings endpoint (/api/settings)...")
    try:
        settings_data = {"threshold": 0.8}
        response = requests.post(
            f"{base_url}/api/settings", 
            json=settings_data, 
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Settings endpoint works: {data.get('message', 'success')}")
        else:
            print(f"❌ Settings endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Settings endpoint error: {e}")
        return False
    
    # Test 6: Clear endpoint
    print("6. Testing clear endpoint (/api/clear)...")
    try:
        response = requests.post(f"{base_url}/api/clear", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Clear endpoint works: {data.get('message', 'success')}")
        else:
            print(f"❌ Clear endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Clear endpoint error: {e}")
        return False
    
    # Test 7: 404 handling
    print("7. Testing 404 handling...")
    try:
        response = requests.get(f"{base_url}/nonexistent", timeout=5)
        if response.status_code == 404:
            print("✅ 404 handling works correctly")
        else:
            print(f"⚠️  404 handling returned: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️  404 test error: {e}")
    
    print("\n🎉 All tests completed!")
    return True

def start_test_server():
    """Start the Flask app in a separate process for testing."""
    print("🚀 Starting test server...")
    try:
        # Start the Flask app
        process = subprocess.Popen([
            sys.executable, "app.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        time.sleep(5)
        
        return process
    except Exception as e:
        print(f"❌ Failed to start test server: {e}")
        return None

def main():
    """Main test function."""
    print("🧪 Flask Web Application Test Suite")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found")
        print("Please run this script from the chatbot directory")
        return 1
    
    # Check if model exists
    if not os.path.exists("models/chatbot_model.h5"):
        print("❌ Error: Chatbot model not found")
        print("Please train the model first: python src/train.py")
        return 1
    
    # Start test server
    server_process = start_test_server()
    if not server_process:
        return 1
    
    try:
        # Run tests
        success = test_web_app()
        
        if success:
            print("\n✅ All tests passed! Web application is working correctly.")
            return 0
        else:
            print("\n❌ Some tests failed. Check the output above.")
            return 1
            
    finally:
        # Clean up
        if server_process:
            server_process.terminate()
            server_process.wait()
            print("\n🛑 Test server stopped")

if __name__ == "__main__":
    sys.exit(main())

