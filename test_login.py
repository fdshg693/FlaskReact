#!/usr/bin/env python3
"""
Simple test script to verify login functionality is working correctly.
Run this to test the authentication endpoints.
"""

import requests
import json

def test_login_functionality():
    """Test the login system functionality."""
    base_url = "http://127.0.0.1:8000"
    session = requests.Session()
    
    print("🧪 Testing Login Functionality")
    print("=" * 40)
    
    # Test 1: Check authentication status (should be false)
    print("1. Testing initial auth status...")
    response = session.get(f"{base_url}/api/auth/status")
    if response.status_code == 200:
        data = response.json()
        if not data.get("authenticated", True):
            print("✅ Initial auth status: Not authenticated (correct)")
        else:
            print("❌ Initial auth status: Unexpected authentication")
    else:
        print(f"❌ Auth status check failed: {response.status_code}")
    
    # Test 2: Test invalid login
    print("\n2. Testing invalid login...")
    response = session.post(f"{base_url}/api/login", 
                           json={"username": "invalid", "password": "wrong"})
    if response.status_code == 401:
        print("✅ Invalid login correctly rejected")
    else:
        print(f"❌ Invalid login should return 401, got {response.status_code}")
    
    # Test 3: Test valid login
    print("\n3. Testing valid login...")
    response = session.post(f"{base_url}/api/login", 
                           json={"username": "admin", "password": "password123"})
    if response.status_code == 200:
        data = response.json()
        if data.get("message") == "Login successful":
            print("✅ Valid login successful")
        else:
            print(f"❌ Unexpected login response: {data}")
    else:
        print(f"❌ Valid login failed: {response.status_code}")
    
    # Test 4: Check authentication status after login
    print("\n4. Testing auth status after login...")
    response = session.get(f"{base_url}/api/auth/status")
    if response.status_code == 200:
        data = response.json()
        if data.get("authenticated", False) and data.get("user", {}).get("username") == "admin":
            print("✅ Auth status after login: Authenticated as admin")
        else:
            print(f"❌ Unexpected auth status: {data}")
    else:
        print(f"❌ Auth status check failed: {response.status_code}")
    
    # Test 5: Test protected API endpoint
    print("\n5. Testing protected API endpoint...")
    response = session.post(f"{base_url}/api/iris", 
                           json={"test": "data"})
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Protected API successful: {data.get('species', 'No species')}")
    else:
        print(f"❌ Protected API failed: {response.status_code}")
    
    # Test 6: Test logout
    print("\n6. Testing logout...")
    response = session.post(f"{base_url}/api/logout")
    if response.status_code == 200:
        print("✅ Logout successful")
    else:
        print(f"❌ Logout failed: {response.status_code}")
    
    # Test 7: Test protected API after logout
    print("\n7. Testing protected API after logout...")
    response = session.post(f"{base_url}/api/iris", 
                           json={"test": "data"})
    if response.status_code == 401:
        print("✅ Protected API correctly requires authentication after logout")
    else:
        print(f"❌ Protected API should return 401 after logout, got {response.status_code}")
    
    print("\n" + "=" * 40)
    print("🎯 Login functionality test completed!")

if __name__ == "__main__":
    test_login_functionality()