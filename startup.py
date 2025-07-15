#!/usr/bin/env python3
"""
Startup script for AI Pump Control System
"""

import sys
import os
import subprocess
import time

def check_requirements():
    """Check if required packages are installed"""
    print("🔍 Checking requirements...")
    
    try:
        import pandas
        import numpy
        import sklearn
        import matplotlib
        import fastapi
        print("✅ Basic requirements satisfied")
        return True
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def run_tests():
    """Run system tests"""
    print("🧪 Running system tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def start_api_server():
    """Start the FastAPI server"""
    print("🚀 Starting API server...")
    
    try:
        subprocess.run([sys.executable, "edge_deployment.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def main():
    """Main startup function"""
    print("🏭 AI Pump Control System - Startup")
    print("=" * 50)
    
    # Check if requirements are satisfied
    if not check_requirements():
        response = input("Would you like to install requirements? (y/n): ")
        if response.lower() == 'y':
            if not install_requirements():
                print("❌ Failed to install requirements. Exiting...")
                return
        else:
            print("❌ Requirements not satisfied. Exiting...")
            return
    
    # Check if synthetic data exists
    if not os.path.exists('Synthetic_Pump_System_Data.csv'):
        print("❌ Synthetic data file not found!")
        print("Please ensure 'Synthetic_Pump_System_Data.csv' is in the current directory")
        return
    
    # Run tests
    print("\n" + "=" * 50)
    if not run_tests():
        print("❌ Some tests failed. Please check the output above.")
        response = input("Would you like to continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Start server
    print("\n" + "=" * 50)
    print("🎯 System ready! Starting API server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔧 System Status: http://localhost:8000/status")
    print("💡 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    time.sleep(2)  # Brief pause
    start_api_server()

if __name__ == "__main__":
    main()
