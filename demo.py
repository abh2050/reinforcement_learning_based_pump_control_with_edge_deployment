#!/usr/bin/env python3
"""
Complete Demo Script for AI Pump Control System
Demonstrates all major features of the system
"""

import requests
import json
import time
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def test_api_endpoint(endpoint, method='GET', data=None, description=""):
    """Test an API endpoint and display results"""
    url = f"http://localhost:8000{endpoint}"
    
    try:
        if method == 'GET':
            response = requests.get(url)
        elif method == 'POST':
            response = requests.post(url, json=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {description}")
            print(f"   Response: {json.dumps(result, indent=2)}")
            return result
        else:
            print(f"❌ {description}")
            print(f"   Status: {response.status_code}, Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ {description}")
        print(f"   Error: {e}")
        return None

def demonstrate_system():
    """Demonstrate the complete AI pump control system"""
    print("🏭 AI Pump Control System - Live Demo")
    print("=" * 60)
    
    # Test basic endpoints
    print("\n1. 🔍 System Health Check")
    test_api_endpoint("/health", description="Health check")
    
    print("\n2. 📊 System Status")
    test_api_endpoint("/status", description="System status check")
    
    print("\n3. 🚀 Start System")
    test_api_endpoint("/start", method='POST', description="Start pump system")
    
    # Wait for system to initialize
    time.sleep(2)
    
    print("\n4. 🔮 Hybrid Model Predictions")
    test_scenarios = [
        {"pump_speed_rpm": 1700, "valve_position": 0.6},
        {"pump_speed_rpm": 1800, "valve_position": 0.7},
        {"pump_speed_rpm": 1900, "valve_position": 0.8},
        {"pump_speed_rpm": 2000, "valve_position": 0.9},
    ]
    
    results = []
    for scenario in test_scenarios:
        result = test_api_endpoint("/predict", method='POST', data=scenario, 
                                  description=f"Predict for pump={scenario['pump_speed_rpm']}rpm, valve={scenario['valve_position']}")
        if result:
            results.append({
                'pump_speed': scenario['pump_speed_rpm'],
                'valve_position': scenario['valve_position'],
                'flow_rate': result['flow_rate_lpm'],
                'pressure': result['pressure_psi'],
                'reward': result['predicted_reward']
            })
    
    print("\n5. 🎮 Control Commands")
    control_commands = [
        {"pump_speed_rpm": 1750, "valve_position": 0.65, "mode": "manual"},
        {"pump_speed_rpm": 1850, "valve_position": 0.75, "mode": "auto"},
    ]
    
    for cmd in control_commands:
        test_api_endpoint("/control", method='POST', data=cmd, 
                         description=f"Send control command: {cmd}")
        time.sleep(1)
    
    print("\n6. 📈 Historical Data")
    test_api_endpoint("/history?limit=5", description="Get historical data")
    
    print("\n7. 📡 MQTT Topics")
    test_api_endpoint("/mqtt/topics", description="Get MQTT topics")
    
    print("\n8. 🛑 Stop System")
    test_api_endpoint("/stop", method='POST', description="Stop pump system")
    
    # Create visualization if we have results
    if results:
        create_demo_visualization(results)
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed successfully!")
    print("📖 For full API documentation, visit: http://localhost:8000/docs")
    print("🔧 For system monitoring, visit: http://localhost:8000/status")

def create_demo_visualization(results):
    """Create a visualization of the demo results"""
    print("\n9. 📊 Creating Demo Visualization...")
    
    try:
        df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Flow rate vs pump speed
        axes[0, 0].scatter(df['pump_speed'], df['flow_rate'], color='blue', alpha=0.7)
        axes[0, 0].set_xlabel('Pump Speed (RPM)')
        axes[0, 0].set_ylabel('Flow Rate (L/min)')
        axes[0, 0].set_title('Flow Rate vs Pump Speed')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Pressure vs valve position
        axes[0, 1].scatter(df['valve_position'], df['pressure'], color='red', alpha=0.7)
        axes[0, 1].set_xlabel('Valve Position')
        axes[0, 1].set_ylabel('Pressure (PSI)')
        axes[0, 1].set_title('Pressure vs Valve Position')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Reward vs pump speed
        axes[1, 0].scatter(df['pump_speed'], df['reward'], color='green', alpha=0.7)
        axes[1, 0].set_xlabel('Pump Speed (RPM)')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].set_title('Reward vs Pump Speed')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 3D surface plot (Flow, Pressure, Reward)
        axes[1, 1].scatter(df['flow_rate'], df['pressure'], c=df['reward'], cmap='viridis', alpha=0.7)
        axes[1, 1].set_xlabel('Flow Rate (L/min)')
        axes[1, 1].set_ylabel('Pressure (PSI)')
        axes[1, 1].set_title('Performance Map (colored by reward)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('demo_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Demo visualization saved as 'demo_results.png'")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")

def check_server_status():
    """Check if the server is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Main demo function"""
    print("🔍 Checking server status...")
    
    if not check_server_status():
        print("❌ Server is not running!")
        print("Please start the server with: python edge_deployment.py")
        return
    
    print("✅ Server is running!")
    
    try:
        demonstrate_system()
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")

if __name__ == "__main__":
    main()
