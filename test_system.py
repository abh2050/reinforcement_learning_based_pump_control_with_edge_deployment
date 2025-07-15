"""
Test script to verify the AI Pump Control System functionality
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os

# Add the current directory to the path
sys.path.append('.')

def test_hybrid_model():
    """Test the hybrid control model"""
    print("🧪 Testing Hybrid Control Model...")
    
    try:
        from hybrid_control_model import HybridControlModel, RewardFunction
        
        # Load synthetic data
        if not os.path.exists('Synthetic_Pump_System_Data.csv'):
            print("❌ Synthetic data file not found!")
            return False
        
        data = pd.read_csv('Synthetic_Pump_System_Data.csv')
        print(f"✅ Loaded {len(data)} data points")
        
        # Initialize hybrid model
        hybrid_model = HybridControlModel()
        
        # Train ML component
        metrics = hybrid_model.train_ml_component(data)
        print(f"✅ ML training completed. Flow R²: {metrics['flow_r2']:.4f}, Pressure R²: {metrics['pressure_r2']:.4f}")
        
        # Test prediction
        test_pump_speed = 1800
        test_valve_position = 0.7
        
        prediction = hybrid_model.predict(test_pump_speed, test_valve_position)
        print(f"✅ Prediction: Flow={prediction['flow_rate_lpm']:.2f} L/min, Pressure={prediction['pressure_psi']:.2f} PSI")
        
        # Test reward function
        reward_func = RewardFunction()
        reward = reward_func.calculate_reward(
            prediction['flow_rate_lpm'],
            prediction['pressure_psi'],
            test_pump_speed,
            test_valve_position
        )
        print(f"✅ Reward: {reward:.4f}")
        
        # Save model
        hybrid_model.save_model('hybrid_control_model.pkl')
        print("✅ Model saved successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in hybrid model test: {e}")
        return False

def test_rl_agent():
    """Test the RL agent (minimal test due to training time)"""
    print("\n🤖 Testing RL Agent...")
    
    try:
        from hybrid_control_model import HybridControlModel, RewardFunction
        from rl_agent import RLAgent, PumpControlEnvironment
        
        # Load trained hybrid model
        hybrid_model = HybridControlModel()
        if os.path.exists('hybrid_control_model.pkl'):
            hybrid_model.load_model('hybrid_control_model.pkl')
        else:
            print("❌ Trained hybrid model not found!")
            return False
        
        # Initialize reward function
        reward_function = RewardFunction(target_flow=200.0, target_pressure=4.0)
        
        # Test environment
        env = PumpControlEnvironment(hybrid_model, reward_function, max_steps=10)
        
        # Test environment reset
        state, _ = env.reset()
        print(f"✅ Environment reset. Initial state shape: {state.shape}")
        
        # Test environment step
        action = np.array([0.0, 0.0])  # Neutral action
        next_state, reward, done, _, info = env.step(action)
        print(f"✅ Environment step. Reward: {reward:.4f}, Flow: {info['flow_rate']:.2f}, Pressure: {info['pressure']:.2f}")
        
        # Initialize RL agent (don't train for testing)
        agent = RLAgent(hybrid_model, reward_function)
        print("✅ RL Agent initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in RL agent test: {e}")
        return False

def test_edge_deployment():
    """Test the edge deployment API"""
    print("\n🌐 Testing Edge Deployment API...")
    
    try:
        import asyncio
        from edge_deployment import load_models, system_state
        
        # Test model loading
        asyncio.run(load_models())
        print("✅ Models loaded successfully")
        
        # Test system state
        print(f"✅ System state: {system_state['system_health']}")
        
        print("✅ Edge deployment components ready")
        print("ℹ️  To test the full API, run: python edge_deployment.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in edge deployment test: {e}")
        return False

def create_visualization():
    """Create a visualization of the system performance"""
    print("\n📊 Creating System Visualization...")
    
    try:
        # Load synthetic data
        data = pd.read_csv('Synthetic_Pump_System_Data.csv')
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Flow rate over time
        axes[0, 0].plot(data['timestamp'], data['flow_rate_lpm'], alpha=0.7)
        axes[0, 0].axhline(y=200, color='r', linestyle='--', label='Target Flow')
        axes[0, 0].set_title('Flow Rate Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Flow Rate (L/min)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Pressure over time
        axes[0, 1].plot(data['timestamp'], data['pressure_psi'], alpha=0.7, color='orange')
        axes[0, 1].axhline(y=4.0, color='r', linestyle='--', label='Target Pressure')
        axes[0, 1].set_title('Pressure Over Time')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Pressure (PSI)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Control inputs
        axes[1, 0].plot(data['timestamp'], data['pump_speed_rpm'], label='Pump Speed', alpha=0.7)
        axes[1, 0].set_title('Pump Speed Over Time')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Pump Speed (RPM)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Valve position
        axes[1, 1].plot(data['timestamp'], data['valve_position'], label='Valve Position', alpha=0.7, color='green')
        axes[1, 1].set_title('Valve Position Over Time')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Valve Position (0-1)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('system_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization created and saved as 'system_performance.png'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 AI Pump Control System - Test Suite")
    print("=" * 50)
    
    # Track test results
    test_results = []
    
    # Run tests
    test_results.append(("Hybrid Model", test_hybrid_model()))
    test_results.append(("RL Agent", test_rl_agent()))
    test_results.append(("Edge Deployment", test_edge_deployment()))
    test_results.append(("Visualization", create_visualization()))
    
    # Print results
    print("\n📋 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        print("🎉 All tests passed! System is ready for deployment.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        
    print("\n📚 Next Steps:")
    print("1. Run 'python edge_deployment.py' to start the API server")
    print("2. Visit http://localhost:8000/docs for API documentation")
    print("3. Use the API endpoints to control the pump system")
    
    return passed == len(test_results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
