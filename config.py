"""
Configuration file for AI Pump Control System
"""

import os
from typing import Dict, Any

# System Configuration
SYSTEM_CONFIG = {
    # Physical system parameters
    "pump_speed_min": 1500.0,  # RPM
    "pump_speed_max": 2200.0,  # RPM
    "valve_position_min": 0.0,  # Normalized (0-1)
    "valve_position_max": 1.0,  # Normalized (0-1)
    "max_flow_rate": 300.0,  # L/min
    "max_pressure": 8.0,  # PSI
    
    # Target setpoints
    "target_flow_rate": 200.0,  # L/min
    "target_pressure": 4.0,  # PSI
    
    # Control parameters
    "control_loop_frequency": 1.0,  # Hz
    "max_control_change_rate": 0.1,  # Maximum change per second
    
    # Model parameters
    "hybrid_model_physics_weight": 0.3,
    "hybrid_model_ml_weight": 0.7,
    "model_noise_std": 0.02,
    
    # Reward function weights
    "reward_flow_weight": 0.6,
    "reward_pressure_weight": 0.4,
    "reward_efficiency_weight": 0.1,
    "reward_stability_weight": 0.1,
    
    # RL training parameters
    "rl_training_timesteps": 50000,
    "rl_evaluation_episodes": 10,
    "rl_environment_max_steps": 1000,
    
    # API configuration
    "api_host": "0.0.0.0",
    "api_port": 8000,
    "api_reload": True,
    
    # MQTT configuration
    "mqtt_broker_host": "localhost",
    "mqtt_broker_port": 1883,
    "mqtt_keepalive": 60,
    
    # Data storage
    "max_historical_data_size": 1000,
    "data_logging_enabled": True,
    "data_log_file": "system_data.csv",
    
    # Safety limits
    "safety_min_flow": 50.0,  # L/min
    "safety_max_flow": 280.0,  # L/min
    "safety_min_pressure": 1.0,  # PSI
    "safety_max_pressure": 7.0,  # PSI
    
    # Monitoring
    "health_check_interval": 5.0,  # seconds
    "alert_threshold_flow_deviation": 50.0,  # L/min
    "alert_threshold_pressure_deviation": 2.0,  # PSI
    
    # File paths
    "hybrid_model_file": "hybrid_control_model.pkl",
    "rl_model_file": "ppo_pump_control",
    "synthetic_data_file": "Synthetic_Pump_System_Data.csv",
    "performance_plot_file": "system_performance.png",
    "tensorboard_log_dir": "./tensorboard_logs/",
}

# MQTT Topics
MQTT_TOPICS = {
    "sensor_data": "pump_system/sensors",
    "control_commands": "pump_system/control",
    "system_status": "pump_system/status",
    "predictions": "pump_system/predictions",
    "alerts": "pump_system/alerts",
    "health": "pump_system/health",
}

# Environment variables override
def get_config() -> Dict[str, Any]:
    """
    Get configuration with environment variable overrides
    """
    config = SYSTEM_CONFIG.copy()
    
    # Override with environment variables if available
    env_mappings = {
        "API_HOST": "api_host",
        "API_PORT": "api_port",
        "MQTT_BROKER_HOST": "mqtt_broker_host",
        "MQTT_BROKER_PORT": "mqtt_broker_port",
        "TARGET_FLOW_RATE": "target_flow_rate",
        "TARGET_PRESSURE": "target_pressure",
        "RL_TRAINING_TIMESTEPS": "rl_training_timesteps",
    }
    
    for env_var, config_key in env_mappings.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            
            # Convert to appropriate type
            if config_key in ["api_port", "mqtt_broker_port", "rl_training_timesteps"]:
                config[config_key] = int(value)
            elif config_key in ["target_flow_rate", "target_pressure"]:
                config[config_key] = float(value)
            else:
                config[config_key] = value
    
    return config

def get_mqtt_topics() -> Dict[str, str]:
    """Get MQTT topics configuration"""
    return MQTT_TOPICS.copy()

# System states
SYSTEM_STATES = {
    "STOPPED": "stopped",
    "STARTING": "starting",
    "RUNNING": "running",
    "STOPPING": "stopping",
    "ERROR": "error",
    "MAINTENANCE": "maintenance",
}

CONTROL_MODES = {
    "MANUAL": "manual",
    "AUTO": "auto",
    "RL": "rl",
    "EMERGENCY": "emergency",
}

HEALTH_STATUS = {
    "OK": "ok",
    "WARNING": "warning",
    "ERROR": "error",
    "CRITICAL": "critical",
}

ALERT_LEVELS = {
    "INFO": "info",
    "WARNING": "warning",
    "ERROR": "error",
    "CRITICAL": "critical",
}

# Validation functions
def validate_pump_speed(speed: float) -> bool:
    """Validate pump speed is within limits"""
    config = get_config()
    return config["pump_speed_min"] <= speed <= config["pump_speed_max"]

def validate_valve_position(position: float) -> bool:
    """Validate valve position is within limits"""
    config = get_config()
    return config["valve_position_min"] <= position <= config["valve_position_max"]

def validate_flow_rate(flow: float) -> bool:
    """Validate flow rate is within safety limits"""
    config = get_config()
    return config["safety_min_flow"] <= flow <= config["safety_max_flow"]

def validate_pressure(pressure: float) -> bool:
    """Validate pressure is within safety limits"""
    config = get_config()
    return config["safety_min_pressure"] <= pressure <= config["safety_max_pressure"]

# Utility functions
def get_system_limits() -> Dict[str, Dict[str, float]]:
    """Get system operational limits"""
    config = get_config()
    
    return {
        "pump_speed": {
            "min": config["pump_speed_min"],
            "max": config["pump_speed_max"],
        },
        "valve_position": {
            "min": config["valve_position_min"],
            "max": config["valve_position_max"],
        },
        "flow_rate": {
            "min": config["safety_min_flow"],
            "max": config["safety_max_flow"],
            "target": config["target_flow_rate"],
        },
        "pressure": {
            "min": config["safety_min_pressure"],
            "max": config["safety_max_pressure"],
            "target": config["target_pressure"],
        },
    }

if __name__ == "__main__":
    # Print configuration for debugging
    config = get_config()
    
    print("🔧 System Configuration:")
    print("=" * 50)
    
    for key, value in config.items():
        print(f"{key:<30}: {value}")
    
    print("\n📡 MQTT Topics:")
    print("=" * 50)
    
    topics = get_mqtt_topics()
    for key, value in topics.items():
        print(f"{key:<20}: {value}")
    
    print("\n🎯 System Limits:")
    print("=" * 50)
    
    limits = get_system_limits()
    for component, values in limits.items():
        print(f"{component}:")
        for param, value in values.items():
            print(f"  {param}: {value}")
        print()
