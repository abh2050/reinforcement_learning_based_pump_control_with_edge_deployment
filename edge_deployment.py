"""
Edge Deployment Interface for Pump Control System
FastAPI REST API and MQTT interface for real-time control
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
import os
import threading
import time

# MQTT imports
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    logging.warning("MQTT client not available. Install paho-mqtt for MQTT functionality.")

# Import our control models
from hybrid_control_model import HybridControlModel, RewardFunction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class SensorReading(BaseModel):
    pump_speed_rpm: float = Field(..., ge=1500, le=2200, description="Pump speed in RPM")
    valve_position: float = Field(..., ge=0.0, le=1.0, description="Valve position (0-1)")
    flow_rate_lpm: Optional[float] = Field(None, ge=0, le=300, description="Flow rate in L/min")
    pressure_psi: Optional[float] = Field(None, ge=0, le=8, description="Pressure in PSI")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)

class ControlCommand(BaseModel):
    pump_speed_rpm: float = Field(..., ge=1500, le=2200, description="Target pump speed in RPM")
    valve_position: float = Field(..., ge=0.0, le=1.0, description="Target valve position (0-1)")
    mode: str = Field(default="manual", description="Control mode: manual, auto, or rl")

class SystemStatus(BaseModel):
    is_running: bool
    current_flow: float
    current_pressure: float
    current_pump_speed: float
    current_valve_position: float
    control_mode: str
    last_update: datetime
    system_health: str

class PredictionRequest(BaseModel):
    pump_speed_rpm: float = Field(..., ge=1500, le=2200)
    valve_position: float = Field(..., ge=0.0, le=1.0)

class PredictionResponse(BaseModel):
    flow_rate_lpm: float
    pressure_psi: float
    predicted_reward: float
    model_confidence: float
    timestamp: datetime

# Global variables for system state
system_state = {
    "is_running": False,
    "current_flow": 0.0,
    "current_pressure": 0.0,
    "current_pump_speed": 1700.0,
    "current_valve_position": 0.7,
    "control_mode": "manual",
    "last_update": datetime.now(),
    "system_health": "ok"
}

# Data storage for historical data
historical_data = []
max_history_size = 1000

# Control models
hybrid_model = None
reward_function = None
rl_agent = None

# MQTT client
mqtt_client = None
mqtt_connected = False

class MQTTHandler:
    """
    MQTT handler for real-time communication
    """
    
    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client = None
        self.connected = False
        
        # MQTT topics
        self.topics = {
            "sensor_data": "pump_system/sensors",
            "control_commands": "pump_system/control",
            "system_status": "pump_system/status",
            "predictions": "pump_system/predictions",
            "alerts": "pump_system/alerts"
        }
        
        if MQTT_AVAILABLE:
            self.setup_client()
    
    def setup_client(self):
        """Setup MQTT client"""
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
    
    def on_connect(self, client, userdata, flags, rc):
        """Callback for MQTT connection"""
        if rc == 0:
            self.connected = True
            logger.info("Connected to MQTT broker")
            
            # Subscribe to control commands
            client.subscribe(self.topics["control_commands"])
            logger.info(f"Subscribed to {self.topics['control_commands']}")
        else:
            logger.error(f"Failed to connect to MQTT broker. Return code: {rc}")
    
    def on_message(self, client, userdata, msg):
        """Handle incoming MQTT messages"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            
            if topic == self.topics["control_commands"]:
                self.handle_control_command(payload)
            
        except Exception as e:
            logger.error(f"Error handling MQTT message: {e}")
    
    def on_disconnect(self, client, userdata, rc):
        """Handle MQTT disconnection"""
        self.connected = False
        logger.warning("Disconnected from MQTT broker")
    
    def handle_control_command(self, payload: Dict):
        """Handle control command from MQTT"""
        try:
            command = ControlCommand(**payload)
            
            # Update system state
            system_state["current_pump_speed"] = command.pump_speed_rpm
            system_state["current_valve_position"] = command.valve_position
            system_state["control_mode"] = command.mode
            system_state["last_update"] = datetime.now()
            
            logger.info(f"MQTT control command processed: {command}")
            
        except Exception as e:
            logger.error(f"Error processing MQTT control command: {e}")
    
    def connect(self):
        """Connect to MQTT broker"""
        if not MQTT_AVAILABLE:
            logger.warning("MQTT not available")
            return False
        
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.client and self.connected:
            self.client.loop_stop()
            self.client.disconnect()
    
    def publish_sensor_data(self, data: Dict):
        """Publish sensor data to MQTT"""
        if self.client and self.connected:
            try:
                self.client.publish(self.topics["sensor_data"], json.dumps(data))
            except Exception as e:
                logger.error(f"Failed to publish sensor data: {e}")
    
    def publish_system_status(self, status: Dict):
        """Publish system status to MQTT"""
        if self.client and self.connected:
            try:
                self.client.publish(self.topics["system_status"], json.dumps(status))
            except Exception as e:
                logger.error(f"Failed to publish system status: {e}")
    
    def publish_prediction(self, prediction: Dict):
        """Publish prediction to MQTT"""
        if self.client and self.connected:
            try:
                self.client.publish(self.topics["predictions"], json.dumps(prediction))
            except Exception as e:
                logger.error(f"Failed to publish prediction: {e}")
    
    def publish_alert(self, alert: Dict):
        """Publish alert to MQTT"""
        if self.client and self.connected:
            try:
                self.client.publish(self.topics["alerts"], json.dumps(alert))
            except Exception as e:
                logger.error(f"Failed to publish alert: {e}")

# Initialize MQTT handler
mqtt_handler = MQTTHandler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI
    """
    # Startup
    logger.info("Starting Pump Control System API...")
    
    # Load models
    await load_models()
    
    # Connect to MQTT
    if MQTT_AVAILABLE:
        mqtt_handler.connect()
    
    # Start background tasks
    background_task = asyncio.create_task(background_system_monitor())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Pump Control System API...")
    
    # Disconnect MQTT
    if MQTT_AVAILABLE:
        mqtt_handler.disconnect()
    
    # Cancel background tasks
    background_task.cancel()
    try:
        await background_task
    except asyncio.CancelledError:
        pass

# Create FastAPI app
app = FastAPI(
    title="Pump Control System API",
    description="AI-driven closed-loop control system for pump stations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def load_models():
    """Load the control models"""
    global hybrid_model, reward_function, rl_agent
    
    try:
        # Load hybrid model
        hybrid_model = HybridControlModel()
        
        # Try to load saved model
        if os.path.exists('hybrid_control_model.pkl'):
            hybrid_model.load_model('hybrid_control_model.pkl')
            logger.info("Hybrid model loaded successfully")
        else:
            # Train on synthetic data if no saved model
            if os.path.exists('Synthetic_Pump_System_Data.csv'):
                data = pd.read_csv('Synthetic_Pump_System_Data.csv')
                hybrid_model.train_ml_component(data)
                hybrid_model.save_model('hybrid_control_model.pkl')
                logger.info("Hybrid model trained and saved")
            else:
                logger.warning("No training data found. Using physics-only model.")
        
        # Initialize reward function
        reward_function = RewardFunction(target_flow=200.0, target_pressure=4.0)
        
        # Try to load RL agent
        try:
            from rl_agent import RLAgent
            rl_agent = RLAgent(hybrid_model, reward_function)
            if os.path.exists('ppo_pump_control.zip'):
                rl_agent.load_model('ppo_pump_control')
                logger.info("RL agent loaded successfully")
            else:
                logger.warning("RL agent model file not found")
                rl_agent = None
        except Exception as e:
            logger.warning(f"RL agent not available: {e}")
            rl_agent = None
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

async def background_system_monitor():
    """Background task to monitor system and publish data"""
    while True:
        try:
            # Simulate system operation
            if system_state["is_running"]:
                await update_system_state()
                await publish_system_data()
            
            # Check system health
            await check_system_health()
            
            # Sleep for 1 second
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error in background system monitor: {e}")
            await asyncio.sleep(5)

async def update_system_state():
    """Update system state based on current control settings"""
    global system_state, historical_data
    
    try:
        # Get current control settings
        pump_speed = system_state["current_pump_speed"]
        valve_position = system_state["current_valve_position"]
        
        # Predict system response
        if hybrid_model:
            prediction = hybrid_model.predict(pump_speed, valve_position)
            
            # Add some noise to simulate real sensor readings
            noise_factor = 0.02
            flow_rate = prediction['flow_rate_lpm'] + np.random.normal(0, noise_factor * 200)
            pressure = prediction['pressure_psi'] + np.random.normal(0, noise_factor * 4)
            
            # Clip to physical limits
            flow_rate = max(0, min(flow_rate, 300))
            pressure = max(0, min(pressure, 8))
            
            # Update system state
            system_state["current_flow"] = flow_rate
            system_state["current_pressure"] = pressure
            system_state["last_update"] = datetime.now()
            
            # Store historical data
            historical_data.append({
                "timestamp": datetime.now(),
                "pump_speed_rpm": pump_speed,
                "valve_position": valve_position,
                "flow_rate_lpm": flow_rate,
                "pressure_psi": pressure,
                "control_mode": system_state["control_mode"]
            })
            
            # Limit historical data size
            if len(historical_data) > max_history_size:
                historical_data.pop(0)
    
    except Exception as e:
        logger.error(f"Error updating system state: {e}")

async def publish_system_data():
    """Publish system data to MQTT"""
    try:
        # Publish sensor data
        sensor_data = {
            "pump_speed_rpm": system_state["current_pump_speed"],
            "valve_position": system_state["current_valve_position"],
            "flow_rate_lpm": system_state["current_flow"],
            "pressure_psi": system_state["current_pressure"],
            "timestamp": system_state["last_update"].isoformat()
        }
        mqtt_handler.publish_sensor_data(sensor_data)
        
        # Publish system status
        status_data = {
            "is_running": system_state["is_running"],
            "current_flow": system_state["current_flow"],
            "current_pressure": system_state["current_pressure"],
            "current_pump_speed": system_state["current_pump_speed"],
            "current_valve_position": system_state["current_valve_position"],
            "control_mode": system_state["control_mode"],
            "last_update": system_state["last_update"].isoformat(),
            "system_health": system_state["system_health"]
        }
        mqtt_handler.publish_system_status(status_data)
        
    except Exception as e:
        logger.error(f"Error publishing system data: {e}")

async def check_system_health():
    """Check system health and generate alerts"""
    try:
        # Check if system is responding
        time_since_update = (datetime.now() - system_state["last_update"]).total_seconds()
        
        if time_since_update > 10:  # No update for 10 seconds
            system_state["system_health"] = "warning"
            
            # Publish alert
            alert = {
                "level": "warning",
                "message": "System not responding",
                "timestamp": datetime.now().isoformat()
            }
            mqtt_handler.publish_alert(alert)
        
        # Check for abnormal values
        if system_state["is_running"]:
            if system_state["current_flow"] < 50 or system_state["current_flow"] > 280:
                system_state["system_health"] = "error"
                
                alert = {
                    "level": "error",
                    "message": f"Abnormal flow rate: {system_state['current_flow']:.2f} L/min",
                    "timestamp": datetime.now().isoformat()
                }
                mqtt_handler.publish_alert(alert)
            
            if system_state["current_pressure"] < 1 or system_state["current_pressure"] > 7:
                system_state["system_health"] = "error"
                
                alert = {
                    "level": "error",
                    "message": f"Abnormal pressure: {system_state['current_pressure']:.2f} PSI",
                    "timestamp": datetime.now().isoformat()
                }
                mqtt_handler.publish_alert(alert)
        
        # Reset health status if no issues
        if system_state["system_health"] in ["warning", "error"] and time_since_update < 5:
            if (50 <= system_state["current_flow"] <= 280 and 
                1 <= system_state["current_pressure"] <= 7):
                system_state["system_health"] = "ok"
        
    except Exception as e:
        logger.error(f"Error checking system health: {e}")

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Pump Control System API",
        "version": "1.0.0",
        "status": "running"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "models_loaded": {
            "hybrid_model": hybrid_model is not None,
            "reward_function": reward_function is not None,
            "rl_agent": rl_agent is not None
        }
    }

@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get current system status"""
    return SystemStatus(**system_state)

@app.post("/control")
async def send_control_command(command: ControlCommand):
    """Send control command to the system"""
    try:
        # Update system state
        system_state["current_pump_speed"] = command.pump_speed_rpm
        system_state["current_valve_position"] = command.valve_position
        system_state["control_mode"] = command.mode
        system_state["last_update"] = datetime.now()
        
        # Publish to MQTT
        mqtt_handler.publish_sensor_data({
            "pump_speed_rpm": command.pump_speed_rpm,
            "valve_position": command.valve_position,
            "control_mode": command.mode,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "message": "Control command sent successfully",
            "command": command.dict()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending control command: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_system_response(request: PredictionRequest):
    """Predict system response for given control inputs"""
    try:
        # Check if models are loaded
        if not hybrid_model:
            # Try to load models if not already loaded
            await load_models()
            
        if not hybrid_model:
            raise HTTPException(status_code=503, detail="Hybrid model not available")
        
        # Make prediction
        prediction = hybrid_model.predict(request.pump_speed_rpm, request.valve_position)
        
        # Calculate reward
        reward = 0.0
        if reward_function:
            reward = reward_function.calculate_reward(
                prediction['flow_rate_lpm'],
                prediction['pressure_psi'],
                request.pump_speed_rpm,
                request.valve_position
            )
        
        # Try to publish prediction to MQTT (optional)
        try:
            prediction_data = {
                "pump_speed_rpm": request.pump_speed_rpm,
                "valve_position": request.valve_position,
                "predicted_flow": prediction['flow_rate_lpm'],
                "predicted_pressure": prediction['pressure_psi'],
                "predicted_reward": reward,
                "timestamp": datetime.now().isoformat()
            }
            mqtt_handler.publish_prediction(prediction_data)
        except Exception as mqtt_error:
            logger.warning(f"Failed to publish MQTT prediction: {mqtt_error}")
        
        return PredictionResponse(
            flow_rate_lpm=prediction['flow_rate_lpm'],
            pressure_psi=prediction['pressure_psi'],
            predicted_reward=reward,
            model_confidence=0.95,  # Placeholder
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.get("/optimize")
async def get_optimal_control():
    """Get optimal control settings using RL agent"""
    try:
        if not rl_agent:
            raise HTTPException(status_code=503, detail="RL agent not available")
        
        # Create current state observation
        current_state = np.array([
            system_state["current_flow"] / 300.0,  # Normalized flow
            system_state["current_pressure"] / 8.0,  # Normalized pressure
            (system_state["current_pump_speed"] - 1500) / 700.0,  # Normalized pump speed
            system_state["current_valve_position"]  # Already normalized
        ], dtype=np.float32)
        
        # Get optimal action
        pump_speed, valve_position = rl_agent.get_optimal_action(current_state)
        
        return {
            "optimal_pump_speed": pump_speed,
            "optimal_valve_position": valve_position,
            "current_state": {
                "flow_rate": system_state["current_flow"],
                "pressure": system_state["current_pressure"],
                "pump_speed": system_state["current_pump_speed"],
                "valve_position": system_state["current_valve_position"]
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting optimal control: {str(e)}")

@app.post("/start")
async def start_system():
    """Start the system"""
    system_state["is_running"] = True
    system_state["last_update"] = datetime.now()
    
    return {
        "success": True,
        "message": "System started successfully"
    }

@app.post("/stop")
async def stop_system():
    """Stop the system"""
    system_state["is_running"] = False
    system_state["last_update"] = datetime.now()
    
    return {
        "success": True,
        "message": "System stopped successfully"
    }

@app.get("/history")
async def get_historical_data(limit: int = 100):
    """Get historical system data"""
    try:
        # Return last 'limit' records
        recent_data = historical_data[-limit:] if len(historical_data) > limit else historical_data
        
        return {
            "data": recent_data,
            "total_records": len(historical_data),
            "returned_records": len(recent_data)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving historical data: {str(e)}")

@app.get("/mqtt/topics")
async def get_mqtt_topics():
    """Get MQTT topics information"""
    if not MQTT_AVAILABLE:
        raise HTTPException(status_code=503, detail="MQTT not available")
    
    return {
        "topics": mqtt_handler.topics,
        "connected": mqtt_handler.connected,
        "broker": {
            "host": mqtt_handler.broker_host,
            "port": mqtt_handler.broker_port
        }
    }

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "edge_deployment:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
