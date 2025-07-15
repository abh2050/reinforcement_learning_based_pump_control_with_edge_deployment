![](https://cdn.vev.design/cdn-cgi/image/f=auto,q=82/private/4VWziEfAhZXU3WGXfH8aAwJf2cJ3/image/ep1Jnw1n7M.jpg)
# AI-Driven Pump Control System

[![CI/CD Pipeline](https://github.com/abh2050/reinforcement_learning_based_pump_control_with_edge_deployment/workflows/AI%20Pump%20Control%20System%20CI/CD/badge.svg)](https://github.com/abh2050/reinforcement_learning_based_pump_control_with_edge_deployment/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![MQTT](https://img.shields.io/badge/MQTT-enabled-orange.svg)](https://mqtt.org/)

## 🚀 Overview

This project demonstrates an advanced AI-driven closed-loop control system for pump stations, combining first-principles physics modeling with machine learning and reinforcement learning techniques. The system is designed to optimize pump operations while minimizing energy consumption and maintaining target flow rates and pressures.

**Key Features:**
- **Hybrid Control Model**: Combines fluid dynamics equations with gradient boosted trees
- **Reinforcement Learning**: PPO agent learns optimal control policies
- **Edge Deployment**: FastAPI REST API and MQTT interface for real-time control
- **Industrial-Grade**: Built with Rockwell Automation principles and best practices

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Control System                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ First-Principles │  │ Machine Learning│  │ Reinforcement   │ │
│  │ Physics Model   │  │ (Gradient Boost)│  │ Learning (PPO)  │ │
│  │ (Fluid Dynamics)│  │                 │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│           │                     │                     │        │
│           └─────────────────────┼─────────────────────┘        │
│                                 │                              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Hybrid Control Model                          │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                 │                              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                Edge Deployment                              │ │
│  │  ┌─────────────┐              ┌─────────────┐             │ │
│  │  │ FastAPI     │              │ MQTT        │             │ │
│  │  │ REST API    │              │ Interface   │             │ │
│  │  └─────────────┘              └─────────────┘             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                 │                              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                Physical System                              │ │
│  │     Pump ──► Valve ──► Flow Sensor ──► Pressure Sensor    │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.8+**: Core programming language
- **Machine Learning**: scikit-learn, XGBoost
- **Reinforcement Learning**: Stable-Baselines3, Gymnasium
- **Web Framework**: FastAPI with automatic OpenAPI documentation
- **Communication**: MQTT (paho-mqtt) for IoT integration
- **Data Processing**: NumPy, Pandas, Matplotlib

### Deployment & Operations
- **Containerization**: Docker, Docker Compose
- **Message Broker**: Eclipse Mosquitto MQTT
- **Caching**: Redis for session management
- **Database**: InfluxDB for time-series data
- **Monitoring**: Grafana dashboards, Prometheus metrics
- **CI/CD**: GitHub Actions, automated testing

### Edge Computing
- **Raspberry Pi**: ARM-compatible deployment
- **NVIDIA Jetson**: GPU-accelerated inference
- **Industrial PCs**: Harsh environment support
- **Cloud Integration**: Azure IoT Hub, AWS IoT Core

## 📁 Project Structure

```
ai-pump-control/
├── 📋 Core System Files
│   ├── hybrid_control_model.py    # Hybrid physics + ML model
│   ├── rl_agent.py                # Reinforcement learning agent
│   ├── edge_deployment.py         # FastAPI production server
│   └── config.py                  # Configuration management
│
├── 🔧 Deployment & Operations
│   ├── deploy.sh                  # Production deployment script
│   ├── Dockerfile                 # Container configuration
│   ├── docker-compose.yml         # Multi-service orchestration
│   └── requirements.txt           # Python dependencies
│
├── 🧪 Testing & Validation
│   ├── test_system.py             # Comprehensive test suite
│   ├── demo.py                    # System demonstration
│   └── startup.py                 # System initialization
│
├── 📊 Data & Models
│   ├── Synthetic_Pump_System_Data.csv    # Training data
│   └── models/                    # Trained model storage
│       ├── hybrid_model.joblib
│       └── ppo_pump_control.zip
│
├── 🐳 Container Configuration
│   ├── mqtt-config/
│   │   └── mosquitto.conf         # MQTT broker settings
│   ├── monitoring/
│   │   ├── dashboards/            # Grafana dashboards
│   │   └── provisioning/          # Monitoring setup
│   └── logs/                      # Application logs
│
├── 🔄 CI/CD Pipeline
│   └── .github/
│       └── workflows/
│           └── ci-cd.yml          # GitHub Actions workflow
│
└── 📚 Documentation
    ├── README.md                  # This file
    ├── PROJECT_SUMMARY.md         # Technical overview
    ├── TRAINING_RESULTS.md        # Model performance
    └── DEPLOYMENT.md              # Deployment guide
```

## 📋 Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Option 1: Automated Deployment (Recommended)

```bash
chmod +x deploy.sh
./deploy.sh
```

This interactive script provides multiple deployment options:
1. **Docker Compose** - Full production environment with monitoring
2. **Standalone Python** - Direct Python deployment
3. **Development Mode** - Quick demo and testing
4. **Edge Device** - Optimized for Raspberry Pi/Jetson Nano

### Option 2: Docker Compose (Production)

```bash
docker-compose up -d
```

This will start:
- AI Pump Control API (port 8000)
- MQTT Broker (port 1883)
- Redis Cache (port 6379)
- InfluxDB (port 8086)
- Grafana Monitoring (port 3000)

### Option 3: Manual Setup

#### 1. Train the Hybrid Control Model

```bash
python hybrid_control_model.py
```

#### 2. Train the Reinforcement Learning Agent

```bash
python rl_agent.py
```

#### 3. Deploy the Edge Interface

```bash
python edge_deployment.py
```

### 4. Access the System

- **API Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`
- **MQTT Broker**: `localhost:1883`
- **Monitoring Dashboard**: `http://localhost:3000` (admin/admin)

## 📊 System Components

### 1. First-Principles Physics Model (`FluidDynamicsModel`)

Models the pump system using fundamental fluid dynamics equations:

- **Flow Rate Calculation**: Based on pump speed and valve position
- **Pressure Calculation**: Using simplified Bernoulli equation
- **Physical Constraints**: Respects system limitations

```python
# Example usage
physics_model = FluidDynamicsModel()
response = physics_model.predict_system_response(1800, 0.7)
print(f"Flow: {response['flow_rate_lpm']:.2f} L/min")
print(f"Pressure: {response['pressure_psi']:.2f} PSI")
```

### 2. Machine Learning Model (`MLControlModel`)

Gradient Boosted Trees model that learns from historical data:

- **Feature Engineering**: Includes polynomial and interaction terms
- **Dual Output**: Predicts both flow rate and pressure
- **Standardization**: Proper scaling for better performance

```python
# Example usage
ml_model = MLControlModel()
ml_model.train(training_data)
prediction = ml_model.predict(1800, 0.7)
```

### 3. Hybrid Control Model (`HybridControlModel`)

Combines physics and ML models with configurable weights:

- **Flexible Weighting**: Adjust physics vs. ML influence
- **Fallback Capability**: Uses physics if ML unavailable
- **Unified Interface**: Single prediction method

```python
# Example usage
hybrid_model = HybridControlModel(physics_weight=0.3, ml_weight=0.7)
hybrid_model.train_ml_component(data)
prediction = hybrid_model.predict(1800, 0.7)
```

### 4. Reinforcement Learning Agent (`RLAgent`)

PPO agent that learns optimal control policies:

- **Custom Environment**: Gym-compliant pump control environment
- **Reward Function**: Optimizes flow/pressure targets with efficiency
- **Real-time Control**: Provides optimal actions for current state

```python
# Example usage
agent = RLAgent(hybrid_model, reward_function)
agent.train(total_timesteps=50000)
optimal_action = agent.get_optimal_action(current_state)
```

#### Training Configuration
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Policy Network**: Multi-Layer Perceptron (MLP)
- **Learning Rate**: 3e-4
- **Batch Size**: 64
- **Training Steps**: 2,048 per update
- **Epochs per Update**: 10
- **Discount Factor (γ)**: 0.99
- **GAE Lambda**: 0.95
- **Clip Range**: 0.2

#### Action Space
- **Pump Speed**: 1,500-2,200 RPM (normalized to [-1, 1])
- **Valve Position**: 0.4-1.0 (normalized to [-1, 1])

#### Observation Space
- **Flow Rate**: 0-300 L/min (normalized)
- **Pressure**: 0-8 PSI (normalized)
- **Pump Speed**: Current pump speed (normalized)
- **Valve Position**: Current valve position (normalized)

### 5. Edge Deployment (`edge_deployment.py`)

Production-ready API for real-time control:

- **FastAPI Server**: High-performance REST API
- **MQTT Integration**: Real-time communication
- **Background Monitoring**: Continuous system health checks
- **Historical Data**: Automatic data logging

## 🔧 API Endpoints

### Control Endpoints

- `POST /control`: Send control commands
- `GET /optimize`: Get optimal control settings using RL
- `POST /start`: Start the system
- `POST /stop`: Stop the system

### Monitoring Endpoints

- `GET /status`: Get current system status
- `GET /health`: Health check
- `GET /history`: Historical data
- `POST /predict`: Predict system response

### MQTT Topics

- `pump_system/sensors`: Sensor data
- `pump_system/control`: Control commands
- `pump_system/status`: System status
- `pump_system/predictions`: Model predictions
- `pump_system/alerts`: System alerts

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

The project includes automated CI/CD pipeline that:

- **Multi-Python Testing**: Tests across Python 3.8, 3.9, and 3.10
- **Automated Testing**: Runs comprehensive test suite on every push
- **Docker Integration**: Builds and tests Docker containers
- **Deployment Artifacts**: Creates production-ready deployment packages

### Test Coverage

- **Unit Tests**: Individual component testing
- **Integration Tests**: Full system testing
- **API Tests**: Endpoint validation
- **Performance Tests**: Load and stress testing

### Deployment Targets

- **Development**: Automatic testing on pull requests
- **Staging**: Automated deployment to staging environment
- **Production**: Tagged releases with deployment artifacts
- **Edge Devices**: Optimized packages for Raspberry Pi/Jetson

### Monitoring & Alerting

- **System Health**: Automated health checks
- **Performance Metrics**: Real-time monitoring
- **Error Tracking**: Comprehensive logging
- **Alerting**: MQTT-based alert system

## 🎯 Reward Function

The reward function optimizes multiple objectives:

```python
reward = -0.6 * (flow_error/target_flow)² 
        - 0.4 * (pressure_error/target_pressure)²
        - 0.1 * (pump_speed/max_speed)²
        - 0.1 * stability_penalty
```

**Components:**
- **Flow Tracking**: Maintains target flow rate (200 L/min)
- **Pressure Control**: Maintains target pressure (4.0 PSI)
- **Energy Efficiency**: Minimizes pump energy consumption
- **Stability**: Reduces system oscillations

## 🐳 Docker Deployment

### Container Architecture

The system uses a multi-service Docker architecture:

```yaml
Services:
├── ai-pump-control     # Main application (Port 8000)
├── mqtt-broker         # Eclipse Mosquitto (Port 1883)
├── redis               # Cache & session storage (Port 6379)
├── influxdb           # Time-series database (Port 8086)
└── monitoring         # Grafana dashboard (Port 3000)
```

### Quick Docker Start

```bash
# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f ai-pump-control

# Stop services
docker-compose down
```

### Individual Container Build

```bash
# Build the main application
docker build -t ai-pump-control .

# Run with port mapping
docker run -p 8000:8000 ai-pump-control

# Run with environment variables
docker run -e PYTHONPATH=/app -p 8000:8000 ai-pump-control
```

### Production Configuration

```bash
# Production deployment with resource limits
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Health check
curl http://localhost:8000/health

# Monitor container resources
docker stats
```

### Container Features

- **Multi-stage Build**: Optimized image size
- **Security**: Non-root user execution
- **Health Checks**: Automated container monitoring
- **Resource Limits**: Memory and CPU constraints
- **Persistent Storage**: Data volumes for models and logs

## 🛠️ Production Deployment

### Automated Deployment Script

The `deploy.sh` script provides interactive deployment options:

```bash
chmod +x deploy.sh
./deploy.sh
```

#### Deployment Options

1. **Docker Compose (Recommended)**
   - Full production environment
   - Automatic service orchestration
   - Monitoring and alerting
   - Persistent data storage

2. **Standalone Python**
   - Direct Python deployment
   - Minimal resource requirements
   - Quick setup for testing

3. **Development Mode**
   - Interactive demo
   - System validation
   - Performance testing

4. **Edge Device Optimization**
   - Raspberry Pi/Jetson Nano support
   - Resource-optimized configuration
   - Systemd service integration

### System Requirements

#### Minimum Requirements
- **CPU**: 2 cores, 1.5 GHz
- **RAM**: 2GB minimum, 4GB recommended
- **Storage**: 1GB free space
- **Network**: Ethernet/WiFi for MQTT
- **OS**: Linux, macOS, Windows (Docker)

#### Recommended Hardware
- **Raspberry Pi 4**: 4GB+ RAM model
- **NVIDIA Jetson Nano**: For ML acceleration
- **Industrial PC**: For harsh environments
- **Cloud Instance**: t3.medium or equivalent

### Environment Variables

```bash
# API Configuration
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000

# MQTT Configuration
MQTT_BROKER=localhost
MQTT_PORT=1883

# Model Configuration
MODEL_PATH=./models/
TRAINING_DATA_PATH=./Synthetic_Pump_System_Data.csv

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/system.log
```

### Production Monitoring

- **Health Checks**: `/health` endpoint
- **Metrics**: Prometheus-compatible metrics
- **Logging**: Structured JSON logging
- **Alerting**: MQTT-based alerts
- **Dashboard**: Grafana visualization

## 🔬 Performance Metrics

### Training Results

The RL agent was successfully trained with the following performance metrics:

#### Training Performance
- **Total Timesteps**: 51,200 (completed in 64 seconds)
- **Training Speed**: 790 FPS
- **Mean Reward (last 100 steps)**: -0.8661
- **Convergence**: Stable training with consistent improvement

#### Model Performance Metrics
- **KL Divergence**: 0.0047 (indicates stable policy updates)
- **Clip Fraction**: 0.0818 (8.18% of updates were clipped)
- **Entropy Loss**: -1.45 (policy exploration level)
- **Value Loss**: 90.5 (critic network performance)
- **Policy Gradient Loss**: -0.00238 (actor network updates)

#### Evaluation Results (5 episodes)
- **Mean Episode Reward**: -632.02 ± 7.08
- **Episode Length**: 1,000 steps (full episodes)
- **Flow Error**: 63.97 ± 4.02 L/min (68% accuracy vs 200 L/min target)
- **Pressure Error**: 0.75 ± 0.08 PSI (81% accuracy vs 4.0 PSI target)

#### Performance Interpretation
✅ **Stable Training**: Low KL divergence and consistent rewards indicate stable learning  
✅ **Pressure Control**: Excellent pressure tracking (81% accuracy)  
⚠️ **Flow Control**: Moderate flow tracking (68% accuracy) - room for improvement  
✅ **Convergence**: Model successfully learned control policies  
✅ **Repeatability**: Low standard deviation shows consistent performance  

#### System Capabilities
- **Control Accuracy**: Flow and pressure tracking with quantified performance
- **Energy Efficiency**: Power consumption optimization through reward function
- **System Stability**: Oscillation minimization with stability penalty
- **Response Time**: Sub-second control loop performance (790 FPS training speed)

## 🚀 Deployment Options

### Edge Devices

- **Raspberry Pi 4**: Suitable for smaller installations
- **NVIDIA Jetson Nano**: For ML inference acceleration
- **Industrial PCs**: For harsh industrial environments
- **Rockwell CompactLogix**: Integration with existing systems

### Cloud Integration

- **AWS IoT Core**: For cloud-based monitoring
- **Azure IoT Hub**: Microsoft ecosystem integration
- **FactoryTalk Cloud**: Rockwell's industrial cloud platform

## 📈 Future Enhancements

### Planned Features

1. **Multi-Pump Systems**: Extend to multiple pump coordination
2. **Predictive Maintenance**: Failure prediction and prevention
3. **Advanced RL**: Multi-agent reinforcement learning
4. **Digital Twin**: Full system simulation
5. **Cybersecurity**: Industrial security implementation

### Research Opportunities

- **Federated Learning**: Distributed model training
- **Explainable AI**: Model interpretability for operators
- **Edge AI**: On-device training and adaptation
- **Quantum Computing**: Optimization algorithms

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-pump-control.git
cd ai-pump-control

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

1. **Fluid Dynamics**: White, F.M. "Fluid Mechanics" 8th Edition
2. **Control Systems**: Ogata, K. "Modern Control Engineering" 5th Edition
3. **Reinforcement Learning**: Sutton, R.S. & Barto, A.G. "Reinforcement Learning: An Introduction"
4. **Industrial AI**: Rockwell Automation White Papers on AI for Control

## 📞 Contact

**Author**: Abhishek Shah  
---

*This project is designed to showcase advanced AI/ML capabilities in industrial control systems, specifically aligned with Rockwell Automation's technology stack and industry focus.*
