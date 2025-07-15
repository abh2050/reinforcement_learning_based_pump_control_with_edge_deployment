#!/bin/bash

# AI Pump Control System - Production Deployment Script
# Compatible with Rockwell Automation industrial environments

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Banner
echo -e "${BLUE}"
echo "================================================================"
echo "    AI Pump Control System - Production Deployment"
echo "    Compatible with Rockwell Automation Environments"
echo "================================================================"
echo -e "${NC}"

# Check system requirements
log "Checking system requirements..."

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ "$PYTHON_VERSION" < "3.8" ]]; then
        error "Python 3.8+ is required. Found: $PYTHON_VERSION"
        exit 1
    fi
    log "Python version: $PYTHON_VERSION ✓"
else
    error "Python 3 is not installed"
    exit 1
fi

# Check Docker (optional)
if command -v docker &> /dev/null; then
    log "Docker available: $(docker --version) ✓"
    DOCKER_AVAILABLE=true
else
    warn "Docker not available - will use Python deployment"
    DOCKER_AVAILABLE=false
fi

# Check available memory
MEMORY_GB=$(free -g | awk 'NR==2{print $2}')
if [ "$MEMORY_GB" -lt 2 ]; then
    warn "Recommended minimum 2GB RAM. Available: ${MEMORY_GB}GB"
fi

# Check available disk space
DISK_SPACE_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$DISK_SPACE_GB" -lt 1 ]; then
    error "Insufficient disk space. Need at least 1GB. Available: ${DISK_SPACE_GB}GB"
    exit 1
fi

log "System requirements check complete ✓"

# Deployment options
echo ""
echo "Select deployment option:"
echo "1) Docker Compose (Recommended for production)"
echo "2) Standalone Python"
echo "3) Development mode"
echo "4) Edge device optimization"

read -p "Enter your choice (1-4): " DEPLOYMENT_CHOICE

case $DEPLOYMENT_CHOICE in
    1)
        if [ "$DOCKER_AVAILABLE" = false ]; then
            error "Docker is not available. Please install Docker first."
            exit 1
        fi
        
        log "Starting Docker Compose deployment..."
        
        # Create necessary directories
        mkdir -p models logs mqtt-data mqtt-logs monitoring/dashboards monitoring/provisioning
        
        # Generate synthetic data if not exists
        if [ ! -f "Synthetic_Pump_System_Data.csv" ]; then
            log "Generating synthetic data..."
            python3 -c "
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n_samples = 1000

data = {
    'timestamp': [datetime.now() + timedelta(minutes=i) for i in range(n_samples)],
    'pump_speed_rpm': np.random.uniform(1500, 2200, n_samples),
    'valve_position': np.random.uniform(0.4, 1.0, n_samples),
    'flow_rate_lpm': np.random.uniform(100, 300, n_samples),
    'pressure_psi': np.random.uniform(1, 8, n_samples),
    'reward': np.random.uniform(-10, -1, n_samples)
}

df = pd.DataFrame(data)
df.to_csv('Synthetic_Pump_System_Data.csv', index=False)
print('Synthetic data generated successfully')
            "
        fi
        
        # Start services
        docker-compose up -d
        
        log "Docker services started successfully!"
        log "API available at: http://localhost:8000"
        log "MQTT broker at: localhost:1883"
        log "Monitoring dashboard: http://localhost:3000 (admin/admin)"
        ;;
        
    2)
        log "Starting standalone Python deployment..."
        
        # Install dependencies
        if [ -f "requirements.txt" ]; then
            log "Installing Python dependencies..."
            pip3 install -r requirements.txt
        else
            error "requirements.txt not found"
            exit 1
        fi
        
        # Generate synthetic data if not exists
        if [ ! -f "Synthetic_Pump_System_Data.csv" ]; then
            log "Generating synthetic data..."
            python3 -c "
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n_samples = 1000

data = {
    'timestamp': [datetime.now() + timedelta(minutes=i) for i in range(n_samples)],
    'pump_speed_rpm': np.random.uniform(1500, 2200, n_samples),
    'valve_position': np.random.uniform(0.4, 1.0, n_samples),
    'flow_rate_lpm': np.random.uniform(100, 300, n_samples),
    'pressure_psi': np.random.uniform(1, 8, n_samples),
    'reward': np.random.uniform(-10, -1, n_samples)
}

df = pd.DataFrame(data)
df.to_csv('Synthetic_Pump_System_Data.csv', index=False)
print('Synthetic data generated successfully')
            "
        fi
        
        # Run system tests
        log "Running system tests..."
        python3 test_system.py
        
        # Start the system
        log "Starting AI Pump Control System..."
        python3 edge_deployment.py &
        
        log "System started successfully!"
        log "API available at: http://localhost:8000"
        log "PID: $!"
        ;;
        
    3)
        log "Starting development mode..."
        
        # Install dependencies
        pip3 install -r requirements.txt
        
        # Generate synthetic data
        python3 -c "
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n_samples = 1000

data = {
    'timestamp': [datetime.now() + timedelta(minutes=i) for i in range(n_samples)],
    'pump_speed_rpm': np.random.uniform(1500, 2200, n_samples),
    'valve_position': np.random.uniform(0.4, 1.0, n_samples),
    'flow_rate_lpm': np.random.uniform(100, 300, n_samples),
    'pressure_psi': np.random.uniform(1, 8, n_samples),
    'reward': np.random.uniform(-10, -1, n_samples)
}

df = pd.DataFrame(data)
df.to_csv('Synthetic_Pump_System_Data.csv', index=False)
print('Synthetic data generated successfully')
        "
        
        # Run demo
        log "Running system demo..."
        python3 demo.py
        ;;
        
    4)
        log "Optimizing for edge device deployment..."
        
        # Lightweight installation
        pip3 install --no-cache-dir -r requirements.txt
        
        # Generate minimal synthetic data
        python3 -c "
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n_samples = 100  # Reduced for edge devices

data = {
    'timestamp': [datetime.now() + timedelta(minutes=i) for i in range(n_samples)],
    'pump_speed_rpm': np.random.uniform(1500, 2200, n_samples),
    'valve_position': np.random.uniform(0.4, 1.0, n_samples),
    'flow_rate_lpm': np.random.uniform(100, 300, n_samples),
    'pressure_psi': np.random.uniform(1, 8, n_samples),
    'reward': np.random.uniform(-10, -1, n_samples)
}

df = pd.DataFrame(data)
df.to_csv('Synthetic_Pump_System_Data.csv', index=False)
print('Minimal synthetic data generated for edge deployment')
        "
        
        # Create systemd service for auto-start
        cat > /tmp/ai-pump-control.service << EOF
[Unit]
Description=AI Pump Control System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PYTHONPATH=$(pwd)
ExecStart=$(which python3) edge_deployment.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        
        log "Edge optimization complete!"
        log "Install systemd service with: sudo cp /tmp/ai-pump-control.service /etc/systemd/system/"
        log "Enable auto-start with: sudo systemctl enable ai-pump-control.service"
        ;;
        
    *)
        error "Invalid choice. Please select 1-4."
        exit 1
        ;;
esac

# Final status check
echo ""
log "Deployment complete!"
log "System Status:"

if [ "$DEPLOYMENT_CHOICE" = "1" ]; then
    log "- Docker containers: $(docker-compose ps --services | wc -l) services"
    log "- Main API: http://localhost:8000/docs"
    log "- MQTT Broker: localhost:1883"
    log "- Monitoring: http://localhost:3000"
elif [ "$DEPLOYMENT_CHOICE" = "2" ]; then
    log "- Python API: http://localhost:8000/docs"
    log "- Health check: curl http://localhost:8000/health"
fi

log "For troubleshooting, check logs in ./logs/ directory"
log "For Rockwell integration, see PROJECT_SUMMARY.md"

echo ""
echo -e "${GREEN}🎉 AI Pump Control System deployed successfully!${NC}"
echo -e "${BLUE}Ready for industrial pump control and optimization.${NC}"
