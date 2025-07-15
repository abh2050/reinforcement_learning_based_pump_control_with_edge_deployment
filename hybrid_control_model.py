"""
Hybrid Control Model for Pump Station System
Combines first-principles physics with machine learning for control optimization
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
from typing import Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FluidDynamicsModel:
    """
    First-principles model based on fluid dynamics equations
    """
    
    def __init__(self):
        # Physical constants and system parameters
        self.rho = 1000  # Water density (kg/m³)
        self.pipe_diameter = 0.1  # Pipe diameter (m)
        self.pipe_area = np.pi * (self.pipe_diameter/2)**2
        self.friction_factor = 0.02  # Darcy friction factor
        self.pump_efficiency = 0.85  # Pump efficiency
        
        # Operational limits
        self.max_flow_rate = 300  # L/min
        self.max_pressure = 8  # PSI
        self.max_pump_speed = 2200  # RPM
        
    def calculate_theoretical_flow(self, pump_speed_rpm: float, valve_position: float) -> float:
        """
        Calculate theoretical flow rate based on pump speed and valve position
        """
        # Normalize pump speed (0-1)
        speed_normalized = min(pump_speed_rpm / self.max_pump_speed, 1.0)
        
        # Flow rate is proportional to pump speed and valve opening
        # Q = k * N * sqrt(valve_position)
        k = 0.8  # Proportionality constant
        flow_rate = k * speed_normalized * np.sqrt(valve_position) * self.max_flow_rate
        
        return max(0, flow_rate)
    
    def calculate_theoretical_pressure(self, flow_rate_lpm: float, valve_position: float) -> float:
        """
        Calculate theoretical pressure based on flow rate and valve position
        """
        # Convert L/min to m³/s
        flow_rate_m3s = flow_rate_lpm / 60000
        
        # Pressure drop across valve (simplified Bernoulli equation)
        # ΔP = 0.5 * ρ * v² * (1/Cv²)
        velocity = flow_rate_m3s / self.pipe_area
        valve_resistance = 1.0 / (valve_position + 0.1)  # Avoid division by zero
        
        pressure_drop = 0.5 * self.rho * velocity**2 * valve_resistance
        
        # Convert Pa to PSI (1 Pa = 0.000145038 PSI)
        pressure_psi = pressure_drop * 0.000145038
        
        return min(pressure_psi, self.max_pressure)
    
    def predict_system_response(self, pump_speed_rpm: float, valve_position: float) -> Dict[str, float]:
        """
        Predict system response using first-principles
        """
        flow_rate = self.calculate_theoretical_flow(pump_speed_rpm, valve_position)
        pressure = self.calculate_theoretical_pressure(flow_rate, valve_position)
        
        return {
            'flow_rate_lpm': flow_rate,
            'pressure_psi': pressure
        }

class MLControlModel:
    """
    Machine Learning model for control optimization
    """
    
    def __init__(self):
        self.flow_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.pressure_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.scaler_X = StandardScaler()
        self.scaler_y_flow = StandardScaler()
        self.scaler_y_pressure = StandardScaler()
        
        self.is_trained = False
    
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature matrix for ML model
        """
        features = ['pump_speed_rpm', 'valve_position']
        
        # Add engineered features
        X = data[features].copy()
        X['pump_speed_squared'] = X['pump_speed_rpm'] ** 2
        X['valve_position_squared'] = X['valve_position'] ** 2
        X['pump_valve_interaction'] = X['pump_speed_rpm'] * X['valve_position']
        X['pump_speed_normalized'] = X['pump_speed_rpm'] / 2200  # Normalize by max RPM
        
        return X
    
    def train(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Train the ML models on historical data
        """
        logger.info("Training ML control models...")
        
        # Prepare features
        X = self.prepare_features(data)
        y_flow = data['flow_rate_lpm'].values.reshape(-1, 1)
        y_pressure = data['pressure_psi'].values.reshape(-1, 1)
        
        # Split data
        X_train, X_test, y_flow_train, y_flow_test, y_pressure_train, y_pressure_test = train_test_split(
            X, y_flow, y_pressure, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # Scale targets
        y_flow_train_scaled = self.scaler_y_flow.fit_transform(y_flow_train)
        y_pressure_train_scaled = self.scaler_y_pressure.fit_transform(y_pressure_train)
        
        # Train models
        self.flow_model.fit(X_train_scaled, y_flow_train_scaled.ravel())
        self.pressure_model.fit(X_train_scaled, y_pressure_train_scaled.ravel())
        
        # Evaluate models
        y_flow_pred = self.scaler_y_flow.inverse_transform(
            self.flow_model.predict(X_test_scaled).reshape(-1, 1)
        )
        y_pressure_pred = self.scaler_y_pressure.inverse_transform(
            self.pressure_model.predict(X_test_scaled).reshape(-1, 1)
        )
        
        # Calculate metrics
        flow_mse = mean_squared_error(y_flow_test, y_flow_pred)
        flow_r2 = r2_score(y_flow_test, y_flow_pred)
        pressure_mse = mean_squared_error(y_pressure_test, y_pressure_pred)
        pressure_r2 = r2_score(y_pressure_test, y_pressure_pred)
        
        self.is_trained = True
        
        metrics = {
            'flow_mse': flow_mse,
            'flow_r2': flow_r2,
            'pressure_mse': pressure_mse,
            'pressure_r2': pressure_r2
        }
        
        logger.info(f"Training completed. Flow R²: {flow_r2:.4f}, Pressure R²: {pressure_r2:.4f}")
        
        return metrics
    
    def predict(self, pump_speed_rpm: float, valve_position: float) -> Dict[str, float]:
        """
        Predict system response using ML model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare features
        feature_dict = {
            'pump_speed_rpm': pump_speed_rpm,
            'valve_position': valve_position
        }
        X = pd.DataFrame([feature_dict])
        X_features = self.prepare_features(X)
        X_scaled = self.scaler_X.transform(X_features)
        
        # Make predictions
        flow_pred_scaled = self.flow_model.predict(X_scaled)
        pressure_pred_scaled = self.pressure_model.predict(X_scaled)
        
        # Inverse transform
        flow_pred = self.scaler_y_flow.inverse_transform(flow_pred_scaled.reshape(-1, 1))[0, 0]
        pressure_pred = self.scaler_y_pressure.inverse_transform(pressure_pred_scaled.reshape(-1, 1))[0, 0]
        
        return {
            'flow_rate_lpm': flow_pred,
            'pressure_psi': pressure_pred
        }
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'flow_model': self.flow_model,
            'pressure_model': self.pressure_model,
            'scaler_X': self.scaler_X,
            'scaler_y_flow': self.scaler_y_flow,
            'scaler_y_pressure': self.scaler_y_pressure,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.flow_model = model_data['flow_model']
        self.pressure_model = model_data['pressure_model']
        self.scaler_X = model_data['scaler_X']
        self.scaler_y_flow = model_data['scaler_y_flow']
        self.scaler_y_pressure = model_data['scaler_y_pressure']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")

class HybridControlModel:
    """
    Hybrid control model combining first-principles and ML approaches
    """
    
    def __init__(self, physics_weight: float = 0.3, ml_weight: float = 0.7):
        self.physics_model = FluidDynamicsModel()
        self.ml_model = MLControlModel()
        self.physics_weight = physics_weight
        self.ml_weight = ml_weight
        
        # Ensure weights sum to 1
        total_weight = physics_weight + ml_weight
        self.physics_weight = physics_weight / total_weight
        self.ml_weight = ml_weight / total_weight
    
    def train_ml_component(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train the ML component of the hybrid model"""
        return self.ml_model.train(data)
    
    def predict(self, pump_speed_rpm: float, valve_position: float) -> Dict[str, float]:
        """
        Make hybrid prediction combining physics and ML models
        """
        # Get physics-based prediction
        physics_pred = self.physics_model.predict_system_response(pump_speed_rpm, valve_position)
        
        # Get ML-based prediction (if trained)
        if self.ml_model.is_trained:
            ml_pred = self.ml_model.predict(pump_speed_rpm, valve_position)
            
            # Combine predictions
            hybrid_pred = {
                'flow_rate_lpm': (
                    self.physics_weight * physics_pred['flow_rate_lpm'] +
                    self.ml_weight * ml_pred['flow_rate_lpm']
                ),
                'pressure_psi': (
                    self.physics_weight * physics_pred['pressure_psi'] +
                    self.ml_weight * ml_pred['pressure_psi']
                )
            }
        else:
            # Use only physics model if ML not trained
            hybrid_pred = physics_pred
        
        return hybrid_pred
    
    def save_model(self, filepath: str):
        """Save the hybrid model"""
        self.ml_model.save_model(filepath)
    
    def load_model(self, filepath: str):
        """Load the hybrid model"""
        self.ml_model.load_model(filepath)

# Reward function for reinforcement learning
class RewardFunction:
    """
    Reward function for RL optimization
    """
    
    def __init__(self, target_flow: float = 200.0, target_pressure: float = 4.0):
        self.target_flow = target_flow
        self.target_pressure = target_pressure
        
        # Penalty weights
        self.flow_weight = 0.6
        self.pressure_weight = 0.4
        self.efficiency_weight = 0.1
        self.stability_weight = 0.1
    
    def calculate_reward(self, 
                        flow_rate: float, 
                        pressure: float, 
                        pump_speed: float,
                        valve_position: float,
                        prev_flow: float = None,
                        prev_pressure: float = None) -> float:
        """
        Calculate reward based on system performance
        """
        # Flow rate penalty (squared error)
        flow_error = abs(flow_rate - self.target_flow)
        flow_penalty = -self.flow_weight * (flow_error / self.target_flow) ** 2
        
        # Pressure penalty (squared error)
        pressure_error = abs(pressure - self.target_pressure)
        pressure_penalty = -self.pressure_weight * (pressure_error / self.target_pressure) ** 2
        
        # Efficiency penalty (minimize energy consumption)
        efficiency_penalty = -self.efficiency_weight * (pump_speed / 2200) ** 2
        
        # Stability penalty (minimize oscillations)
        stability_penalty = 0
        if prev_flow is not None and prev_pressure is not None:
            flow_change = abs(flow_rate - prev_flow)
            pressure_change = abs(pressure - prev_pressure)
            stability_penalty = -self.stability_weight * (flow_change + pressure_change)
        
        # Total reward
        total_reward = flow_penalty + pressure_penalty + efficiency_penalty + stability_penalty
        
        return total_reward

if __name__ == "__main__":
    # Load synthetic data
    data = pd.read_csv('Synthetic_Pump_System_Data.csv')
    
    # Initialize and train hybrid model
    hybrid_model = HybridControlModel()
    metrics = hybrid_model.train_ml_component(data)
    
    print("Training Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Test prediction
    test_pump_speed = 1800
    test_valve_position = 0.7
    
    prediction = hybrid_model.predict(test_pump_speed, test_valve_position)
    print(f"\nPrediction for pump_speed={test_pump_speed}, valve_position={test_valve_position}:")
    print(f"Flow rate: {prediction['flow_rate_lpm']:.2f} L/min")
    print(f"Pressure: {prediction['pressure_psi']:.2f} PSI")
    
    # Test reward function
    reward_func = RewardFunction()
    reward = reward_func.calculate_reward(
        prediction['flow_rate_lpm'],
        prediction['pressure_psi'],
        test_pump_speed,
        test_valve_position
    )
    print(f"Reward: {reward:.4f}")
    
    # Save model
    hybrid_model.save_model('hybrid_control_model.pkl')
    print("\nModel saved successfully!")
