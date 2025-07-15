"""
Reinforcement Learning Agent for Pump Control System
Implements PPO agent for learning optimal control policies
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import logging
from hybrid_control_model import HybridControlModel, RewardFunction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PumpControlEnvironment(gym.Env):
    """
    Gym environment for pump control system
    """
    
    def __init__(self, 
                 hybrid_model: HybridControlModel,
                 reward_function: RewardFunction,
                 max_steps: int = 1000,
                 noise_std: float = 0.02):
        super().__init__()
        
        self.hybrid_model = hybrid_model
        self.reward_function = reward_function
        self.max_steps = max_steps
        self.noise_std = noise_std
        
        # Define action space: [pump_speed_rpm, valve_position]
        # Normalize to [-1, 1] for better RL training
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(2,), 
            dtype=np.float32
        )
        
        # Define observation space: [flow_rate, pressure, pump_speed, valve_position]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 1500.0, 0.0], dtype=np.float32),
            high=np.array([300.0, 8.0, 2200.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action bounds for denormalization
        self.pump_speed_min = 1500.0
        self.pump_speed_max = 2200.0
        self.valve_position_min = 0.4
        self.valve_position_max = 1.0
        
        # State variables
        self.current_step = 0
        self.current_state = None
        self.prev_flow = None
        self.prev_pressure = None
        self.episode_rewards = []
        self.episode_actions = []
        
        # Performance tracking
        self.flow_history = []
        self.pressure_history = []
        self.reward_history = []
        
    def _denormalize_action(self, action: np.ndarray) -> Tuple[float, float]:
        """
        Denormalize action from [-1, 1] to actual control values
        """
        # Denormalize pump speed
        pump_speed = self.pump_speed_min + (action[0] + 1) * 0.5 * (self.pump_speed_max - self.pump_speed_min)
        
        # Denormalize valve position
        valve_position = self.valve_position_min + (action[1] + 1) * 0.5 * (self.valve_position_max - self.valve_position_min)
        
        # Clip to bounds
        pump_speed = np.clip(pump_speed, self.pump_speed_min, self.pump_speed_max)
        valve_position = np.clip(valve_position, self.valve_position_min, self.valve_position_max)
        
        return pump_speed, valve_position
    
    def _normalize_observation(self, flow: float, pressure: float, pump_speed: float, valve_position: float) -> np.ndarray:
        """
        Normalize observation to standard ranges
        """
        # Normalize flow rate (0-300 L/min)
        flow_norm = flow / 300.0
        
        # Normalize pressure (0-8 PSI)
        pressure_norm = pressure / 8.0
        
        # Normalize pump speed (1500-2200 RPM)
        pump_speed_norm = (pump_speed - self.pump_speed_min) / (self.pump_speed_max - self.pump_speed_min)
        
        # Valve position already normalized (0-1)
        valve_position_norm = valve_position
        
        return np.array([flow_norm, pressure_norm, pump_speed_norm, valve_position_norm], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment
        """
        super().reset(seed=seed)
        
        # Reset state variables
        self.current_step = 0
        self.prev_flow = None
        self.prev_pressure = None
        self.episode_rewards = []
        self.episode_actions = []
        
        # Initialize with random but reasonable values
        initial_pump_speed = np.random.uniform(1600, 2000)
        initial_valve_position = np.random.uniform(0.5, 0.9)
        
        # Get initial system response
        system_response = self.hybrid_model.predict(initial_pump_speed, initial_valve_position)
        
        # Add noise to simulate real-world conditions
        flow_with_noise = system_response['flow_rate_lpm'] + np.random.normal(0, self.noise_std * 200)
        pressure_with_noise = system_response['pressure_psi'] + np.random.normal(0, self.noise_std * 4)
        
        # Normalize observation
        self.current_state = self._normalize_observation(
            flow_with_noise, pressure_with_noise, initial_pump_speed, initial_valve_position
        )
        
        return self.current_state, {}
    
    def step(self, action: np.ndarray):
        """
        Execute one step in the environment
        """
        # Denormalize action
        pump_speed, valve_position = self._denormalize_action(action)
        
        # Get system response from hybrid model
        system_response = self.hybrid_model.predict(pump_speed, valve_position)
        
        # Add noise to simulate real-world conditions
        flow_rate = system_response['flow_rate_lpm'] + np.random.normal(0, self.noise_std * 200)
        pressure = system_response['pressure_psi'] + np.random.normal(0, self.noise_std * 4)
        
        # Clip to physical limits
        flow_rate = np.clip(flow_rate, 0, 300)
        pressure = np.clip(pressure, 0, 8)
        
        # Calculate reward
        reward = self.reward_function.calculate_reward(
            flow_rate, pressure, pump_speed, valve_position,
            self.prev_flow, self.prev_pressure
        )
        
        # Update state
        self.current_state = self._normalize_observation(flow_rate, pressure, pump_speed, valve_position)
        
        # Store for next step
        self.prev_flow = flow_rate
        self.prev_pressure = pressure
        
        # Store episode data
        self.episode_rewards.append(reward)
        self.episode_actions.append([pump_speed, valve_position])
        
        # Store performance tracking
        self.flow_history.append(flow_rate)
        self.pressure_history.append(pressure)
        self.reward_history.append(reward)
        
        # Check if episode is done
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False
        
        return self.current_state, reward, done, truncated, {
            'flow_rate': flow_rate,
            'pressure': pressure,
            'pump_speed': pump_speed,
            'valve_position': valve_position
        }
    
    def render(self, mode='human'):
        """
        Render the environment (optional)
        """
        if len(self.flow_history) > 0:
            print(f"Step {self.current_step}: Flow={self.flow_history[-1]:.2f}, Pressure={self.pressure_history[-1]:.2f}, Reward={self.reward_history[-1]:.4f}")

class TrainingCallback(BaseCallback):
    """
    Custom callback for training monitoring
    """
    
    def __init__(self, eval_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log training progress
        if self.num_timesteps % self.eval_freq == 0:
            logger.info(f"Timestep: {self.num_timesteps}")
            
        return True
    
    def _on_rollout_end(self) -> None:
        # Log episode statistics
        if hasattr(self.training_env, 'get_attr'):
            reward_histories = self.training_env.get_attr('reward_history')
            if reward_histories and len(reward_histories[0]) > 0:
                mean_reward = np.mean([np.mean(hist[-100:]) for hist in reward_histories if len(hist) > 0])
                logger.info(f"Mean reward (last 100 steps): {mean_reward:.4f}")

class RLAgent:
    """
    Reinforcement Learning Agent for pump control
    """
    
    def __init__(self, 
                 hybrid_model: HybridControlModel,
                 reward_function: RewardFunction,
                 model_save_path: str = 'ppo_pump_control'):
        
        self.hybrid_model = hybrid_model
        self.reward_function = reward_function
        self.model_save_path = model_save_path
        
        # Create environment
        self.env = PumpControlEnvironment(hybrid_model, reward_function)
        
        # Initialize PPO agent
        self.model = None
        self.training_callback = TrainingCallback()
        
        # Training metrics
        self.training_rewards = []
        self.training_steps = []
        
    def create_model(self, 
                    policy: str = 'MlpPolicy',
                    learning_rate: float = 3e-4,
                    n_steps: int = 2048,
                    batch_size: int = 64,
                    n_epochs: int = 10,
                    gamma: float = 0.99,
                    gae_lambda: float = 0.95,
                    clip_range: float = 0.2,
                    verbose: int = 1):
        """
        Create PPO model
        """
        
        # Create vectorized environment
        vec_env = DummyVecEnv([lambda: self.env])
        
        # Create PPO model
        self.model = PPO(
            policy=policy,
            env=vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            verbose=verbose,
            tensorboard_log="./tensorboard_logs/"
        )
        
        logger.info("PPO model created successfully")
    
    def train(self, total_timesteps: int = 100000):
        """
        Train the RL agent
        """
        if self.model is None:
            self.create_model()
        
        logger.info(f"Starting training for {total_timesteps} timesteps...")
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.training_callback,
            progress_bar=True
        )
        
        logger.info("Training completed!")
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the trained agent
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        episode_rewards = []
        episode_lengths = []
        flow_errors = []
        pressure_errors = []
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_flow_errors = []
            episode_pressure_errors = []
            
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Calculate errors
                flow_error = abs(info['flow_rate'] - self.reward_function.target_flow)
                pressure_error = abs(info['pressure'] - self.reward_function.target_pressure)
                
                episode_flow_errors.append(flow_error)
                episode_pressure_errors.append(pressure_error)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            flow_errors.extend(episode_flow_errors)
            pressure_errors.extend(episode_pressure_errors)
        
        # Calculate metrics
        metrics = {
            'mean_episode_reward': np.mean(episode_rewards),
            'std_episode_reward': np.std(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'mean_flow_error': np.mean(flow_errors),
            'mean_pressure_error': np.mean(pressure_errors),
            'flow_error_std': np.std(flow_errors),
            'pressure_error_std': np.std(pressure_errors)
        }
        
        return metrics
    
    def save_model(self, filepath: str = None):
        """
        Save the trained model
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        if filepath is None:
            filepath = self.model_save_path
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = None):
        """
        Load a trained model
        """
        if filepath is None:
            filepath = self.model_save_path
        
        # Create vectorized environment
        vec_env = DummyVecEnv([lambda: self.env])
        
        self.model = PPO.load(filepath, env=vec_env)
        logger.info(f"Model loaded from {filepath}")
    
    def get_optimal_action(self, current_state: np.ndarray) -> Tuple[float, float]:
        """
        Get optimal action for current state
        """
        if self.model is None:
            raise ValueError("Model must be trained before getting actions")
        
        action, _ = self.model.predict(current_state, deterministic=True)
        pump_speed, valve_position = self.env._denormalize_action(action)
        
        return pump_speed, valve_position
    
    def visualize_performance(self, save_path: str = None):
        """
        Visualize agent performance
        """
        if len(self.env.flow_history) == 0:
            logger.warning("No performance data available for visualization")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Flow rate over time
        axes[0, 0].plot(self.env.flow_history, label='Actual Flow', alpha=0.7)
        axes[0, 0].axhline(y=self.reward_function.target_flow, color='r', linestyle='--', label='Target Flow')
        axes[0, 0].set_title('Flow Rate Over Time')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Flow Rate (L/min)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Pressure over time
        axes[0, 1].plot(self.env.pressure_history, label='Actual Pressure', alpha=0.7, color='orange')
        axes[0, 1].axhline(y=self.reward_function.target_pressure, color='r', linestyle='--', label='Target Pressure')
        axes[0, 1].set_title('Pressure Over Time')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Pressure (PSI)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Reward over time
        axes[1, 0].plot(self.env.reward_history, label='Reward', alpha=0.7, color='green')
        axes[1, 0].set_title('Reward Over Time')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Error distribution
        flow_errors = [abs(f - self.reward_function.target_flow) for f in self.env.flow_history]
        pressure_errors = [abs(p - self.reward_function.target_pressure) for p in self.env.pressure_history]
        
        axes[1, 1].hist(flow_errors, bins=30, alpha=0.5, label='Flow Error', density=True)
        axes[1, 1].hist(pressure_errors, bins=30, alpha=0.5, label='Pressure Error', density=True)
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].set_xlabel('Error')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

if __name__ == "__main__":
    # Load synthetic data and train hybrid model
    data = pd.read_csv('Synthetic_Pump_System_Data.csv')
    
    # Initialize hybrid model
    hybrid_model = HybridControlModel()
    hybrid_model.train_ml_component(data)
    
    # Initialize reward function
    reward_function = RewardFunction(target_flow=200.0, target_pressure=4.0)
    
    # Initialize RL agent
    agent = RLAgent(hybrid_model, reward_function)
    
    # Train the agent
    agent.train(total_timesteps=50000)
    
    # Evaluate the agent
    metrics = agent.evaluate(n_episodes=5)
    
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save the model
    agent.save_model('ppo_pump_control')
    
    # Visualize performance
    agent.visualize_performance('rl_performance.png')
    
    print("\nRL training completed successfully!")
