"""
Environment wrapper for ANYmal teacher policy training
Provides a Gym-like interface for RL training
"""
import os
import sys
import numpy as np
from collections import deque

# Add bindings to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, "training", "envs"))
sys.path.insert(0, project_root)

import anymal_env
from training.utils.rewards import compute_reward, get_foot_target_positions


class AnymalTeacherEnv:
    """
    Gym-like wrapper for ANYmal teacher training
    Uses privileged state (212-dim)
    """
    
    def __init__(self, 
                 visualize=False,
                 urdf_path=None,
                 actuator_path=None,
                 max_episode_length=1000,
                 control_dt=0.02):
        """
        Args:
            visualize: Whether to enable visualization
            urdf_path: Path to URDF file
            actuator_path: Path to actuator model file
            max_episode_length: Maximum steps per episode (default 1000 = 20s at 50Hz)
            control_dt: Control timestep (default 0.02s = 50Hz)
        """
        # Set default paths
        if urdf_path is None:
            urdf_path = os.path.join(project_root, "rsc/robot/c100/urdf/anymal_minimal.urdf")
        if actuator_path is None:
            actuator_path = os.path.join(project_root, "rsc/actuator/C100/seaModel_2500.txt")
        
        # Create environment
        self.env = anymal_env.BlindLocomotionC100(
            visualize=visualize,
            instance=0,
            urdf_path=urdf_path,
            actuator_path=actuator_path
        )
        
        self.visualize = visualize
        self.max_episode_length = max_episode_length
        self.control_dt = control_dt
        
        # State dimensions
        self.state_dim = anymal_env.PRIVILEGED_STATE_DIM  # 212
        self.action_dim = anymal_env.ACTION_DIM  # 16
        
        # Tracking variables
        self.episode_length = 0
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.prev_prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.foot_targets_history = deque(maxlen=3)
        
        # Episode statistics
        self.episode_reward = 0.0
        self.episode_info = {}
        
        # Task curriculum parameters (will be set by trainer)
        # Format: task_params = [param1, param2, param3]
        # Terrain type 0 (Hills): param1=roughness, param2=frequency, param3=amplitude
        # Terrain type 1 (Steps): param1=roughness, param2=stepSize, param3=stepHeight
        # Terrain type 2 (Stairs): param1=roughness, param2=stepLength, param3=stepHeight
        # test_c100.cpp uses: terrain=0, params=[0.05, 0.5, 0.5] for stable Hills
        self.current_task_idx = 0
        self.task_params = np.array([0.05, 0.5, 0.5], dtype=np.float32)
        
        # Gym-style space attributes
        self.observation_space = type('obj', (object,), {'shape': (self.state_dim,)})
        self.action_space = type('obj', (object,), {'shape': (self.action_dim,)})
        
    def reset(self, task_idx=None, task_params=None):
        """
        Reset environment and sample new terrain/task
        
        Args:
            task_idx: Terrain type index (0-8), None for random
            task_params: Terrain parameters [p1, p2, p3], None for random
        
        Returns:
            state: Initial privileged state (212-dim)
        """
        # Sample or set task
        if task_idx is None:
            # Random terrain type (0-8)
            # self.current_task_idx = np.random.randint(0, 9)
            self.current_task_idx = 0
        else:
            self.current_task_idx = task_idx
        
        if task_params is None:
            # Use stable parameters based on test_c100.cpp: [0.05, 0.5, 0.5]
            # with small randomization for better generalization
            self.task_params = np.array([
                np.random.uniform(0.03, 0.05),   # Roughness/frequency around 0.05
                np.random.uniform(0.3, 0.5),     # Frequency/step length around 0.5
                np.random.uniform(0.2, 0.5)      # Amplitude/step height around 0.5
            ], dtype=np.float32)
        else:
            self.task_params = task_params
        
        # Update terrain in environment
        # Format: [terrain_type_index, param1, param2, param3]
        task_config = np.concatenate([[self.current_task_idx], self.task_params])
        self.env.updateTask(task_config)
        
        # Reset environment
        self.env.init()
        
        # Sample new command
        self.env.sampleCommand()
        
        # Reset tracking variables
        self.episode_length = 0
        self.episode_reward = 0.0
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.prev_prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.foot_targets_history.clear()
        
        # Initialize foot targets history
        for _ in range(3):
            self.foot_targets_history.append(get_foot_target_positions(self.env))
        
        # Get initial state
        state = self.env.getPrivilegedState().flatten().astype(np.float32)
        
        # Safety: clip extreme values (rare cases from initialization noise)
        state = np.clip(state, -100.0, 100.0)
        
        return state
    
    def step(self, action):
        """
        Execute one step in the environment
        
        Args:
            action: Action vector (16-dim)
        
        Returns:
            next_state: Next privileged state (212-dim)
            reward: Scalar reward
            done: Whether episode is terminated
            info: Dictionary with episode information
        """
        # Convert action to numpy array
        action = np.array(action, dtype=np.float32)
        
        # Apply action and integrate
        self.env.updateAction(action)
        self.env.integrate()
        
        # Update foot targets history
        foot_targets = get_foot_target_positions(self.env)
        self.foot_targets_history.append(foot_targets)
        
        # Get next state
        next_state = self.env.getPrivilegedState().flatten().astype(np.float32)
        
        # Safety: clip extreme values
        next_state = np.clip(next_state, -100.0, 100.0)
        
        # Compute reward
        reward, info = compute_reward(
            self.env, 
            action, 
            self.prev_action,
            self.prev_prev_action,
            list(self.foot_targets_history)
        )
        
        # Update action history
        self.prev_prev_action = self.prev_action.copy()
        self.prev_action = action.copy()
        
        # Check termination
        self.episode_length += 1
        
        # Episode terminates if:
        # 1. Base contact or badly conditioned (from C++)
        # 2. Maximum episode length reached
        done = bool(self.env.isTerminal() or self.episode_length >= self.max_episode_length)
        
        # Add episode tracking info
        info['episode_length'] = self.episode_length
        info['task_idx'] = self.current_task_idx
        info['is_terminal'] = self.env.isTerminal()
        info['is_timeout'] = (self.episode_length >= self.max_episode_length)
        
        # Accumulate episode reward
        self.episode_reward += reward
        
        if done:
            info['episode_reward'] = self.episode_reward
            info['episode_success'] = not self.env.isTerminal()  # Success if not terminated early
        
        return next_state, reward, done, info
    
    def get_state_dim(self):
        """Get state dimension"""
        return self.state_dim
    
    def get_action_dim(self):
        """Get action dimension"""
        return self.action_dim
    
    def close(self):
        """Close environment"""
        # Environment cleanup happens in destructor
        pass
    
    def set_command(self, command):
        """
        Manually set locomotion command
        
        Args:
            command: [vx, vy, omega] command vector
        """
        self.env.command = np.array(command, dtype=np.float64)
    
    def get_command(self):
        """Get current locomotion command"""
        return np.array(self.env.command, dtype=np.float32)
    
    def seed(self, seed):
        """Set random seed"""
        self.env.seed(seed)
        np.random.seed(seed)
