"""
PPO-based Teacher Policy Training Script

Trains a teacher policy on privileged state information using PPO algorithm.
The teacher policy will be used later to supervise student policy training.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, Tuple, List
import argparse

from env_wrapper import AnymalTeacherEnv


class PolicyNetwork(nn.Module):
    """MLP policy network that outputs action mean and log_std"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU(),
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        # Initialize log_std to -0.5 (std = 0.6) and bound it to prevent explosion
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(state)
        mean = self.mean_layer(features)
        # Clamp log_std to prevent explosion: log_std in [-5, 2] => std in [0.0067, 7.39]
        log_std_clamped = torch.clamp(self.log_std, -5, 2)
        std = torch.exp(log_std_clamped)
        return mean, std
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        mean, std = self.forward(state)
        
        if deterministic:
            return mean, torch.zeros_like(mean)
        
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy of given actions"""
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy, mean


class ValueNetwork(nn.Module):
    """MLP value network for state value estimation"""
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [256, 256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU(),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)


class PPOBuffer:
    """Rollout buffer for PPO training"""
    
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, device: str):
        self.buffer_size = buffer_size
        self.device = device
        
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.full = False
        self.last_value = 0.0  # For bootstrapping
        
    def add(self, state, action, reward, value, log_prob, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        if self.ptr == 0:
            self.full = True
    
    def get(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> Dict[str, torch.Tensor]:
        """Compute advantages using GAE and return all data as tensors"""
        assert self.full, "Buffer must be full before sampling"
        
        # Compute GAE advantages with proper bootstrapping
        advantages = np.zeros(self.buffer_size, dtype=np.float32)
        last_gae = 0
        
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                # Bootstrap from last state value if episode didn't end
                next_value = self.last_value * (1 - self.dones[t])
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae
        
        returns = advantages + self.values
        
        # Normalize advantages (with safety check)
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - adv_mean) / adv_std
        
        return {
            'states': torch.tensor(self.states, device=self.device),
            'actions': torch.tensor(self.actions, device=self.device),
            'old_log_probs': torch.tensor(self.log_probs, device=self.device),
            'returns': torch.tensor(returns, device=self.device),
            'advantages': torch.tensor(advantages, device=self.device),
        }
    
    def reset(self):
        self.ptr = 0
        self.full = False


class PPOTrainer:
    """PPO trainer for teacher policy"""
    
    def __init__(
        self,
        env: AnymalTeacherEnv,
        state_dim: int,
        action_dim: int,
        device: str = 'cuda',
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        entropy_coef: float = 0.001,  # Lower entropy coefficient
        max_grad_norm: float = 1.0,
        n_epochs: int = 5,  # Fewer epochs to prevent overfitting
        batch_size: int = 256,
        buffer_size: int = 4096,
    ):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.value = ValueNetwork(state_dim).to(device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        
        # Buffer
        self.buffer = PPOBuffer(buffer_size, state_dim, action_dim, device)
        
        # Stats
        self.total_steps = 0
        self.total_episodes = 0
        
    def collect_rollouts(self, n_steps: int) -> Dict[str, float]:
        """Collect n_steps of experience"""
        self.policy.eval()
        self.value.eval()
        
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0
        
        state = self.env.reset()
        
        for _ in range(n_steps):
            with torch.no_grad():
                state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                action, log_prob = self.policy.get_action(state_tensor)
                value = self.value(state_tensor)
                
                action = action.cpu().numpy()[0]
                # Clip actions to reasonable range to prevent extreme values
                action = np.clip(action, -10.0, 10.0)
                log_prob = log_prob.cpu().item()
                value = value.cpu().item()
            
            next_state, reward, done, _ = self.env.step(action)
            
            self.buffer.add(state, action, reward, value, log_prob, done)
            
            state = next_state
            current_episode_reward += reward
            current_episode_length += 1
            self.total_steps += 1
            
            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                current_episode_reward = 0
                current_episode_length = 0
                self.total_episodes += 1
                state = self.env.reset()
        
        # Compute value for last state for bootstrapping
        with torch.no_grad():
            state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
            self.buffer.last_value = self.value(state_tensor).cpu().item()
        
        stats = {
            'mean_episode_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'mean_episode_length': np.mean(episode_lengths) if episode_lengths else 0,
            'n_episodes': len(episode_rewards),
        }
        
        return stats
    
    def train_step(self) -> Dict[str, float]:
        """Perform one PPO update"""
        self.policy.train()
        self.value.train()
        
        # Get data from buffer
        data = self.buffer.get(self.gamma, self.gae_lambda)
        
        # Train for n_epochs
        policy_losses = []
        value_losses = []
        entropies = []
        clip_fractions = []
        
        for _ in range(self.n_epochs):
            # Create minibatches
            indices = torch.randperm(len(data['states']), device=self.device)
            
            for start in range(0, len(data['states']), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = data['states'][batch_indices]
                batch_actions = data['actions'][batch_indices]
                batch_old_log_probs = data['old_log_probs'][batch_indices]
                batch_returns = data['returns'][batch_indices]
                batch_advantages = data['advantages'][batch_indices]
                
                # Policy loss
                log_probs, entropy, _ = self.policy.evaluate_actions(batch_states, batch_actions)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values = self.value(batch_states)
                value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy.mean()
                
                # Update
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                # Stats
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                    clip_fractions.append(clip_fraction)
        
        self.buffer.reset()
        
        # Get log_std stats for monitoring
        with torch.no_grad():
            log_std_mean = self.policy.log_std.mean().item()
            log_std_max = self.policy.log_std.max().item()
            log_std_min = self.policy.log_std.min().item()
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'clip_fraction': np.mean(clip_fractions),
            'log_std_mean': log_std_mean,
            'log_std_max': log_std_max,
            'log_std_min': log_std_min,
        }
    
    def save_checkpoint(self, save_path: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
        }, save_path)
        print(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, load_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self.total_episodes = checkpoint['total_episodes']
        print(f"Checkpoint loaded from {load_path}")


def train(args):
    """Main training loop"""
    
    # Create environment
    env = AnymalTeacherEnv(visualize=args.visualize)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")
    
    trainer = PPOTrainer(
        env=env,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        vf_coef=args.vf_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Training loop
    print(f"\nStarting training for {args.total_timesteps} timesteps...")
    
    start_time = time.time()
    iteration = 0
    
    while trainer.total_steps < args.total_timesteps:
        iteration += 1
        
        # Collect rollouts
        rollout_stats = trainer.collect_rollouts(args.buffer_size)
        
        # Train
        train_stats = trainer.train_step()
        
        # Logging
        elapsed_time = time.time() - start_time
        fps = trainer.total_steps / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"Iteration: {iteration} | Steps: {trainer.total_steps}/{args.total_timesteps} | Episodes: {trainer.total_episodes}")
        print(f"Time: {elapsed_time:.1f}s | FPS: {fps:.1f}")
        print(f"Rollout - Mean Reward: {rollout_stats['mean_episode_reward']:.2f} | Mean Length: {rollout_stats['mean_episode_length']:.1f} | Episodes: {rollout_stats['n_episodes']}")
        print(f"Training - Policy Loss: {train_stats['policy_loss']:.4f} | Value Loss: {train_stats['value_loss']:.4f}")
        print(f"Training - Entropy: {train_stats['entropy']:.4f} | Clip Fraction: {train_stats['clip_fraction']:.4f}")
        print(f"Training - log_std: mean={train_stats['log_std_mean']:.3f}, min={train_stats['log_std_min']:.3f}, max={train_stats['log_std_max']:.3f}")
        
        # Save checkpoint
        if iteration % args.save_freq == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_{trainer.total_steps}.pt")
            trainer.save_checkpoint(checkpoint_path)
            
            # Also save as latest
            latest_path = os.path.join(args.save_dir, "checkpoint_latest.pt")
            trainer.save_checkpoint(latest_path)
    
    # Final save
    final_path = os.path.join(args.save_dir, "checkpoint_final.pt")
    trainer.save_checkpoint(final_path)
    
    print(f"\nTraining complete! Total time: {time.time() - start_time:.1f}s")
    print(f"Final checkpoint saved to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train teacher policy with PPO")
    
    # Environment
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    
    # Training
    parser.add_argument('--total-timesteps', type=int, default=20_000_000, help='Total training timesteps')
    parser.add_argument('--buffer-size', type=int, default=4096, help='Rollout buffer size')
    parser.add_argument('--batch-size', type=int, default=256, help='Minibatch size')
    parser.add_argument('--n-epochs', type=int, default=5, help='Number of PPO epochs per update')
    
    # PPO hyperparameters
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip-range', type=float, default=0.2, help='PPO clip range')
    parser.add_argument('--vf-coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--entropy-coef', type=float, default=0.001, help='Entropy coefficient (lower to prevent explosion)')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='Max gradient norm')
    
    # Checkpointing
    parser.add_argument('--save-dir', type=str, default='checkpoints/teacher', help='Directory to save checkpoints')
    parser.add_argument('--save-freq', type=int, default=1000, help='Save checkpoint every N iterations')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    
    # Device
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {config_path}")
    
    # Train
    train(args)


if __name__ == '__main__':
    main()
