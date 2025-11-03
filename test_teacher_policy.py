"""
Test script for evaluating trained teacher policy
"""

import os
import sys
import numpy as np
import torch
import argparse
from typing import List

from env_wrapper import AnymalTeacherEnv
from teacher_trainer import PolicyNetwork


def test_policy(checkpoint_path: str, 
                n_episodes: int = 10,
                visualize: bool = True,
                deterministic: bool = True,
                terrain_type: int = None,
                terrain_params: List[float] = None):
    """
    Test a trained teacher policy
    
    Args:
        checkpoint_path: Path to model checkpoint
        n_episodes: Number of episodes to test
        visualize: Whether to enable visualization
        deterministic: Use deterministic actions (mean) vs stochastic
        terrain_type: Specific terrain type (0-8) or None for random
        terrain_params: Specific terrain parameters or None for random
    """
    
    # Load checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Using device: {device}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create environment
    env = AnymalTeacherEnv(visualize=visualize)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create policy and load weights
    policy = PolicyNetwork(state_dim, action_dim).to(device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    print(f"\nCheckpoint info:")
    print(f"  Total steps: {checkpoint['total_steps']}")
    print(f"  Total episodes: {checkpoint['total_episodes']}")
    
    # Test policy
    print(f"\n{'='*80}")
    print(f"Testing policy for {n_episodes} episodes...")
    print(f"Deterministic: {deterministic}")
    print(f"{'='*80}\n")
    
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    
    for ep in range(n_episodes):
        # Reset with specific terrain if provided
        if terrain_type is not None:
            state = env.reset(task_idx=terrain_type, task_params=terrain_params)
        else:
            state = env.reset()
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\nEpisode {ep + 1}/{n_episodes}")
        print(f"  Terrain type: {env.current_task_idx}")
        print(f"  Terrain params: {env.task_params}")
        
        while not done:
            # Get action from policy
            with torch.no_grad():
                state_tensor = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
                
                if deterministic:
                    # Use mean action
                    mean, _ = policy.forward(state_tensor)
                    action = mean.cpu().numpy()[0]
                else:
                    # Sample from distribution
                    action, _ = policy.get_action(state_tensor, deterministic=False)
                    action = action.cpu().numpy()[0]
            
            # Step environment
            state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Print progress every 100 steps
            if episode_length % 100 == 0:
                print(f"    Step {episode_length}: reward={episode_reward:.2f}")
        
        # Episode finished
        success = info.get('episode_success', False)
        is_timeout = info.get('is_timeout', False)
        is_terminal = info.get('is_terminal', False)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_successes.append(success)
        
        print(f"  Finished:")
        print(f"    Total reward: {episode_reward:.2f}")
        print(f"    Episode length: {episode_length}")
        print(f"    Success: {success}")
        print(f"    Timeout: {is_timeout}")
        print(f"    Terminal: {is_terminal}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"Test Summary:")
    print(f"{'='*80}")
    print(f"Episodes: {n_episodes}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Success rate: {np.mean(episode_successes) * 100:.1f}%")
    print(f"{'='*80}\n")
    
    env.close()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_successes': episode_successes,
    }


def main():
    parser = argparse.ArgumentParser(description="Test trained teacher policy")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--n-episodes', type=int, default=10,
                        help='Number of test episodes')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Disable visualization')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic actions instead of deterministic')
    parser.add_argument('--terrain-type', type=int, default=None,
                        help='Specific terrain type (0-8), None for random')
    parser.add_argument('--terrain-params', type=float, nargs=3, default=None,
                        help='Terrain parameters [p1, p2, p3]')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    # Test policy
    test_policy(
        checkpoint_path=args.checkpoint,
        n_episodes=args.n_episodes,
        visualize=not args.no_visualize,
        deterministic=not args.stochastic,
        terrain_type=args.terrain_type,
        terrain_params=args.terrain_params if args.terrain_params else None,
    )


if __name__ == '__main__':
    main()
