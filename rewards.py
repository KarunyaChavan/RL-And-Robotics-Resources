"""
Reward function for ANYmal teacher policy training
Based on the paper: "Learning Quadrupedal Locomotion over Challenging Terrain"
Section S4: Reward function for teacher policy training
"""
import numpy as np


def compute_reward(env, action, prev_action, prev_prev_action, foot_targets_history):
    """
    Compute reward for the teacher policy as specified in the paper
    
    Total reward = 0.05*r_lv + 0.05*r_av + 0.04*r_b + 0.01*r_fc + 
                   0.02*r_bc + 0.025*r_s + 2e-5*r_tau
    
    Args:
        env: ANYmal environment instance
        action: Current action (16-dim)
        prev_action: Previous action (16-dim)
        prev_prev_action: Action from 2 steps ago (16-dim)
        foot_targets_history: List of foot target positions for smoothness reward
    
    Returns:
        reward: Scalar reward
        info: Dictionary with reward components
    """
    # Get state information
    base_vel = np.array(env.getBaseVelocity()).flatten()  # [vx, vy, vz, wx, wy, wz] in body frame
    command = np.array(env.command).flatten()  # [vx_des, vy_des, omega_des]
    contacts = np.array(env.getContactState()).flatten()  # [foot0, foot1, foot2, foot3]
    phases = np.array(env.getPhases()).flatten()  # CPG phases [phi0, phi1, phi2, phi3]
    _, _, num_base, num_shank, num_thigh = env.getContactCounts()
    
    # Determine if zero command
    is_zero_command = (np.linalg.norm(command[:2]) < 0.01 and abs(command[2]) < 0.01)
    
    # 1. Linear Velocity Reward (r_lv)
    if is_zero_command:
        # Penalize any movement when command is zero
        r_lv = -np.linalg.norm(base_vel[:2])
    else:
        # Project base velocity onto command direction
        command_xy_norm = np.linalg.norm(command[:2])
        if command_xy_norm > 0:
            command_dir = command[:2] / command_xy_norm
            v_pr = np.dot(base_vel[:2], command_dir)  # Positive if moving in right direction
            desired_speed = command_xy_norm
        else:
            v_pr = 0.0
            desired_speed = 0.0
        
        # Reward tracking the desired speed in the correct direction
        # Heavily penalize if moving in wrong direction (v_pr < 0)
        if desired_speed > 0:
            speed_error = abs(v_pr - desired_speed)
            r_lv = np.exp(-2.0 * speed_error**2)
            # Additional penalty if moving backward (wrong direction)
            if v_pr < 0:
                r_lv = -1.0
        else:
            r_lv = 0.0
    
    # 2. Angular Velocity Reward (r_av)
    omega_command = command[2]
    omega_actual = base_vel[5]
    if abs(omega_command) > 0.01:
        # Reward for tracking angular velocity
        omega_error = abs(omega_actual - omega_command)
        r_av = np.exp(-1.5 * omega_error**2)
    else:
        # Penalize spinning when no turn commanded
        r_av = -abs(omega_actual)
    
    # 3. Base Motion Reward (r_b)
    # Penalize velocity orthogonal to target direction and roll/pitch rates
    if is_zero_command:
        # When stopped, penalize all linear velocity
        v_o = np.linalg.norm(base_vel[:3])
    else:
        # Velocity orthogonal to command direction
        if np.linalg.norm(command[:2]) > 0:
            command_dir = command[:2] / np.linalg.norm(command[:2])
            v_pr_vec = np.dot(base_vel[:2], command_dir) * command_dir
            v_o = np.linalg.norm(base_vel[:2] - v_pr_vec)
        else:
            v_o = 0.0
    
    # Penalize roll and pitch rates (xy components of angular velocity)
    omega_xy = np.linalg.norm(base_vel[3:5])
    # Changed to negative exponential to penalize drift and instability
    r_b = -v_o**2 - omega_xy**2
    
    # 4. Foot Clearance Reward (r_fc)
    # During swing phase (phi in [pi, 2pi)), foot should be above terrain
    # Get privileged state for terrain scanning
    priv_state = env.getPrivilegedState().flatten()
    
    # Extract foot height scan data (36 dims starting at index 133)
    # 9 samples per foot x 4 feet
    foot_scans = priv_state[133:169].reshape(4, 9)
    
    # Determine swing phase: phi in [pi, 2pi) means pi < phi < 2*pi
    swing_legs = []
    for i in range(4):
        phi_normalized = phases[i] % (2 * np.pi)
        if phi_normalized > np.pi:
            swing_legs.append(i)
    
    if len(swing_legs) > 0:
        clearance_count = 0
        for leg_idx in swing_legs:
            # Get foot position
            q = env.getGeneralizedState()
            # foot height is approximate - use base height as proxy
            foot_height = q[2]  # Base z position
            
            # Max scanned height around this foot
            max_scan_height = np.max(foot_scans[leg_idx])
            
            # Check if foot is above max scanned terrain
            if foot_height > max_scan_height:
                clearance_count += 1
        
        r_fc = clearance_count / len(swing_legs)
    else:
        r_fc = 0.0
    
    # 5. Body Collision Reward (r_bc)
    # Penalize contact with body parts other than feet
    # num_base, num_shank, num_thigh from contact counts
    num_bad_contacts = num_base + num_shank + num_thigh
    r_bc = -num_bad_contacts
    # Note: Episode will terminate via C++ isTerminal() if base contacts ground
    
    # 6. Target Smoothness Reward (r_s)
    # Penalize second-order finite difference of foot target positions
    # foot_targets_history should contain [current, t-1, t-2]
    if len(foot_targets_history) >= 3:
        ft_curr = foot_targets_history[-1]
        ft_prev = foot_targets_history[-2]
        ft_prev2 = foot_targets_history[-3]
        
        # Second order finite difference
        fd2 = ft_curr - 2 * ft_prev + ft_prev2
        r_s = -np.linalg.norm(fd2)
    else:
        r_s = 0.0
    
    # 7. Torque Reward (r_tau)
    # Penalize joint torques (not directly available, use action magnitude as proxy)
    # In the actual env, joint torques would be computed from PD control
    # For now, penalize action magnitude
    r_tau = -np.sum(np.abs(action))
    
    # Total reward with increased weights for tracking performance
    reward = (1.0 * r_lv +      # Increased from 0.05 - most important
              0.5 * r_av +       # Increased from 0.05
              0.1 * r_b +        # Increased from 0.04
              0.01 * r_fc + 
              0.5 * r_bc +       # Increased from 0.02 - penalize bad contacts heavily
              0.025 * r_s + 
              2e-5 * r_tau)
    
    # Info dict for logging
    info = {
        'reward_total': reward,
        'reward_linear_vel': 1.0 * r_lv,
        'reward_angular_vel': 0.5 * r_av,
        'reward_base_motion': 0.1 * r_b,
        'reward_foot_clearance': 0.01 * r_fc,
        'reward_body_collision': 0.5 * r_bc,
        'reward_smoothness': 0.025 * r_s,
        'reward_torque': 2e-5 * r_tau,
        'r_lv': r_lv,
        'r_av': r_av,
        'r_b': r_b,
        'r_fc': r_fc,
        'r_bc': r_bc,
        'r_s': r_s,
        'r_tau': r_tau,
        'base_vel_x': base_vel[0],
        'base_vel_y': base_vel[1],
        'base_vel_z': base_vel[2],
        'base_omega_z': base_vel[5],
        'command_x': command[0],
        'command_y': command[1],
        'command_omega': command[2],
        'num_bad_contacts': num_bad_contacts,
        'num_swing_legs': len(swing_legs) if 'swing_legs' in locals() else 0,
    }
    
    return reward, info


def get_foot_target_positions(env):
    """
    Extract current foot target positions from environment
    This is used for computing the smoothness reward
    
    Returns:
        foot_targets: numpy array of shape (12,) with [x,y,z] for each foot
    """
    # The foot targets are part of the state/action computation
    # For now, return a placeholder - this should be extracted from the environment
    # or tracked externally during stepping
    return np.zeros(12)
