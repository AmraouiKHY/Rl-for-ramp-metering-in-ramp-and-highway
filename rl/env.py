import os
import sys
import traci
import numpy as np
from utils.state_representation import StateRepresentation
from utils.action_control import ActionControl
from utils.reward_calculation import RewardCalculation
from collections import deque
import time

class SUMOEnvironment:
    def __init__(self, sumocfg_file, num_seconds=3600):
        self.state_size = 5  # [highway_speed, vehicles, density, queue, wait_time]
        self.action_size = 3
        
        self.sumocfg_file = sumocfg_file
        self.num_seconds = num_seconds
        
        # Performance tracking
        self.episode_start_time = None
        self.total_steps = 0
        self.rewards_history = deque(maxlen=100)
        self.steps_history = deque(maxlen=100)
        
        # Early stopping criteria
        self.convergence_threshold = 0.01
        self.convergence_window = 20
        
        # Initialize components
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            raise Exception("Please declare SUMO_HOME")
            
        self.state_rep = StateRepresentation()
        self.action_control = ActionControl()
        self.reward_calc = RewardCalculation()
        
    def reset(self):
        try:
            traci.close()
        except:
            pass
            
        sumo_cmd = ["sumo-gui", "-c", self.sumocfg_file]
        traci.start(sumo_cmd)
        self.episode_start_time = time.time()
        self.total_steps = 0
        return self.state_rep.get_state()
        
    def step(self, action):
        try:
            self.action_control.apply_action(action)
            traci.simulationStep()
            
            # Add early termination
            if self.total_steps > 2000:  # Max steps per episode
                done = True
            else:
                done = traci.simulation.getMinExpectedNumber() <= 0
                
            next_state = self.state_rep.get_state()
            reward = self.reward_calc.calculate_reward()
            
            return next_state, reward, done
            
        except traci.exceptions.TraCIException as e:
            print(f"TraCI Exception: {e}")
            return None, 0, True
            
    def close(self):
        try:
            traci.close()
        except:
            pass
            
    def check_convergence(self):
        if len(self.rewards_history) < self.convergence_window:
            return False
            
        recent_rewards = list(self.rewards_history)[-self.convergence_window:]
        variance = np.var(recent_rewards)
        return variance < self.convergence_threshold