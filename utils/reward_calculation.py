# rl/reward_calculation.py
import traci

class RewardCalculation:
    def __init__(self):
        self.prev_waiting_time = 0
        self.max_reward = 1000
        
    def calculate_reward(self):
        # Get metrics
        highway_flow = traci.edge.getLastStepVehicleNumber("2to3")
        highway_speed = traci.edge.getLastStepMeanSpeed("2to3")
        current_wait = traci.edge.getWaitingTime("intramp")
        
        # Calculate reward components
        flow_reward = min(highway_flow * highway_speed, self.max_reward)
        wait_penalty = -100 * (current_wait - self.prev_waiting_time)
        
        # Update previous waiting time
        self.prev_waiting_time = current_wait
        
        return flow_reward + wait_penalty