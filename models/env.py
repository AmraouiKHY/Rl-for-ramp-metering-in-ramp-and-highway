import os
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not found")

import traci

class RampMeteringEnv:
    def __init__(self, use_traffic_light=True):
        self.cfg_path = str(Path("D:/3CS/RL/rl for ramp metering/sumo/mynet.sumocfg"))
        self.sumoCmd = ["sumo", "-c", self.cfg_path, "--no-step-log", "--no-warnings"]
        self.use_traffic_light = use_traffic_light
        
        # Fixed lengths from SUMO network
        self.highway_length = 1000  # meters
        self.ramp_length = 100      # meters
        
        # Normalization constants
        self.MAX_DENSITY = 20.0
        self.MAX_QUEUE = 10.0
        self.MAX_SPEED = 13.89
        self.MAX_WAIT_TIME = 300.0
        
        # Q-learning parameters
        self.n_states = 4
        self.n_actions = 4
        self.q_table = {}
        
        # Network elements
        self.highway = "2to3"
        self.ramp = "intramp"
        self.tl_id = "node6"

    def start_simulation(self):
        try:
            traci.start(self.sumoCmd)
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def get_state(self):
        # Highway metrics (using fixed length)
        highway_density = traci.edge.getLastStepVehicleNumber(self.highway) / (self.highway_length/1000)
        highway_speed = traci.edge.getLastStepMeanSpeed(self.highway)
        
        # Ramp metrics
        ramp_queue = traci.edge.getLastStepHaltingNumber(self.ramp)
        ramp_speed = traci.edge.getLastStepMeanSpeed(self.ramp)
        
        # Normalize values
        norm_density = min(highway_density / self.MAX_DENSITY, 1.0)
        norm_queue = min(ramp_queue / self.MAX_QUEUE, 1.0)
        norm_highway_speed = min(highway_speed / self.MAX_SPEED, 1.0)
        norm_ramp_speed = min(ramp_speed / self.MAX_SPEED, 1.0)
        
        return tuple([norm_density, norm_queue, norm_highway_speed, norm_ramp_speed])

    def get_q_value(self, state, action=None):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        if action is None:
            return self.q_table[state]
        return self.q_table[state][action]

    def take_action(self, action):
        if not self.use_traffic_light:
            inflow_rates = [0.2, 0.5, 1.0, 2.0]  # vehicles/second
            inflow_rate = inflow_rates[action]
            try:
                # Control ramp inflow by adding vehicles at a controlled rate
                traci.edge.adaptTraveltime(self.ramp, 1 / inflow_rate)  # Adjust travel time to simulate metering

                # Simulate for a fixed number of steps (e.g., 10 seconds)
                for _ in range(10):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        return self.get_state(), 0, True
                    traci.simulationStep()

                return self.get_state(), self.calculate_reward(), False
            except Exception as e:
                print(f"Error during action execution: {e}")
                return self.get_state(), 0, True
           
        green_times = [5, 10, 15, 20]
        try:
            traci.trafficlight.setPhaseDuration(self.tl_id, green_times[action])
            
            # Simulate for green time duration
            for _ in range(green_times[action]):
                if traci.simulation.getMinExpectedNumber() <= 0:
                    return self.get_state(), 0, True
                traci.simulationStep()
            
            return self.get_state(), self.calculate_reward(), False
        except:
            return self.get_state(), 0, True

    def calculate_reward(self):
        waiting_time = (traci.edge.getWaitingTime(self.highway) + 
                     traci.edge.getWaitingTime(self.ramp))
        
        vehicles = (traci.edge.getLastStepVehicleNumber(self.highway) + 
                   traci.edge.getLastStepVehicleNumber(self.ramp))
        
        return (-waiting_time + vehicles * 10)