import os
import sys
import numpy as np
import traci

# Add SUMO_HOME to path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare SUMO_HOME")

class RampMeteringEnv:
    def __init__(self):
        # State space: discretized density levels for mainline and ramp
        self.n_density_levels = 10
        self.n_states = self.n_density_levels * self.n_density_levels  # mainline × ramp
        
        # Action space: different traffic light phase durations
        self.actions = [5, 10, 15, 20]  # seconds for green phase
        self.n_actions = len(self.actions)
        
        # Initialize Q-table
        self.q_table = np.zeros((self.n_states, self.n_actions))
        
        # Traffic light ID
        self.tl_id = "node6"
        
        # Edge IDs for measurement
        self.mainline_edge = "2to3"
        self.ramp_edge = "intramp"

    def get_state(self):
        # Get vehicle counts/density
        mainline_density = traci.edge.getLastStepVehicleNumber(self.mainline_edge) / traci.edge.getLength(self.mainline_edge)
        ramp_density = traci.edge.getLastStepVehicleNumber(self.ramp_edge) / traci.edge.getLength(self.ramp_edge)
        
        # Discretize densities
        mainline_state = min(int(mainline_density * 10), self.n_density_levels - 1)
        ramp_state = min(int(ramp_density * 10), self.n_density_levels - 1)
        
        return mainline_state * self.n_density_levels + ramp_state

    def take_action(self, action):
        # Set traffic light phase duration
        phase_duration = self.actions[action]
        traci.trafficlight.setPhaseDuration(self.tl_id, phase_duration)
        
        # Run simulation for the action duration
        for _ in range(phase_duration):
            traci.simulationStep()
        
        # Calculate reward (negative of total waiting time)
        mainline_wait = traci.edge.getWaitingTime(self.mainline_edge)
        ramp_wait = traci.edge.getWaitingTime(self.ramp_edge)
        reward = -(mainline_wait + ramp_wait)
        
        # Get new state
        next_state = self.get_state()
        
        # Check if simulation is done
        done = traci.simulation.getMinExpectedNumber() <= 0
        
        return next_state, reward, done

def train():
    # Hyperparameters
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01
    episodes = 100

    env = RampMeteringEnv()
    
    for episode in range(episodes):
        # Start SUMO for this episode
        traci.start(["sumo", "-c", "mynet.sumocfg"])
        
        state = env.get_state()
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(env.q_table[state])
            
            # Take action and observe result
            next_state, reward, done = env.take_action(action)
            
            # Update Q-value
            best_next_action = np.argmax(env.q_table[next_state])
            td_target = reward + gamma * env.q_table[next_state, best_next_action] * (1 - done)
            env.q_table[state, action] += alpha * (td_target - env.q_table[state, action])
            
            state = next_state
        
        # Close SUMO
        traci.close()
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        print(f"Episode {episode + 1}/{episodes} completed")

if __name__ == "__main__":
    train()