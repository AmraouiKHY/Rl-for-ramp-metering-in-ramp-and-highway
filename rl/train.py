import os
import time
import matplotlib.pyplot as plt
from .dqn import DQNAgent
from .env import SUMOEnvironment

class TrafficSimulator:
    def __init__(self, sumocfg_file):
        # Create models directory if it doesn't exist
        self.model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.env = SUMOEnvironment(sumocfg_file)
        self.agent = DQNAgent(state_size=self.env.state_size, 
                            action_size=self.env.action_size)
        
    def train(self, episodes=200):  # Reduced episodes
        scores = []
        early_stop = False
        
        for e in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            max_steps = 1000  # Reduced max steps
            
            while steps < max_steps:
                action = self.agent.act(state)
                next_state, reward, done = self.env.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                if len(self.agent.memory) >= self.agent.batch_size:
                    self.agent.replay()
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done or self.env.check_convergence():
                    early_stop = True
                    break
            
            scores.append(total_reward)
            print(f"Episode: {e}/{episodes}, Score: {total_reward}, Steps: {steps}")
            
            if early_stop:
                print("Early stopping triggered")
                break
                
        return scores

def main():
    sumocfg_file = "D:/3CS/RL/Rl project/sumo/mynet.sumocfg"
    simulator = TrafficSimulator(sumocfg_file)
    scores = simulator.train()
    
    # Plot and save results
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'training_progress.png'))

if __name__ == "__main__":
    main()