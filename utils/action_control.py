# rl/action_control.py
import traci

class ActionControl:
    def __init__(self):
        self.tl_id = "node6"
        self.phase_durations = [30, 45, 60]  # Possible green light durations
        
    def apply_action(self, action):
        """
        Apply selected action to traffic light
        action: index of phase_durations
        """
        duration = self.phase_durations[action]
        traci.trafficlight.setPhaseDuration(self.tl_id, duration)