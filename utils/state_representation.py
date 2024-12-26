import traci
import numpy as np

class StateRepresentation:
    def __init__(self):
        # State normalization parameters
        self.speed_max = 50.0  # Maximum speed in m/s
        self.max_vehicles = 50  # Maximum number of vehicles
        self.max_density = 1.0  # Maximum density (vehicles per meter)
        self.max_queue = 20     # Maximum queue length
        self.max_wait = 300     # Maximum waiting time in seconds

    def get_state(self):
        """Get normalized state variables"""
        # Get highway metrics
        highway_vehicles = traci.edge.getLastStepVehicleNumber("2to3")
        highway_speed = traci.edge.getLastStepMeanSpeed("2to3")
        
        # Get edge length using lane length
        lane_id = "2to3_0"
        highway_length = traci.lane.getLength(lane_id)
        highway_density = highway_vehicles / highway_length if highway_length > 0 else 0
        
        # Get ramp metrics
        ramp_queue = traci.edge.getLastStepHaltingNumber("intramp")
        ramp_wait_time = traci.edge.getWaitingTime("intramp")
        
        # Normalize state variables
        state = np.array([
            highway_speed / self.speed_max,
            highway_vehicles / self.max_vehicles,
            highway_density / self.max_density,
            ramp_queue / self.max_queue,
            ramp_wait_time / self.max_wait
        ], dtype=np.float32)
        
        return state