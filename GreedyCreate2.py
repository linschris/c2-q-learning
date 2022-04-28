from pycreate2 import Create2
import numpy as np
import time

'''
    A GreedyCreate2 class.
    Inherits the methods from Create2, but has the power to take its environment
    and develop a policy, and further actions from it.
'''
class GreedyCreate2(Create2):
    def __init__(self, policy, port, baud=115200, curr_dir='N', state=(1, 1), debug=True):
        # super().__init__(port, baud=baud)
        self.policy = policy
        self.state = state
        self.curr_dir = curr_dir
        self.debug = debug # Debug mode mainly refers to if we're disconnected from IRL Create2
        self.path_taken = []

        # (x, y) diff = direction
        self.possible_directions = {(0, -1): "N", (0, 1): "S", (1, 0): "E", (-1, 0): "W"}
    
    def __del__(self):
        if not self.debug:
            super().__del__()
        else:
            print(self.policy.best_actions)
            self.policy.printQTable()
            pass

    def get_relative_direction(self):
        if self.curr_dir == "W":
            self.possible_directions = {(-1, 0): "N", (1, 0): "S", (0, -1): "E", (0, 1): "W"}
        elif self.curr_dir == "E":
            self.possible_directions = {(1, 0): "N", (-1, 0): "S", (0, 1): "E", (0, -1): "W"}
        elif self.curr_dir == "S":
            self.possible_directions = {(0, 1): "N", (0, -1): "S", (-1, 0): "E", (1, 0): "W"}
        else: 
            self.possible_directions = {(0, -1): "N", (0, 1): "S", (1, 0): "E", (-1, 0): "W"}

    def determine_direction(self, state1, state2):
        # Determines direction to go based on initial and final state.
        self.get_relative_direction()
        state_diff = tuple(np.subtract(state2, state1)) # subtract x,y tuples
        return self.possible_directions[state_diff] if state_diff in self.possible_directions else -1
    
    def turn(self, direction):
        # Turns a given amount based on the direction given
        self.curr_dir = direction
        if direction == "E":
            self.turn_angle(90)
        elif direction == "W":
            self.turn_angle(270)
        elif direction == "S":
            self.turn_angle(180)

    def turn_angle(self, angle):
        angle_t = curr = super().get_sensors()['angle']
        while abs(curr - angle_t) < angle:
            super().drive_direct(0, 25) # Turn left slowly
            curr = super().get_sensors()['angle']
        super().drive_stop()


    def move(self, direction):
        # Move by one tile in a specific direction
        self.turn(self.determine_direction(direction))
        start_time = curr_time = time.time()
        while curr_time - start_time < 5:
            super().drive_direct(50, 50)
            curr_time = time.time()
        super().drive_stop()

    def execute_policy(self):
        '''
            Psuedocode:
                While not at goal:
                    Grab the next best move
                    Determine the direction
                    Move in that direction
                    Update state
                    Repeat
        '''
        dir_path = []
        while not self.policy.atGoal(self.state):
            curr_action = self.policy.getAction(self.state, self.state in self.path_taken)
            curr_direction = self.determine_direction(self.state, curr_action)
            self.path_taken.append(self.state)
            dir_path.append(self.policy.q_learner.observe_reward_value(self.state, curr_action))
            if not self.debug:
                self.move(curr_direction) 
            else:
                print(f'Current state: {self.state}, {self.policy.atGoal(self.state)}')
            self.state = curr_action
        self.path_taken.append(self.state)
        print(sum(dir_path))
        

