from pycreate2 import Create2
import numpy as np
import time

'''
    A GreedyCreate2 class.
    Inherits the methods from Create2, but has the power to take a policy
    and act upon it.
'''
class GreedyCreate2(Create2):
    def __init__(self, policy, port, baud=115200, curr_dir='N', state=(1, 1), debug=True):
        self.debug = debug # Debug mode mainly refers to if we're disconnected from IRL Create2 (but it also prints statements)
        if not self.debug:
            # Start bot
            super().__init__(port, baud=baud)
            self.start()
            self.full()
        
        self.policy = policy
        self.state = state
        self.curr_dir = curr_dir 
        self.path_taken = []

        # (x, y) diff = direction
        self.possible_directions = {(0, -1): "N", (0, 1): "S", (1, 0): "E", (-1, 0): "W"}
    
    def __del__(self):
        if not self.debug:
            self.drive_stop()
            self.stop()
            super().__del__()
        else:
            print(self.policy.best_actions)
            print(self.policy.possible_actions)
            self.policy.printQTable()
        print(self.path_taken)

    def get_relative_direction(self):
        # Based on our current facing direction, affect what direction we need to
        # turn to get the next state. 
        # 
        # Ex: If we're facing west, we need to go west to go one y down (South)

        if self.curr_dir == "W":
            self.possible_directions = {(-1, 0): "N", (1, 0): "S", (0, 1): "W", (0, -1): "E"}
        elif self.curr_dir == "E":
            self.possible_directions = {(1, 0): "N", (-1, 0): "S", (0, -1): "W", (0, 1): "E"}
        elif self.curr_dir == "S":
            self.possible_directions = {(0, 1): "N", (0, -1): "S", (-1, 0): "E", (1, 0): "W"}
        else: 
            self.possible_directions = {(0, -1): "N", (0, 1): "S", (1, 0): "E", (-1, 0): "W"}

    def determine_direction(self, state1, state2):
        # Determines direction to go based on initial and final state.
        state_diff = tuple(np.subtract(state2, state1)) # subtract x,y tuples
        self.get_relative_direction() 
        if self.debug:
            print(f"Currently, I need to go {state_diff} while facing {self.curr_dir}.")
            print(f"Currently, I need to go {self.possible_directions[state_diff]}")
            print(f"So, in reality, I'm now going to face: {self.determine_real_direction(self.possible_directions[state_diff])} after turning.")

        # Given that we'll turn the given direction in self.possible_directions[state_diff]
        # Change our current direction to our new current facing direction 
        self.curr_dir = self.determine_real_direction(self.possible_directions[state_diff])

        # Return the direction we need to turn
        return self.possible_directions[state_diff] if state_diff in self.possible_directions else -1

    def determine_real_direction(self, new_dir):
        # This is useful to keep track of our real direction
        # As going "south" twice, we should be facing north (in memory) for example. 
        if self.curr_dir == "S":
            real_dir = {"N": "S", "S": "N", "E": "W", "W": "E"}
        elif self.curr_dir == "W":
            real_dir = {"N": "W", "S": "E", "E": "N", "W": "S"}
        elif self.curr_dir == "E":
            real_dir = {"N": "E", "S": "W", "E": "S", "W": "N"}
        else:
            real_dir = {"N": "N", "S": "S", "E": "E", "W": "W"}
        return real_dir[new_dir]


    def turn(self, direction):
        # Turns a given amount to the right based on the direction given
        if direction == "E":
            self.turn_angle(90)
        elif direction == "W":
            self.turn_angle(-90)
        elif direction == "S":
            self.turn_angle(180)

    def turn_angle(self, given_angle, speed=100):
        # Turns a given angle in increments of 90
        # in an attempt to minimize error.  
        speed_r, speed_l = (speed, -speed) if given_angle < 0 else (-speed, speed)

        for i in range(0, abs(given_angle) // 90):
            angle_t = curr = self.get_sensors().angle
            while abs(curr - angle_t) < 97: # Turned more than 90 to compensate for wheel slip and encoder errors
                self.drive_direct(speed_r, speed_l) 
                curr += self.get_sensors().angle
            self.drive_stop()

    # def turn_angle(self, given_angle):
    #     curr_time = time.time()
    #     while time.time() - curr_time < (1.35 * given_angle // 90):
    #         self.drive_direct(-100, 100)
    #     self.drive_stop()

    def move(self, direction):
        # Move by one "tile"/unit in a specific direction
        self.turn(direction)
        start_time = curr_time = time.time()
        while curr_time - start_time < 4.5:
            self.drive_direct(75, 75)
            curr_time = time.time()
        self.drive_stop()
    

    def execute_policy(self):
        '''
            Psuedocode:
                While not at goal:
                    Grab the next best move
                    Determine the relative direction
                    Move in that direction
                    Append old state to path_taken
                    Update state to new state (or the action we took)
                    Repeat
        '''
        while not self.policy.atGoal(self.state):
            # if self.state in self.path_taken, we've hit a cycle.
            curr_action = self.policy.getAction(self.state, self.state in self.path_taken) 
            if curr_action == (-1, -1):
                # Invalid state
                break
            curr_direction = self.determine_direction(self.state, curr_action)
            self.path_taken.append(self.state)
            if not self.debug:
                self.move(curr_direction) 
            else:
                # Create2 is not connected.
                print(f'Current state: {self.state}, Moving: {curr_direction}, {self.policy.atGoal(self.state)}')
            self.state = curr_action
        self.path_taken.append(self.state)
        

