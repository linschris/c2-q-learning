from pycreate2 import Create2
import numpy as np

'''
    A GreedyCreate2 class.
    Inherits the methods from Create2, but has the power to take its environment
    and develop a policy, and further actions from it.
'''
class GreedyCreate2(Create2):

    possible_directions = {(0, 1): "N", (0, -1): "S", (1, 0): "E", (-1, 0): "W"} # (x, y) diff = direction

    def __init__(self, policy, port, baud=115200):
        super().__init__(port, baud=baud)
        self.policy = policy
        self.state = None
        self.curr_dir = None

    #TODO: keep track of its current state and functions below

    def get_action(self, state_key):
        # Determines action based on a state tuple (state_key)
        pass

    def determine_direction(self, state1, state2):
        # Determines direction to go based on initial and final state.
        state_diff = tuple(np.subtract(state1, state2)) # subtract x,y tuples
        return GreedyCreate2.possible_directions[state_diff] if state_diff in GreedyCreate2.possible_directions else -1
    
    def turn(self, direction):
        # Turns a given amount based on the direction given
        pass
    def move(self, direction):
        # Move by one tile in a specific direction
        pass

    def execute_policy():
        '''
            Psuedocode:
                While not at goal:
                    Grab the next best move
                    Determine the direction
                    Move in that direction
                    Update state
                    Repeat
        '''
        pass

