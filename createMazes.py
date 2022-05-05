import numpy as np

# All the mazes are stored here to declutter the main file.

def createSimpleMaze(map_d = 5):
    ''' 
        Creates maze with goal of having agent reach the goal state of (3,3) in an efficient manner.
        Note: the map dimensions will be 2 less than the given map dimension, as surrounding walls are a part of the maze.
        For map_d = 5 --> 3x3 maze
    '''
    
    start_point_label, end_point_label, wall_label, agent_label = ("S", "G", "#", "@")
    map_arr = 10 * np.random.rand(map_d, map_d)
    map_arr = map_arr.astype(int)
    map_arr += np.diag(list(range(map_d))) * 10
    map_arr = map_arr.astype(object)
    map_arr[:, 0] = wall_label
    map_arr[0, :] = wall_label
    map_arr[:, -1] = wall_label
    map_arr[-1, :] = wall_label
    map_arr[1][1] = start_point_label
    map_arr[map_d - 2][map_d - 2] = end_point_label

    # Values were reduced to all 1s, to make reward of reaching goal high.
    map_arr[1][2] = 1
    map_arr[2][2] = 1
    map_arr[3][2] = 1
    map_arr[2][1] = 1
    map_arr[3][1] = 1
    map_arr[2][3] = 1
    map_arr[1][3] = 1
    # print(map_arr)
    return map_arr

def createBridgeMaze():
    '''Creates maze where the goal is the agent should go straight along the bridge to the goal (and not "fall off")'''

    start_point_label, end_point_label, wall_label, agent_label = ("S", "G", "#", "@")
    map_arr = [[wall_label] * 7,
              [wall_label, wall_label, 1,1,1, wall_label, wall_label],
              [wall_label, start_point_label, 20,40,70, end_point_label, wall_label],
              [wall_label, wall_label, 1,1,1, wall_label, wall_label],
              [wall_label] * 7
              ]
    return np.asanyarray(map_arr)

def createLongPathMaze():
    '''Creates maze where the goal is the agent should go down the reward-filled, longer path to the goal.'''

    start_point_label, end_point_label, wall_label, agent_label = ("S", "G", "#", "@")
    map_arr = [[wall_label] * 11,
              [wall_label, wall_label, wall_label, wall_label, wall_label, wall_label, wall_label, wall_label, wall_label, end_point_label, wall_label],
              [wall_label, wall_label, 1, 1, 1, 1, 1, 1, 1, 1, wall_label],
              [wall_label, start_point_label, 1, wall_label, wall_label, wall_label, 1, wall_label, wall_label, 10, wall_label],
              [wall_label, wall_label, 1, 1, 1, 1, 1, 10, 10, 10, wall_label],
              [wall_label] * 11
              ]
    return np.asanyarray(map_arr)