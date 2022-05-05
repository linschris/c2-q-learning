from prelab_files.demo_maze_greedy_q_learning import MazeGreedyQLearning
import numpy as np
from mazePolicy import Policy
from GreedyCreate2 import GreedyCreate2
from createMazes import createBridgeMaze, createSimpleMaze, createLongPathMaze


def agentLearn(map_arr, alpha_value, gamma_value, epsilon_value, limit, start_state=(1,1), num_episodes=1):
    '''
        Initializes Q-Learning agent and simulates a Q-Learning agent learning the maze, storing its associated
        Q(state, action) values in a Q table (nd array).

        Returns the instance to be used in the Policy class.

        Map_array: numpy of the map array
        alpha_value: learning rate of the q-learner (affects how much the q-values change by every iteration)
        gamma_value: discount rate (0 < g < 1) 
        - (1 = all long-term rewards, 0 = all short-term rewards)
        epsilon_value: greedy rate (0 < e < 1)
        - (1 = always executes best found move to "exploit" (i.e. be greedy), 0 = always executes randomly to "explore")
        limit: max_num of q-value iterations
        - Note that the eposides if the agent hits the goal OR the limit is reached.
    '''
    # Initialize map array
    start_point_label, end_point_label, wall_label, agent_label = ("S", "G", "#", "@")
    maze_q_learning = MazeGreedyQLearning()
    maze_q_learning.epsilon_greedy_rate = epsilon_value
    maze_q_learning.alpha_value = alpha_value
    maze_q_learning.gamma_value = gamma_value
    maze_q_learning.initialize(
        map_arr=map_arr,
        start_point_label=start_point_label,
        end_point_label=end_point_label,
        wall_label=wall_label,
        agent_label=agent_label
    )

    # Learn from the start state with a maximum "limit" of value iterations until eposide ends
    for i in range(num_episodes):
        maze_q_learning.learn(state_key=start_state, limit=limit)

    maze_q_learning.q_df['q_value'] = maze_q_learning.q_df['q_value'].fillna(0) # Replace NaNs with 0s
    maze_q_learning.q_df = maze_q_learning.q_df.sort_values(by=["state_key"], ascending=False) # Sort by state_key
    return maze_q_learning

def main():
    map_arr = createSimpleMaze() # createSimpleMaze(), createBridgeMaze(), createLongPathMaze() were used

    starting_state = (1,3)

    ''' Variables of learning, discount, greedy rate, as well the max iterations per episode and # of eposides'''
    # a    g    e  limit eps 
    # 0.9, 0.9, 0.1, 10, 20 for simple maze
    # 0.9, 0.9, 0.9, 200, 20 for bridge maze
    # 0.9, 0.99, 0.9, 200, 20 for long path maze

    q_learner = agentLearn(map_arr=map_arr, alpha_value=0.9, gamma_value=0.99, epsilon_value=0.9, limit=200, start_state = starting_state, num_episodes=20)
    c2_policy = Policy(q_learner)
    greedy_c2 = GreedyCreate2(c2_policy, '/dev/tty.usbserial-DN0266RJ', state=starting_state, debug=True)
    
    # time.sleep(5) # Not needed, but it gave me time to get phone ready to record
    greedy_c2.execute_policy()



if __name__ == "__main__":
    main()

