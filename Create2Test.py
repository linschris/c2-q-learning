import random
from prelab_files.demo_maze_greedy_q_learning import MazeGreedyQLearning
import numpy as np
from mazePolicy import Policy
from GreedyCreate2 import GreedyCreate2


def createEmptyMaze(map_d = 10):
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
    # print(map_arr)
    return map_arr



def agentLearn(map_arr, alpha_value, gamma_value, epsilon_value, limit, num_episodes=1):
    '''
        Map_array: numpy of the map array
        alpha_value: learning rate of the q-learner (affects how much the q-values change by every iteration)
        gamma_value: discount rate (0 < g < 1) 
        - (1 = all long-term rewards, 0 = all short-term rewards)
        epsilon_value: greedy rate (0 < e < 1)
        - (1 = always executes best found move to "exploit" (i.e. be greedy), 0 = always executes randomly to "explore")
        limit: max_num of q-value iterations
        - Note that the eposides if the agent hits the goal OR the limit is reached.
    '''
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
    for i in range(num_episodes):
        # random_x = random.randint(1, 8)
        # random_y = random.randint(1, 8)
        maze_q_learning.learn(state_key=(1,1), limit=limit)
        # q_df = maze_q_learning.q_df.sort_values(by=["q_value"], ascending=False)
        # print(maze_q_learning.q_df)


    maze_q_learning.q_df['q_value'] = maze_q_learning.q_df['q_value'].fillna(0)
    maze_q_learning.q_df = maze_q_learning.q_df.sort_values(by=["state_key"], ascending=False)
    return maze_q_learning

def main():
    # map_arr = createEmptyMaze(5)
    # map_arr = np.asanyarray(createComplexMaze())
    print(createEmptyMaze(5))
    print(createBridgeMaze())
    print(createComplexMaze())
    map_arr = createEmptyMaze(5)

    # 0.9, 0.9, 0.1, 10, 20 for simple maze

    q_learner = agentLearn(map_arr=map_arr, alpha_value=0.9, gamma_value=0.9, epsilon_value=0.009, limit=100000, num_episodes=1)
    c2_policy = Policy(q_learner)
    c2_policy.printQTable()
    greedy_c2 = GreedyCreate2(c2_policy, '/dev/tty.usbserial-DN02693O', state=(1,1), debug=True)
    greedy_c2.execute_policy()
    # print(q_learner.inference(10))

def createBridgeMaze():
    start_point_label, end_point_label, wall_label, agent_label = ("S", "G", "#", "@")
    map_arr = [[wall_label] * 7,
              [wall_label, wall_label, 1, 1, 1, wall_label, wall_label],
              [wall_label, start_point_label, 10, 10, 10, end_point_label, wall_label],
              [wall_label, wall_label, 1, 1, 1, wall_label, wall_label],
              [wall_label] * 7
              ]
    return np.asanyarray(map_arr)

def createComplexMaze():
    start_point_label, end_point_label, wall_label, agent_label = ("S", "G", "#", "@")
    map_arr = [[wall_label] * 12,
              [wall_label, wall_label, 10, wall_label, 0, -1, -2, 5, 5, 5, 5, wall_label],
              [wall_label, start_point_label, 1, 1, 1, wall_label, wall_label, wall_label, wall_label, wall_label, 0, wall_label],
              [wall_label, wall_label, 1, wall_label, wall_label, wall_label, wall_label, wall_label, wall_label, wall_label, end_point_label, wall_label],
              [wall_label, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, wall_label],
              [wall_label] * 12
              ]
    return np.asanyarray(map_arr)


if __name__ == "__main__":
    main()

