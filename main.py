import numpy as np
import pandas as pd
from MazeQLearning import MazeQLearning



def main():
    maze_q_learning = MazeQLearning()
    # Create map, from demo file
    map_d = 5
    map_arr = 0 * np.random.rand(map_d, map_d)
    map_arr = map_arr.astype(int)
    map_arr += np.diag(list(range(map_d))) * 0
    map_arr = map_arr.astype(object)
    map_arr[:, 0] = '#'
    map_arr[0, :] = '#'
    map_arr[:, -1] = '#'
    map_arr[-1, :] = '#'
    map_arr[1][1] = 'S'
    map_arr[map_d - 2][map_d - 2] = 'G'
    print(map_arr)
    maze_q_learning.initialize(map_arr)

    # alpha = learning rate (1 = more short-term, but can overshoot, 0.1 = smaller, but can better adjust).
    # gamma = how much it value later rewards vs short-term rewards (1 = only long-term)
    # epsilon = how much it exploits (0.1 = 10% of the time)
    alpha_value, gamma_value, epsilon_value = 0.9, 0.9, 0.1

    maze_q_learning.alpha_value = alpha_value
    maze_q_learning.gamma_value = gamma_value
    maze_q_learning.epsilon_greedy_rate = epsilon_value

    for i in range(2):
        print(i)
        maze_q_learning.learn(state_key=(1, 1), limit=10000)
        maze_q_learning.q_df['q_value'] = maze_q_learning.q_df['q_value'].fillna(0)
        print(maze_q_learning.q_df)
    # maze_q_learning.visualize_learning_result((1,1))

    # If there's a NaN, replace it with a 0.
    # maze_q_learning.q_df['q_value'] = maze_q_learning.q_df['q_value'].fillna(0)
    q_df = maze_q_learning.q_df.sort_values(by=["q_value"], ascending=False)
    print(q_df)
    # getPolicy()


if __name__ == "__main__":
    main()
