
import numpy as np
import pandas as pd
from MazeQLearning import MazeQLearning
# from prelab_files.Maze.maze import generate_maze
from prelab_files.demo_maze_greedy_q_learning import MazeGreedyQLearning
# from prelab_files.Maze import maze

def create_maze():
    maze = [
        ["#", "-1", "-1", "-1", "#"],
        ["#", "#", "#", "-1", "#"],
        ["-5", "5", "-10", "2", "#"],
        ["#", "5", "#", "#", "#"],
        ["#", "#", "#", "#", "#"]
    ]
    return maze


def main():
    # "S": Start point, "G": End point(goal), "#": wall, "@": Agent.
    start_point_label, end_point_label, wall_label, agent_label = ("S", "G", "#", "@")
    map_d = 10
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

    # alpha = learning rate (1 = more short-term, but can overshoot, 0.1 = smaller, but can better adjust).
    # gamma = how much it value later rewards vs short-term rewards (1 = only long-term)
    # epsilon = how much it exploits (0.1 = 10% of the time)
    alpha_value, gamma_value, epsilon_value = 0.9, 0.9, 0.1

    maze_q_learning = MazeGreedyQLearning()
    maze_q_learning.initialize(
        map_arr=map_arr,
        start_point_label=start_point_label,
        end_point_label=end_point_label,
        wall_label=wall_label,
        agent_label=agent_label
    )
    maze_q_learning.alpha_value = alpha_value
    maze_q_learning.gamma_value = gamma_value
    maze_q_learning.epsilon_greedy_rate = epsilon_value
    maze_q_learning.learn(state_key=(1, 1), limit=10000)



    # If there's a NaN, replace it with a 0.
    maze_q_learning.q_df['q_value'] = maze_q_learning.q_df['q_value'].fillna(0)
    q_df = maze_q_learning.q_df.sort_values(by=["q_value"], ascending=False)
    for index, row in q_df.iterrows():
        print(row['state_key'], row['action_key'], row['q_value'])
    # getPolicy()


if __name__ == "__main__":
    main()
