from prelab_files.demo_maze_greedy_q_learning import MazeGreedyQLearning

class MazeQLearning(MazeGreedyQLearning):
    '''
        Contains my modifications to any of the methods inherited
        from MazeGreedyQLearning or QLearning, the base class.
    '''
    # def initialize(self, map_arr, start_point_label="S", end_point_label="G", wall_label="#", agent_label="@"):
    #     self.__agent_label = agent_label
    #     self.__map_arr = map_arr
    #     self.__start_point_label = start_point_label
    #     # start_arr_tuple = (np.where(self.__map_arr == self.__start_point_label))
    #     # x_arr, y_arr = start_arr_tuple
    #     self.__start_point_tuple = (1,1)
    #     # end_arr_tuple = np.where(self.__map_arr == self.__end_point_label)
    #     # x_arr, y_arr = end_arr_tuple
    #     self.__end_point_tuple = (5,5)
    #     self.__wall_label = wall_label

    #     # for x in range(self.__map_arr.shape[1]):
    #     #     for y in range(self.__map_arr.shape[0]):
    #     #         if (x, y) == self.__start_point_tuple or (x, y) == self.__end_point_tuple:
    #     #             continue
    #     #         arr_value = self.__map_arr[y][x]
    #     #         if arr_value == self.__wall_label:
    #     #             continue
    #     for key in self.__map_arr:
    #         directions = list(self.__map_arr[key].keys())
    #         for dir in directions:
    #             if self.__map_arr[key][dir] == "#":
    #                 continue
    #             else:
    #                 self.save_r_df(dir, self.__map_arr[key][dir])

    # def extract_possible_actions(self, state_key):
    #     # x, y = state_key
    #     if self.__map_arr[state_key] == self.__wall_label:
    #         raise ValueError("It is the wall. (x, y)=(%d, %d)" % (x, y))

    #     # around_map = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]

    #     # possible_actoins_list = [(_x, _y) for _x, _y in around_map if self.__map_arr[_y][_x] != self.__wall_label and self.__map_arr[_y][_x] != self.__start_point_label]
    #     return [k for k,v in self.__map_arr[state_key].items() if v != "#"]
        

    def learn(self, state_key, limit=1000):
        self.t = 1
        while self.t <= limit:
            next_action_list = self.extract_possible_actions(state_key)
            if len(next_action_list):
                action_key = self.select_action(
                    state_key=state_key,
                    next_action_list=next_action_list
                )
                reward_value = self.observe_reward_value(state_key, action_key)

            if len(next_action_list):
                # Max-Q-Value in next action time.
                next_state_key = self.update_state(
                    state_key=state_key,
                    action_key=action_key
                )

                prev_state = state_key
                next_next_action_list = self.extract_possible_actions(next_state_key)
                next_action_key = self.predict_next_action(next_state_key, next_next_action_list)
                next_max_q = self.extract_q_df(next_state_key, next_action_key)
                # x, y = state_key
                # print(self.__map_arr[x][y])
                print(f'At a state {state_key} with action {action_key}, the maximum value (q) at state {next_state_key} with best next_action {next_action_key} is {next_max_q}.')
                # Update Q-Value.
                self.update_q(
                    state_key=state_key,
                    action_key=action_key,
                    reward_value=reward_value,
                    next_max_q=next_max_q
                )
                # Update State.
                state_key = next_state_key

            # Normalize.
            self.normalize_q_value()
            self.normalize_r_value()

            # Vis.
            # self.visualize_learning_result(state_key)
            # Check.
            if self.check_the_end_flag(state_key) is True:
                break

            # Epsode.
            self.t += 1
    

