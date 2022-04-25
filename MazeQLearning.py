from prelab_files.demo_maze_greedy_q_learning import MazeGreedyQLearning

class MazeQLearning(MazeGreedyQLearning):
    '''
        Contains my modifications to any of the methods inherited
        from MazeGreedyQLearning or QLearning, the base class.
    '''

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
            if self.check_the_end_flag(prev_state) is True:
                break

            # Epsode.
            self.t += 1
    pass

