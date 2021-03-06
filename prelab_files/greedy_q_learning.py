# -*- coding: utf-8 -*-
import random
import numpy as np
from prelab_files.q_learning import QLearning


class GreedyQLearning(QLearning):
    '''
    ε-greedy Q-Learning.

    Epsilon Greedy Q-Leanring algorithm is a typical off-policy algorithm. 
    In this paradigm, stochastic searching and deterministic searching can 
    coexist by hyperparameter `epsilon_greedy_rate` that is probability 
    that agent searches greedy. Greedy searching is deterministic in the sense 
    that policy of agent follows the selection that maximizes the Q-Value.

    References:
        - Agrawal, S., & Goyal, N. (2011). Analysis of Thompson sampling for the multi-armed bandit problem. arXiv preprint arXiv:1111.1797.
        - Bubeck, S., & Cesa-Bianchi, N. (2012). Regret analysis of stochastic and nonstochastic multi-armed bandit problems. arXiv preprint arXiv:1204.5721.
        - Chapelle, O., & Li, L. (2011). An empirical evaluation of thompson sampling. In Advances in neural information processing systems (pp. 2249-2257).
        - Du, K. L., & Swamy, M. N. S. (2016). Search and optimization by metaheuristics (p. 434). New York City: Springer.
        - Kaufmann, E., Cappe, O., & Garivier, A. (2012). On Bayesian upper confidence bounds for bandit problems. In International Conference on Artificial Intelligence and Statistics (pp. 592-600).
        - Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.
        - Richard Sutton and Andrew Barto (1998). Reinforcement Learning. MIT Press.
        - Watkins, C. J. C. H. (1989). Learning from delayed rewards (Doctoral dissertation, University of Cambridge).
        - Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.
        - White, J. (2012). Bandit algorithms for website optimization. ” O’Reilly Media, Inc.”.

    '''
    
    # ε-greedy rate.
    __epsilon_greedy_rate = 0.75

    def get_epsilon_greedy_rate(self):
        ''' getter '''
        if isinstance(self.__epsilon_greedy_rate, float) is True:
            return self.__epsilon_greedy_rate
        else:
            raise TypeError("The type of __epsilon_greedy_rate must be float.")

    def set_epsilon_greedy_rate(self, value):
        ''' setter '''
        if isinstance(value, float) is True:
            self.__epsilon_greedy_rate = value
        else:
            raise TypeError("The type of __epsilon_greedy_rate must be float.")

    epsilon_greedy_rate = property(get_epsilon_greedy_rate, set_epsilon_greedy_rate)

    def select_action(self, state_key, next_action_list):
        '''
        Select action by Q(state, action).
        
        Concreat method.

        ε-greedy.

        Args:
            state_key:              The key of state.
            next_action_list:       The possible action in `self.t+1`.
                                    If the length of this list is 0, all action should be possible.

        Returns:
            The key of action.

        '''
        epsilon_greedy_flag = bool(np.random.binomial(n=1, p=self.epsilon_greedy_rate))
        
        if epsilon_greedy_flag is False:
            action_key = random.choice(next_action_list)
        else:
            action_key = self.predict_next_action(state_key, next_action_list)
        return action_key

    def learn(self, state_key, limit=1000):
        '''
        Learning and searching the optimal solution.
        
        Args:
            state_key:      Initial state.
            limit:          The maximum number of iterative updates based on value iteration algorithms.
        '''
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

                next_next_action_list = self.extract_possible_actions(next_state_key)
                next_action_key = self.predict_next_action(next_state_key, next_next_action_list)
                next_max_q = self.extract_q_df(next_state_key, next_action_key)

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
            self.visualize_learning_result(state_key)
            # Check.
            if self.check_the_end_flag(state_key) is True:
                break

            # Epsode.
            self.t += 1
