import random

class Policy:
    def __init__(self, q_learning_instance):
        self.q_learner = q_learning_instance
        self.q_table = self.q_learner.q_df
        self.best_actions = {}
        self.possible_actions = {} 
        self.max_q_val = {} # Store maximum Q_value for a particular state (comparing amongst all possible actions)
        
        self.initializePolicy()

    def printQTable(self):
        # Helpful debugging tool
        for index, row in self.q_table.iterrows():
            state, action, q_val = row['state_key'], row['action_key'], row['q_value']
            print(f'{state} -> {action} has a value of {q_val}')


    def initializeNextStates(self, next_states):
        '''
            next_states = arr of next states \n
            Recursively initializes the states, by setting a state's possible actions and their maximum q-value to 0
            (at the moment).
        '''
        for state in next_states:
            if state in self.possible_actions and state in self.max_q_val:
                continue
            self.possible_actions[state] = self.q_learner.extract_possible_actions(state)
            self.max_q_val[state] = 0
            self.initializeNextStates(self.possible_actions[state])


    def initializePolicy(self):
        # Determines a policy from its given q-table.
        self.initializeNextStates([self.q_table.iloc[0]['state_key']])
        for index, row in self.q_table.iterrows():
            state, action, q_val = row['state_key'], row['action_key'], row['q_value']
            if ((state not in self.max_q_val) or (state in self.max_q_val and q_val > self.max_q_val[state])) and q_val > 0:
                self.max_q_val[state] = q_val
                self.best_actions[state] = action
    
        

    def getAction(self, state, in_cycle=False):
        # Determines what action to take at a given state
        # in_cycle refers to if we've been to this state before
        # Returns in tuple (x, y) form
        if state in self.best_actions and not in_cycle:
            return self.best_actions[state]
        elif state not in self.possible_actions:
            print(f"Invalid state. No found actions to perform at state {state}.")
            return (-1,-1)
        else:
            return random.choice(self.possible_actions[state])
    
    def atGoal(self, state):
        return self.q_learner.check_the_end_flag(state)
    

