import numpy as np
import matplotlib.pyplot as plt

class GamblersProblem:
    def __init__(self, p_head=0.4):
        self.num_states = 100
        self.num_actions = 51
        self.p_head = p_head
        self.p_tail = 1 - p_head
        self.theta = 0.0000001
        self.V_state = np.zeros(self.num_states)
        self.pi_state = np.ones(self.num_states, dtype=int)

    def valid_actions(self, state):
        action_list = np.arange(1, min(state,self.num_states-state)+1, dtype=int)
        return action_list

    def expectation(self, state, action):
        returned_expectation = 0
        state_win = state+action
        state_lose = state-action
        reward = 0
        if state_win == 100:
            reward_win = 1
            returned_expectation += self.p_head * (reward_win)
            returned_expectation += self.p_tail * (reward + self.V_state[state_lose])
        else:
            returned_expectation += self.p_head * (reward + self.V_state[state_win])
            returned_expectation += self.p_tail * (reward + self.V_state[state_lose])
        return returned_expectation

    def policy(self):
        for state in range(1, self.num_states):
            action_list = self.valid_actions(state)
            best_action = -1
            best_value = -float("inf")
            for action in action_list:
                expectation = self.expectation(state, action)
                if expectation >= best_value:
                    best_value = expectation
                    best_action = action
            self.pi_state[state] = best_action

    def value_iteration(self):
        while True:
            delta = 0
            for state in range(1, self.num_states):
                v = self.V_state[state]
                action_list = self.valid_actions(state)
                new_v = -float("inf")
                for action in action_list:
                    expectation = self.expectation(state, action)
                    if expectation >= new_v:
                        new_v = expectation
                self.V_state[state] = new_v
                delta = max(delta, abs(v-new_v))
            if delta < self.theta:
                break
        self.policy()
        return self.V_state.copy(), self.pi_state.copy()


if __name__ == "__main__":
    gambler = GamblersProblem()
    V_state, pi_state = gambler.value_iteration()

    capital = np.arange(1, gambler.num_states, dtype=int)
    plt.plot(capital, V_state[1:])
    plt.xlabel("Capital")
    plt.ylabel("Value estimates")
    plt.show()

    plt.plot(capital, pi_state[1:])
    plt.xlabel("Capital")
    plt.ylabel("Final policy (stake)")
    plt.show()
