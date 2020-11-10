import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns

class JacksCarRental:
    def __init__(self):
        self.rent_credit = 10
        self.cost_move_car = 2
        self.lambda_req1, self.lambda_req2 = 3, 4
        self.lambda_return1, self.lambda_return2 = 3, 2
        self.max_car = 20
        self.max_move = 5
        self.num_states = (21,21)
        self.num_actions = 11
        self.V_state = np.zeros(self.num_states)
        self.pi_state = np.zeros(self.num_states, dtype=np.int) + 5
        self.theta = 0.1
        self.gamma = 0.9
        self.poisson_cache = dict()

    def valid_actions(self, state):
        # Negative action means moving cars from loc1 to loc2. Positive is opposite of that.
        action_list = np.arange(max(-self.max_move,-state[0],state[1]-self.max_car), min(self.max_move,state[1],self.max_car-state[0])+1)
        action_list += self.max_move
        return action_list

    def poisson_prob(self, n, lam):
        key = (n, lam)
        if key not in self.poisson_cache:
            self.poisson_cache[key] = poisson.pmf(n, lam)
        return self.poisson_cache[key]

    def expectation(self, state, action_index):
        action = action_index - self.max_move
        after_move1 = state[0] + action
        after_move2 = state[1] - action
        reward_move = -self.cost_move_car * abs(action)
        returned_expectation = 0
        for rent1 in range(after_move1):
            for rent2 in range(after_move2):
                after_rent1 = after_move1 - rent1
                after_rent2 = after_move2 - rent2
                reward_rent = self.rent_credit * (rent1+rent2)
                prob_rent1 = self.poisson_prob(rent1, self.lambda_req1)
                prob_rent2 = self.poisson_prob(rent2, self.lambda_req2)
                prob_rent = prob_rent1 * prob_rent2
                for return1 in range(self.max_car-after_rent1+1):
                    for return2 in range(self.max_car-after_rent2+1):
                        after_return1 = after_rent1 + return1
                        after_return2 = after_rent2 + return2
                        prob_return1 = self.poisson_prob(return1, self.lambda_return1)
                        prob_return2 = self.poisson_prob(return2, self.lambda_return2)
                        prob_return = prob_return1 * prob_return2
                        prob = prob_rent * prob_return
                        next_state = (after_return1, after_return2)
                        reward = reward_rent + reward_move
                        next_V_value = self.V_state[next_state].copy()
                        returned_expectation += prob*(reward+self.gamma*next_V_value)
        return returned_expectation

    def policy_evaluation(self):
        while True:
            delta = 0
            for i in range(self.num_states[0]):
                for j in range(self.num_states[1]):
                    state = (i,j)
                    v = self.V_state[state].copy()
                    action = self.pi_state[state]
                    new_v = self.expectation(state, action)
                    self.V_state[state] = new_v
                    delta = max(delta, abs(v-new_v))
            print(delta)
            if delta < self.theta:
                break
        print(self.V_state)

    def policy_improvement(self):
        policy_stable = True
        for i in range(self.num_states[0]):
            for j in range(self.num_states[1]):
                state = (i,j)
                action_list = self.valid_actions(state)
                old_action = self.pi_state[state]
                max_expectation = -float("inf")
                best_action = -1
                for action in action_list:
                    returned_expectation = self.expectation(state, action)
                    if returned_expectation > max_expectation:
                        max_expectation = returned_expectation
                        best_action = action
                self.pi_state[state] = best_action
                if old_action != best_action:
                    policy_stable = False
        print(self.pi_state)
        return policy_stable

    def policy_iteration(self):
        while True:
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            if policy_stable:
                return self.V_state.copy(), self.pi_state.copy()


if __name__ == "__main__":
    jack = JacksCarRental()
    V_state, pi_state = jack.policy_iteration()

    ax1 = sns.heatmap(pi_state-jack.max_move)
    ax1.invert_yaxis()
    ax1.set_title('Optimal Policy')
    ax1.set_xlabel('#Cars at second location')
    ax1.set_ylabel('#Cars at first location')
    plt.show()

    ax2 = sns.heatmap(V_state)
    ax2.invert_yaxis()
    ax2.set_title('Optimal State Value')
    ax2.set_xlabel('#Cars at second location')
    ax2.set_ylabel('#Cars at first location')
    plt.show()
