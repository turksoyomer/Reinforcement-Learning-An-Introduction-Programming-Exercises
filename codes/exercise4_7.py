"""Exercise 4.7 (programming) Write a program for policy iteration and re-solve Jack’s car
rental problem with the following changes. One of Jack’s employees at the first location
rides a bus home each night and lives near the second location. She is happy to shuttle
one car to the second location for free. Each additional car still costs $2, as do all cars
moved in the other direction. In addition, Jack has limited parking space at each location.
If more than 10 cars are kept overnight at a location (after any moving of cars), then an
additional cost of $4 must be incurred to use a second parking lot (independent of how
many cars are kept there). These sorts of nonlinearities and arbitrary dynamics often
occur in real problems and cannot easily be handled by optimization methods other than
dynamic programming. To check your program, first replicate the results given for the
original problem."""

import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns
from jacks_car_rental import JacksCarRental

class NonLinearJacksCarRental(JacksCarRental):
    def __init__(self):
        super().__init__()
        self.cost_parking = 4
        self.max_space = 10

    def expectation(self, state, action_index):
        action = action_index - self.max_move
        after_move1 = state[0] + action
        after_move2 = state[1] - action
        reward_move = -self.cost_move_car * abs(action) if action != -1 else 0
        reward_move -= self.cost_parking if after_move1 > self.max_space else 0
        reward_move -= self.cost_parking if after_move2 > self.max_space else 0
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


if __name__ == "__main__":
    jack = NonLinearJacksCarRental()
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
