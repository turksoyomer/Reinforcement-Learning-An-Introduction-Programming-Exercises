"""Exercise 2.11 (programming) Make a figure analogous to Figure 2.6 for the nonstationary
case outlined in Exercise 2.5. Include the constant-step-size epsilon-greedy algorithm with
alpha=0.1. Use runs of 200,000 steps and, as a performance measure for each algorithm and
parameter setting, use the average reward over the last 100,000 steps."""

import numpy as np
import matplotlib.pyplot as plt

k = 10
q_values = np.zeros((k))
steps = 200000
step_size = 0.1
epsilon_values = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4]
incrementally_computed_avg_rewards = []
constant_step_avg_rewards = []

for epsilon in epsilon_values:
    incrementally_computed_q = np.zeros((k))
    incrementally_computed_n = np.zeros((k))
    incrementally_computed_total_reward = 0

    constant_step_q = np.zeros((k))
    constant_step_total_reward = 0

    for step in range(1, steps+1):
        increments = np.random.normal(loc=0.0, scale=0.01, size=k)
        q_values += increments
        opt_action = np.argmax(increments)

        if np.random.random() > epsilon:
            incrementally_computed_a = np.argmax(incrementally_computed_q)
            constant_step_a = np.argmax(constant_step_q)
        else:
            random_action = np.random.randint(k)
            incrementally_computed_a = random_action
            constant_step_a = random_action

        incrementally_computed_r = q_values[incrementally_computed_a]
        incrementally_computed_total_reward += incrementally_computed_r if step > 100000 else 0
        incrementally_computed_n[incrementally_computed_a] += 1
        incrementally_computed_q[incrementally_computed_a] += 1/incrementally_computed_n[incrementally_computed_a]*(incrementally_computed_r-incrementally_computed_q[incrementally_computed_a])

        constant_step_r = q_values[constant_step_a]
        constant_step_total_reward += constant_step_r if step > 100000 else 0
        constant_step_q[constant_step_a] += step_size*(constant_step_r-constant_step_q[constant_step_a])

    incrementally_computed_avg_reward = incrementally_computed_total_reward / 100000
    incrementally_computed_avg_rewards.append(incrementally_computed_avg_reward)
    constant_step_avg_reward = constant_step_total_reward / 100000
    constant_step_avg_rewards.append(constant_step_avg_reward)

fig = plt.figure()

plt.plot(epsilon_values, incrementally_computed_avg_rewards, 'm', label='incrementally computed')
plt.plot(epsilon_values, constant_step_avg_rewards, 'c', label='constant step size')
plt.legend(loc='upper left', frameon=False)
plt.xlabel("epsilon values")
plt.ylabel("Average reward over last 100000 steps")

plt.show()