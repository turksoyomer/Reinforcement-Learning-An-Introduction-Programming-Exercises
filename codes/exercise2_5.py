"""Exercise 2.5 (programming) Design and conduct an experiment to demonstrate the
difficulties that sample-average methods have for nonstationary problems. Use a modified
version of the 10-armed testbed in which all the q_star(a) start out equal and then take
independent random walks (say by adding a normally distributed increment with mean
zero and standard deviation 0.01 to all the q_star(a) on each step). Prepare plots like
Figure 2.2 for an action-value method using sample averages, incrementally computed,
and another action-value method using a constant step-size parameter, alpha = 0.1. Use
epsilon = 0.1 and longer runs, say of 10,000 steps."""

import numpy as np
import matplotlib.pyplot as plt

k = 10
q_values = np.zeros((k))
steps = 10000
step_size = 0.1
epsilon = 0.1

incrementally_computed_q = np.zeros((k))
incrementally_computed_n = np.zeros((k))
incrementally_computed_optimal_act_perc = []
incrementally_computed_avg_rewards = []
incrementally_computed_total_reward = 0
incrementally_computed_total_optActions = 0

constant_step_q = np.zeros((k))
constant_step_optimal_act_perc = []
constant_step_avg_rewards = []
constant_step_total_reward = 0
constant_step_total_optActions = 0

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
    incrementally_computed_total_reward += incrementally_computed_r
    incrementally_computed_avg_rewards.append(incrementally_computed_total_reward/step)
    incrementally_computed_total_optActions += 1 if incrementally_computed_a == opt_action else 0
    incrementally_computed_optimal_act_perc.append(incrementally_computed_total_optActions/step)
    incrementally_computed_n[incrementally_computed_a] += 1
    incrementally_computed_q[incrementally_computed_a] += 1/incrementally_computed_n[incrementally_computed_a]*(incrementally_computed_r-incrementally_computed_q[incrementally_computed_a])

    constant_step_r = q_values[constant_step_a]
    constant_step_total_reward += constant_step_r
    constant_step_avg_rewards.append(constant_step_total_reward/step)
    constant_step_total_optActions += 1 if constant_step_a == opt_action else 0
    constant_step_optimal_act_perc.append(constant_step_total_optActions/step)
    constant_step_q[constant_step_a] += step_size*(constant_step_r-constant_step_q[constant_step_a])

fig = plt.figure()

plt.subplot(2, 1, 1)
plt.plot(np.arange(1,steps+1), np.array(incrementally_computed_avg_rewards), 'm', label='incrementally computed')
plt.plot(np.arange(1,steps+1), np.array(constant_step_avg_rewards), 'c', label='constant step size')
plt.legend(loc='upper left', frameon=False)
plt.xlabel("Steps")
plt.ylabel("Average reward")

plt.subplot(2, 1, 2)
plt.plot(np.arange(1,steps+1), np.array(incrementally_computed_optimal_act_perc), 'm')
plt.plot(np.arange(1,steps+1), np.array(constant_step_optimal_act_perc), 'c')
plt.xlabel("Steps")
plt.ylabel("% Optimal action")

plt.show()