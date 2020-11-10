import numpy as np
import matplotlib.pyplot as plt
from gamblers_problem import GamblersProblem

gambler1 = GamblersProblem(p_head=0.25)
gambler2 = GamblersProblem(p_head=0.55)

V_state1, pi_state1 = gambler1.value_iteration()
V_state2, pi_state2 = gambler2.value_iteration()
capital = np.arange(1, gambler1.num_states, dtype=int)

fig = plt.figure()

plt.subplot(2, 2, 1)
plt.plot(capital, V_state1[1:])
plt.title("p_head 0.25 - State Values")
plt.xlabel("Capital")
plt.ylabel("Value estimates")

plt.subplot(2, 2, 2)
plt.scatter(capital, pi_state1[1:])
plt.title("p_head 0.25 - Policy")
plt.xlabel("Capital")
plt.ylabel("Final policy (stake)")

plt.subplot(2, 2, 3)
plt.plot(capital, V_state2[1:])
plt.title("p_head 0.55 - State Values")
plt.xlabel("Capital")
plt.ylabel("Value estimates")

plt.subplot(2, 2, 4)
plt.scatter(capital, pi_state2[1:])
plt.title("p_head 0.55 - Policy")
plt.xlabel("Capital")
plt.ylabel("Final policy (stake)")

plt.show()