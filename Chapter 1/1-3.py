import numpy as np
q_table = np.zeros((5, 5))

# Define an example state, action, and reward
state = 2
action = 1
reward = 10
learning_rate = 0.1
discount_factor = 0.9

# Simplified Q-learning update rule:
# Q(s, a) = Q(s, a) + learning_rate * [reward + discount_factor * max(Q(s') - Q(s, a))]
q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[state]) - q_table[state, action])

print("Updated Q-Table:", q_table)
