import chess
import chess.engine
import random
import numpy as np
import pickle

# Simple Q-table as a dictionary
q_table = {}

# Parameters
alpha = 0.1     # Learning rate
gamma = 0.9     # Discount factor
epsilon = 0.2   # Exploration rate
episodes = 1000

# Function to convert board to a hashable state
def board_to_state(board):
    return board.fen()

# Choose action with epsilon-greedy
def choose_action(board, legal_moves):
    if random.random() < epsilon:
        return random.choice(legal_moves)
    state = board_to_state(board)
    if state not in q_table:
        return random.choice(legal_moves)
    move_scores = q_table[state]
    return max(move_scores, key=move_scores.get)

# Reward function
def get_reward(board):
    if board.is_checkmate():
        return 1
    elif board.is_stalemate() or board.is_insufficient_material():
        return 0.5
    elif board.is_game_over():
        return -1
    return 0

# Training loop
for episode in range(episodes):
    board = chess.Board()
    while not board.is_game_over():
        state = board_to_state(board)
        legal_moves = list(board.legal_moves)

        if state not in q_table:
            q_table[state] = {move: 0 for move in legal_moves}

        action = choose_action(board, legal_moves)
        board.push(action)

        reward = get_reward(board)
        next_state = board_to_state(board)
        if next_state not in q_table:
            q_table[next_state] = {move: 0 for move in board.legal_moves}

        if not board.is_game_over():
            next_max = max(q_table[next_state].values()) if q_table[next_state] else 0
        else:
            next_max = 0

        # Q-learning update
        q_table[state][action] += alpha * (reward + gamma * next_max - q_table[state][action])

    if episode % 100 == 0:
        print(f"Episode {episode} completed.")

# Save Q-table
with open("q_chess_agent.pkl", "wb") as f:
    pickle.dump(q_table, f)

