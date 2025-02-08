# self_play.py (Corrected)
import math
import torch
import numpy as np
from collections import deque
from mcts import MCTSNode
from utils import board_to_tensor, move_to_index
from model import ChaturajiNN  # Import the model class
from board import Board

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _generate_game_static(network_class, state_dict, simulations_per_move, temp_threshold, temperature=1.0):
    # Reconstruct the network *inside* the child process
    network = network_class().to(device)  # Create a NEW instance
    network.load_state_dict(state_dict)  # Load the weights
    network.eval()  # IMPORTANT: Set to evaluation mode

    board = Board()
    game_data = []
    move_count = 1

    while not board.is_game_over():
        if move_count % 5 == 0:
            print(f"Move {move_count} - {len(game_data)} positions generated")

        root = MCTSNode(board)

        for sim_num in range(simulations_per_move):
            node = root
            search_path = [node]

            while not node.is_leaf():
                node = node.select_child()
                search_path.append(node)

            if not node.board.is_game_over():
                leaf_nodes = [node]
                batch_states = [board_to_tensor(n.board, device=device) for n in leaf_nodes]  # Use device
                batch = torch.cat(batch_states)

                with torch.no_grad():
                    policy_logits, values = network(batch)  # Now network is correct

                for i, leaf_node in enumerate(leaf_nodes):
                    legal_moves = leaf_node.board.get_psuedo_legal_moves(leaf_node.board.current_player)
                    policy_logit = policy_logits[i].squeeze()
                    move_probs = _process_policy_static(policy_logit, legal_moves)
                    leaf_node.expand(move_probs)

            if leaf_nodes:
                leaf_value = values[0].item()
                for n in search_path:
                    n.update(leaf_value)

        policy = _get_action_probs_static(root, temperature=(1.0 if move_count < temp_threshold else 0.0))
        game_data.append((board.copy(), policy, root.board.current_player))

        move = _choose_move_static(root, temperature=(1.0 if move_count < temp_threshold else 0.0))
        board.make_move(move)
        move_count += 1

    game_data = _process_game_result_static(game_data, board)
    print(f"Game finished. {len(game_data)} positions generated in total.")
    return game_data



# Define _process_game_result_static as a module-level function
def _process_game_result_static(game_data, final_board): # Static version - removed self
    winner = final_board.get_winner()
    for i, (board, policy, player) in enumerate(game_data):
        # Value target: +1 if player wins, -1 if loses, 0 for draw
        value = 1.0 if player == winner else -1.0 if winner is not None else 0.0
        game_data[i] = (board, policy, value)
    return game_data

# Define _process_policy_static as a module-level function
def _process_policy_static(policy_logits, legal_moves): # Static version
    move_indices = [move_to_index(move) for move in legal_moves]
    legal_mask = torch.zeros_like(policy_logits)
    legal_mask[move_indices] = 1.0
    probs = torch.softmax(policy_logits * legal_mask - 1e10*(1 - legal_mask), dim=0)
    return {move: probs[move_to_index(move)].item() for move in legal_moves}

# Define _get_action_probs_static as a module-level function
def _get_action_probs_static(root, temperature=1.0): # Static version
    visit_counts = np.array([child.visit_count for child in root.children])
    if temperature == 0:
        probs = np.zeros_like(visit_counts)
        probs[np.argmax(visit_counts)] = 1.0
    else:
        # Use log-sum-exp trick for numerical stability
        visit_counts = visit_counts + 1e-6
        log_counts = np.log(visit_counts) / temperature
        # Find max for stability
        max_log_counts = np.max(log_counts)
        log_counts -= max_log_counts
        # Exponentiate and normalize
        probs = np.exp(log_counts)
        probs /= np.sum(probs)
    return {child.move: prob for child, prob in zip(root.children, probs)}

# Define _choose_move_static as a module-level function
def _choose_move_static(root, temperature): # Static version
    move_probs = _get_action_probs_static(root, temperature) # Use module-level function
    moves = list(move_probs.keys())
    probs = list(move_probs.values())
    return np.random.choice(moves, p=probs)


class SelfPlay:
    def __init__(self, network, simulations_per_move=100, buffer_size=10000, temp_threshold=5):
        self.network = network
        self.device = next(network.parameters()).device
        self.simulations = simulations_per_move
        self.buffer = deque(maxlen=buffer_size)
        self.temp_threshold = temp_threshold

    def generate_game(self, temperature=1.0):
        return _generate_game_static(ChaturajiNN, self.network.state_dict(), self.simulations, self.temp_threshold, temperature)


    def _process_policy(self, policy_logits, legal_moves):
        return _process_policy_static(policy_logits, legal_moves)

    def _get_action_probs(self, root, temperature=1.0):
        return _get_action_probs_static(root, temperature)

    def _choose_move(self, root, temperature):
        return _choose_move_static(root, temperature)


    def _evaluate_node(self, node):
        if node.board.is_game_over():
            winner = node.board.get_winner()
            return 1.0 if winner == node.board.current_player else -1.0
        else:
            state_tensor = board_to_tensor(node.board, device=self.device)
            with torch.no_grad():
                _, value = self.network(state_tensor)
            return value.item()