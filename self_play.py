# self_play.py (CORRECTED AND COMPLETE)
import math
import torch
import numpy as np
from collections import deque
from board import Board
from mcts import MCTSNode
from utils import board_to_tensor, move_to_index
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelfPlay:
    def __init__(self, network, simulations_per_move=100, buffer_size=10000):
        self.network = network
        self.device = next(network.parameters()).device  # Get device from network
        self.simulations = simulations_per_move
        self.buffer = deque(maxlen=buffer_size)
        self.temp_threshold = 5  # Moves before temperature becomes 0

    def generate_game(self, temperature=1.0):
        board = Board()
        game_data = []  # Temporary storage for the game
        move_count = 0

        while not board.is_game_over():
            # show training progress
            if move_count % 5 == 0:
                print(f"Move {move_count} - {len(game_data)} positions generated")

            root = MCTSNode(board)

            # Run MCTS simulations
            for sim_num in range(self.simulations):
                node = root
                search_path = [node]

                # Selection
                while not node.is_leaf(): # add and not node.board.is_game_over():?
                    node = node.select_child()
                    search_path.append(node)

                # -- BATCHED EXPANSION AND EVALUATION --
                if not node.board.is_game_over():
                    # 1. Prepare batch for all leaf nodes (just this node in our case)
                    leaf_nodes = [node]
                    device = next(self.network.parameters()).device  # Get the network's device directly
                    # UNSQUEEZE HERE, adding a batch dimension:
                    batch_states = [board_to_tensor(n.board, device=device).unsqueeze(0) for n in leaf_nodes]
                    batch = torch.cat(batch_states)


                    # 2. Evaluate batch
                    with torch.no_grad():
                        policy_logits, values = self.network(batch)

                    # 3. Expand leaf nodes
                    for i, leaf_node in enumerate(leaf_nodes):
                        legal_moves = leaf_node.board.get_psuedo_legal_moves(leaf_node.board.current_player)
                        policy_logit = policy_logits[i].squeeze()
                        move_probs = self._process_policy(policy_logit, legal_moves)
                        leaf_node.expand(move_probs)

                # CORRECTED BACKPROPAGATION
                # Get the leaf value (only 1 in our case)
                if leaf_nodes:
                    leaf_value = values[0].item()
                    # Update all nodes in the search path with this value
                    for n in search_path:
                        n.update(leaf_value)

            # -- END OF BATCHED EXPANSION AND EVALUATION --

            # Get action probabilities and choose a move
            policy = self._get_action_probs(root, temperature=(1.0 if move_count < self.temp_threshold else 0.0))

            # Store the board and policy *before* making the move
            game_data.append((board.copy(), policy, board.current_player))

            move = self._choose_move(root, temperature)
            board.make_move(move)
            move_count += 1

        # Process the game result *after* the game is finished
        self._process_game_result(game_data, board)
        print(f"Game finished. {len(game_data)} positions generated in total.")  # Added
        # --- PRINT FINAL SCORES ---
        print("Final Scores:")
        for player, score in board.player_points.items():
            print(f"  {player}: {score}")
        # --- END PRINT FINAL SCORES ---
        return game_data # This isn't directly used by train() anymore, but it can be helpful for debugging/analysis.


    def _process_policy(self, policy_logits, legal_moves):
        move_indices = [move_to_index(move) for move in legal_moves]
        legal_mask = torch.zeros_like(policy_logits)
        legal_mask[move_indices] = 1.0
        probs = torch.softmax(policy_logits * legal_mask - 1e10*(1 - legal_mask), dim=0)
        return {move: probs[move_to_index(move)].item() for move in legal_moves}

    def _get_action_probs(self, root, temperature=1.0):
        visit_counts = np.array([child.visit_count for child in root.children])
        if temperature == 0:
            probs = np.zeros_like(visit_counts)
            probs[np.argmax(visit_counts)] = 1.0
        else:
            # Use log-sum-exp trick for numerical stability
            visit_counts = visit_counts + 1e-6  # Add a small epsilon to visit_counts
            log_counts = np.log(visit_counts) / temperature
            # Find max for stability (subtract it from all elements)
            max_log_counts = np.max(log_counts)
            log_counts -= max_log_counts
            # Exponentiate and normalize
            probs = np.exp(log_counts)
            probs /= np.sum(probs)
        return {child.move: prob for child, prob in zip(root.children, probs)}

    def _choose_move(self, root, temperature):
        move_probs = self._get_action_probs(root, temperature)
        moves = list(move_probs.keys())
        probs = list(move_probs.values())
        return np.random.choice(moves, p=probs)

    def _process_game_result(self, game_data, final_board):
        # --- CHANGED SECTION START ---
        final_scores = final_board.player_points.items()
        sorted_players = sorted(final_scores, key=lambda item: item[1], reverse=True)

        rewards = {
            sorted_players[0][0]: 2.0,
            sorted_players[1][0]: 0.5,
            sorted_players[2][0]: -0.5,
            sorted_players[3][0]: -2.0,
        }

        for player, _ in sorted_players:
            if player not in rewards:
                rewards[player] = -2.0

        for board, policy, player in game_data:
            reward = rewards[player]
            self.buffer.append((board, policy, float(reward)))
        # --- CHANGED SECTION END ---

    def _evaluate_node(self, node): # NO LONGER CALLED - but kept for clarity
        if node.board.is_game_over():
            winner = node.board.get_winner()
            return 1.0 if winner == node.board.current_player else -1.0
        else:
            state_tensor = board_to_tensor(node.board, device=self.device).unsqueeze(0)
            with torch.no_grad():
                _, value = self.network(state_tensor)
            return value.item()