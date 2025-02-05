from board import Board
import torch
import numpy as np
from collections import deque
from mcts import MCTSNode
from utils import board_to_tensor, move_to_index

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelfPlay:
    def __init__(self, network, simulations_per_move=100, buffer_size=10000, temp_threshold=5): # Added temp_threshold to init
        self.network = network
        self.device = next(network.parameters()).device  # Get device from network
        self.simulations = simulations_per_move
        self.buffer = deque(maxlen=buffer_size)
        self.temp_threshold = temp_threshold  # Moves before temperature becomes 0

    def generate_game(self, temperature=1.0): # Renamed instance method
        return SelfPlay._generate_game_static(self.network, self.simulations, self.temp_threshold, temperature)

    @staticmethod # Make it a static method
    def _generate_game_static(network, simulations_per_move, temp_threshold, temperature=1.0): # Static method (correct name)
        board = Board()
        game_data = []
        move_count = 0

        while not board.is_game_over():
            # show training progress
            if move_count % 5 == 0:
                print(f"Move {move_count} - {len(game_data)} positions generated")

            root = MCTSNode(board)

            # Run MCTS simulations
            for sim_num in range(simulations_per_move):
                node = root
                search_path = [node]

                # Selection
                while not node.is_leaf():
                    node = node.select_child()
                    search_path.append(node)

                # -- BATCHED EXPANSION AND EVALUATION --
                if not node.board.is_game_over():
                    # 1. Prepare batch for all leaf nodes (just this node in our case)
                    leaf_nodes = [node]
                    device_static = next(network.parameters()).device  # Get the network's device directly
                    batch_states = [board_to_tensor(n.board, device=device) for n in leaf_nodes]
                    batch = torch.cat(batch_states)

                    # 2. Evaluate batch
                    with torch.no_grad():
                        policy_logits, values = network(batch)

                    # 3. Expand leaf nodes
                    for i, leaf_node in enumerate(leaf_nodes):
                        legal_moves = leaf_node.board.get_psuedo_legal_moves(leaf_node.board.current_player)
                        policy_logit = policy_logits[i].squeeze()
                        move_probs = SelfPlay._process_policy_static(policy_logit, legal_moves) # Call static _process_policy_static
                        leaf_node.expand(move_probs)


                # CORRECTED BACKPROPAGATION
                # Get the leaf value (only 1 in our case)
                if leaf_nodes:
                    leaf_value = values[0].item()
                    # Update all nodes in the search path with this value
                    for n in search_path:
                        n.update(leaf_value)

            # -- END OF BATCHED EXPANSION AND EVALUATION --

            # Store training data
            policy = SelfPlay._get_action_probs_static(root, temperature=(1.0 if move_count < temp_threshold else 0.0)) # Call static _get_action_probs_static
            game_data.append((board.copy(), policy, root.board.current_player))

            # Make move
            move = SelfPlay._choose_move_static(root, temperature) # Call static _choose_move_static
            board.make_move(move)
            move_count += 1

        # Assign final values
        game_data = SelfPlay._process_game_result_static(game_data, board) # Call static version and assign back to game_data
        print(f"Game finished. {len(game_data)} positions generated in total.")
        return game_data

    @staticmethod # Make static - removed self
    def _process_game_result_static(game_data, final_board): # Static version - removed self
        winner = final_board.get_winner()
        for i, (board, policy, player) in enumerate(game_data):
            # Value target: +1 if player wins, -1 if loses, 0 for draws
            value = 1.0 if player == winner else -1.0 if winner is not None else 0.0
            game_data[i] = (board, policy, value)
        return game_data

    @staticmethod # Make static
    def _process_policy_static(policy_logits, legal_moves): # Static version
        move_indices = [move_to_index(move) for move in legal_moves]
        legal_mask = torch.zeros_like(policy_logits)
        legal_mask[move_indices] = 1.0
        probs = torch.softmax(policy_logits * legal_mask - 1e10*(1 - legal_mask), dim=0)
        return {move: probs[move_to_index(move)].item() for move in legal_moves}

    @staticmethod # Make static
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

    @staticmethod # Make static
    def _choose_move_static(root, temperature): # Static version
        move_probs = SelfPlay._get_action_probs_static(root, temperature) # Use static version
        moves = list(move_probs.keys())
        probs = list(move_probs.values())
        return np.random.choice(moves, p=probs)


    def _process_policy(self, policy_logits, legal_moves): # Instance version - will not be used directly in multiprocessing, but kept for potential future single-process usage or reference
        return SelfPlay._process_policy_static(policy_logits, legal_moves) # Call static version

    def _get_action_probs(self, root, temperature=1.0): # Instance version
        return SelfPlay._get_action_probs_static(root, temperature) # Call static version

    def _choose_move(self, root, temperature): # Instance version
        return SelfPlay._choose_move_static(root, temperature) # Call static version


    def _evaluate_node(self, node): # Instance method - kept as is, not used directly in static game generation
        if node.board.is_game_over():
            winner = node.board.get_winner()
            return 1.0 if winner == node.board.current_player else -1.0
        else:
            # Create tensor directly on network's device
            state_tensor = board_to_tensor(node.board, device=self.device)
            with torch.no_grad():
                _, value = self.network(state_tensor)
            return value.item()