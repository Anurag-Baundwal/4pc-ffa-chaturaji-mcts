from board import Board
import torch
import numpy as np
from collections import deque
from mcts import MCTSNode
from utils import board_to_tensor, move_to_index

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
        game_data = []
        move_count = 0

        while not board.is_game_over():
            # show training progress
            if move_count % 5 == 0:
                print(f"Move {move_count} - {len(game_data)} positions generated")
            
            
            root = MCTSNode(board)
            
            print(f"  Running MCTS for move {move_count + 1}...")  

            # Run MCTS simulations
            for sim_num in range(self.simulations):
                if (sim_num + 1) % 25 == 0:  # Print every 25 simulations
                    print(f"    Simulation {sim_num + 1}/{self.simulations}")

                node = root
                search_path = [node]

                # Selection
                while not node.is_leaf(): # add and not node.board.is_game_over():?
                    node = node.select_child()
                    search_path.append(node)

                # # Expansion
                # if not node.board.is_game_over():
                #     state_tensor = board_to_tensor(node.board)
                #     with torch.no_grad():
                #         policy_logits, value = self.network(state_tensor)
                    
                #     # Convert policy to move distribution
                #     legal_moves = node.board.get_psuedo_legal_moves(node.board.current_player)
                #     move_probs = self._process_policy(policy_logits.squeeze(), legal_moves)
                #     node.expand(move_probs)

                # -- BATCHED EXPANSION AND EVALUATION --
                if not node.board.is_game_over():
                    # 1. Prepare batch for all leaf nodes (just this node in our case)
                    leaf_nodes = [node]
                    device = next(self.network.parameters()).device  # Get the network's device directly
                    batch_states = [board_to_tensor(n.board, device=device) for n in leaf_nodes]                    
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

                # # Backpropagation (using values from the batch)
                # # value = self._evaluate_node(node) # Removed as it is no longer used when we are using a value from the batch
                # for i, n in enumerate(search_path):
                #     # Check if n is in leaf_nodes and use the corresponding value
                #     if n in leaf_nodes:
                #         value_index = leaf_nodes.index(n)
                #         n.update(values[value_index].item())
                #     else:
                #         n.update(values[i].item())

                # CORRECTED BACKPROPAGATION
                # Get the leaf value (only 1 in our case)
                if leaf_nodes:
                    leaf_value = values[0].item()
                    # Update all nodes in the search path with this value
                    for n in search_path:
                        n.update(leaf_value)

            # -- END OF BATCHED EXPANSION AND EVALUATION --

            # Store training data
            # policy = self._get_action_probs(root, temperature=(1.0 if move_count < self.temp_threshold else 0.001))
            policy = self._get_action_probs(root, temperature=(1.0 if move_count < self.temp_threshold else 0.0))
            game_data.append((board.copy(), policy, root.board.current_player))

            # Make move
            move = self._choose_move(root, temperature)
            board.make_move(move)
            move_count += 1

        # Assign final values
        self._process_game_result(game_data, board)
        print(f"Game finished. {len(game_data)} positions generated in total.")  # Added  
        return game_data

    def _process_policy(self, policy_logits, legal_moves):
        move_indices = [move_to_index(move) for move in legal_moves]
        legal_mask = torch.zeros_like(policy_logits)
        legal_mask[move_indices] = 1.0
        probs = torch.softmax(policy_logits * legal_mask - 1e10*(1 - legal_mask), dim=0)
        return {move: probs[move_to_index(move)].item() for move in legal_moves}

    # def _get_action_probs(self, root, temperature=1.0):
    #     visit_counts = np.array([child.visit_count for child in root.children])
    #     if temperature == 0:
    #         probs = np.zeros_like(visit_counts)
    #         probs[np.argmax(visit_counts)] = 1.0
    #     else:
    #         visit_counts = visit_counts ** (1 / temperature)
    #         probs = visit_counts / np.sum(visit_counts)
    #     return {child.move: prob for child, prob in zip(root.children, probs)}
    # # BUG:
    # (.venv) C:\Users\dell3\source\repos3\4pc-ffa-chaturaji-mcts>python train.py
    # Iteration 1: Generating games...
    # C:\Users\dell3\source\repos3\4pc-ffa-chaturaji-mcts\self_play.py:75: RuntimeWarning: overflow encountered in power
    #   visit_counts = visit_counts ** (1 / temperature)
    # C:\Users\dell3\source\repos3\4pc-ffa-chaturaji-mcts\self_play.py:76: RuntimeWarning: invalid value encountered in divide
    #   probs = visit_counts / np.sum(visit_counts)

    # bugfix:
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
        winner = final_board.get_winner()  # You'll need to implement get_winner() in Board
        for i, (board, policy, player) in enumerate(game_data):
            # Value target: +1 if player wins, -1 if loses, 0 for draws
            value = 1.0 if player == winner else -1.0 if winner is not None else 0.0
            self.buffer.append((board, policy, value))

    def _evaluate_node(self, node):
        if node.board.is_game_over():
            winner = node.board.get_winner()
            return 1.0 if winner == node.board.current_player else -1.0
        else:
            state_tensor = board_to_tensor(node.board).to(self.device)  # Use self.device here
            with torch.no_grad():
                _, value = self.network(state_tensor)
            return value.item()