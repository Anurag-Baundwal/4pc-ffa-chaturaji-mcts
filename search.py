from board import Board
from mcts import MCTSNode
import torch

from utils import board_to_tensor

import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_best_move_mcts(board: Board, network, simulations=250): # Reduced from 800 to 250 for faster nn training
    root = MCTSNode(board)
    for _ in range(simulations):
        node = root
        while not node.is_leaf():
            node = node.select_child()
        
        if node.board.is_game_over():
            # value = 1.0 if node.board.get_winner() == root.board.current_player else -1.0 # bug
            value = torch.tensor(1.0) if node.board.get_winner() == root.board.current_player else torch.tensor(-1.0) # Convert to tensor # bug fixed
        else:
            with torch.no_grad():
                state_tensor = board_to_tensor(node.board).to(device) # --- ADDED .to(device) HERE ---
                policy_logits, value = network(state_tensor)
                policy_probs = process_policy(policy_logits, node.board)
            
            node.expand(policy_probs)
        
        node.update(value.item())
    
    best_child = max(root.children, key=lambda c: c.visit_count)
    return best_child.move

def process_policy(logits, board):
    legal_moves = board.get_psuedo_legal_moves(board.current_player)
    move_indices = [move_to_index(move) for move in legal_moves]
    mask = torch.zeros_like(logits).squeeze()
    mask[move_indices] = 1.0
    probs = F.softmax(logits.squeeze() * mask - 1e10*(1-mask), dim=0)
    return {move: probs[idx].item() for idx, move in enumerate(legal_moves)}

def move_to_index(move):
    fr = move.from_loc.row * 8 + move.from_loc.col
    to = move.to_loc.row * 8 + move.to_loc.col
    return fr * 64 + to