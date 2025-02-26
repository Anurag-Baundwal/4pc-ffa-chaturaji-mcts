# search.py (MODIFIED - FIX FOR EMPTY root.children)

from board import Board
from mcts import MCTSNode
import torch
import torch.nn.functional as F
from utils import board_to_tensor, move_to_index

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_best_move_mcts(board: Board, network, simulations=250):
    device = next(network.parameters()).device
    root = MCTSNode(board)
    for _ in range(simulations):
        node = root
        while not node.is_leaf():
            node = node.select_child()

        if node.board.is_game_over():
            # NEW: Get game results for terminal node
            game_results = node.board.get_game_result()
            sorted_results = sorted(game_results.items(), key=lambda x: -x[1])
            reward_map = {
                sorted_results[0][0]: 2.0,
                sorted_results[1][0]: 0.5,
                sorted_results[2][0]: -0.5,
                sorted_results[3][0]: -2.0
            }
            value = torch.tensor(
                reward_map.get(root.board.current_player, -2.0),
                device=device
            )
        else:
            with torch.no_grad():
                state_tensor = board_to_tensor(node.board, device=device)
                policy_logits, value = network(state_tensor)
                policy_probs = process_policy(policy_logits, node.board)
            node.expand(policy_probs)
        node.update(value.item())

    # FIX: Handle empty root.children (no legal moves)
    if not root.children:
        return None  # No legal moves available

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