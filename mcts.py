import math
import torch
from board import Board

class MCTSNode:
    def __init__(self, board: Board, parent=None, move=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.children = []
        self.visit_count = 0
        self.total_value = 0.0  # From root player's perspective
        self.prior = 0.0

    def is_leaf(self):
        return not self.children

    def select_child(self, c_puct=1.0):
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            if child.visit_count == 0:
                u = c_puct * child.prior * math.sqrt(self.visit_count + 1e-8) / 1.0
                score = u
            else:
                q = child.total_value / child.visit_count
                u = c_puct * child.prior * math.sqrt(self.visit_count) / (child.visit_count + 1)
                score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self, policy_probs):
        legal_moves = self.board.get_psuedo_legal_moves(self.board.current_player)
        for move, prob in policy_probs.items():
            new_board = self.board.copy()
            new_board.make_move(move)
            child = MCTSNode(new_board, parent=self, move=move)
            child.prior = prob
            self.children.append(child)

    def update(self, value):
        self.visit_count += 1
        self.total_value += value
        if self.parent:
            self.parent.update(value)