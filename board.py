# 4 player chaturaji chess engine - board.py 

import copy
from pieces import Piece, PieceType, Player

class BoardLocation:
    def __init__(self, row, col):
        self.row = row
        self.col = col

class Move:
    def __init__(self, from_loc, to_loc, promotion_piece_type=None):
        self.from_loc = from_loc
        self.to_loc = to_loc
        self.promotion_piece_type = promotion_piece_type
        
class Board:
    def __init__(self):
        self.board = [[None] * 8 for _ in range(8)]
        self.active_players = set(Player)
        self.player_points = {player: 0 for player in Player}
        self.current_player = Player.RED
        self.setup_initial_board()
        # New properties
        self.position_history = []
        self.fifty_move_counter = 0
        self.termination_reason = None
        self.game_history = []  # store moves
        self._previous_board_state = None  # store previous state
        #-------

    def setup_initial_board(self):
        # Red pieces
        self.board[7][0] = Piece(Player.RED, PieceType.ROOK)
        self.board[7][1] = Piece(Player.RED, PieceType.KNIGHT)
        self.board[7][2] = Piece(Player.RED, PieceType.BISHOP)
        self.board[7][3] = Piece(Player.RED, PieceType.KING)
        for col in range(0, 4):
            self.board[6][col] = Piece(Player.RED, PieceType.PAWN)

        # Blue pieces
        self.board[0][0] = Piece(Player.BLUE, PieceType.ROOK)
        self.board[1][0] = Piece(Player.BLUE, PieceType.KNIGHT)
        self.board[2][0] = Piece(Player.BLUE, PieceType.BISHOP)
        self.board[3][0] = Piece(Player.BLUE, PieceType.KING)
        for row in range(0, 4):
            self.board[row][1] = Piece(Player.BLUE, PieceType.PAWN)

        # Yellow pieces
        self.board[0][7] = Piece(Player.YELLOW, PieceType.ROOK)
        self.board[0][6] = Piece(Player.YELLOW, PieceType.KNIGHT)
        self.board[0][5] = Piece(Player.YELLOW, PieceType.BISHOP)
        self.board[0][4] = Piece(Player.YELLOW, PieceType.KING)
        for col in range(4, 8):
            self.board[1][col] = Piece(Player.YELLOW, PieceType.PAWN)

        # Green pieces
        self.board[4][7] = Piece(Player.GREEN, PieceType.KING)
        self.board[5][7] = Piece(Player.GREEN, PieceType.BISHOP)
        self.board[6][7] = Piece(Player.GREEN, PieceType.KNIGHT)
        self.board[7][7] = Piece(Player.GREEN, PieceType.ROOK)
        for row in range(4, 8):
            self.board[row][6] = Piece(Player.GREEN, PieceType.PAWN)

    def is_valid_square(self, row, col):
        if not (0 <= row < 8 and 0 <= col < 8):
            return False
        return True
    
    def copy(self):
        new_board = Board()
        new_board.board = [
            [
                Piece(piece.player, piece.piece_type) if piece else None
                for piece in row
            ]
            for row in self.board
        ]
        new_board.active_players = set(self.active_players)
        new_board.player_points = dict(self.player_points)
        new_board.current_player = self.current_player
        new_board.position_history = list(self.position_history)
        new_board.fifty_move_counter = self.fifty_move_counter
        new_board.termination_reason = self.termination_reason
        new_board.game_history = list(self.game_history)
        # No need to copy _previous_board_state, it's only for undoing the *last* move
        return new_board
    
    def get_psuedo_legal_moves(self, player): # TODO: check if the player is dead?
        psuedo_legal_moves = []

        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                # if piece and piece.player == player:
                if piece and piece.player == player and piece.is_dead == False:
                    match piece.piece_type:
                        case PieceType.PAWN:
                            psuedo_legal_moves.extend(self.get_pawn_moves(row, col))
                        case PieceType.KNIGHT:
                            psuedo_legal_moves.extend(self.get_knight_moves(row, col))
                        case PieceType.BISHOP:
                            psuedo_legal_moves.extend(self.get_bishop_moves(row, col))
                        case PieceType.ROOK:
                            psuedo_legal_moves.extend(self.get_rook_moves(row, col))
                        case PieceType.KING:
                            psuedo_legal_moves.extend(self.get_king_moves(row, col))
                            # psuedo_legal_moves = self.get_king_moves(row, col) + psuedo_legal_moves


        return psuedo_legal_moves

    def get_pawn_moves(self, row, col):
        moves = []
        player = self.board[row][col].player

        promotion_row_red = 0
        promotion_col_blue = 7
        promotion_row_yellow = 7
        promotion_col_green = 0

        # Non-capture moves and promotion check
        match player:
            case Player.RED:
                if row > 0 and self.board[row - 1][col] is None:
                    to_row, to_col = row - 1, col
                    if to_row == promotion_row_red:
                        for piece_type in [PieceType.ROOK]: # Promotion piece types
                            moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col), piece_type))
                    else:
                        moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col)))
            case Player.BLUE:
                if col < 7 and self.board[row][col + 1] is None:
                    to_row, to_col = row, col + 1
                    if to_col == promotion_col_blue:
                        for piece_type in [PieceType.ROOK]: # Promotion piece types
                            moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col), piece_type))
                    else:
                        moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col)))
            case Player.YELLOW:
                if row < 7 and self.board[row + 1][col] is None:
                    to_row, to_col = row + 1, col
                    if to_row == promotion_row_yellow:
                        for piece_type in [PieceType.ROOK]: # Promotion piece types
                            moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col), piece_type))
                    else:
                        moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col)))
            case Player.GREEN:
                if col > 0 and self.board[row][col - 1] is None:
                    to_row, to_col = row, col - 1
                    if to_col == promotion_col_green:
                        for piece_type in [PieceType.ROOK]: # Promotion piece types
                            moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col), piece_type))
                    else:
                        moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col)))

        # Capture moves and promotion check
        match player:
            case Player.RED:
                if row > 0 and col > 0 and self.board[row - 1][col - 1] and self.board[row - 1][col - 1].player != player:
                    to_row, to_col = row - 1, col - 1
                    if to_row == promotion_row_red:
                        for piece_type in [PieceType.ROOK]: # Promotion piece types
                            moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col), piece_type))
                    else:
                        moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col)))
                if row > 0 and col < 7 and self.board[row - 1][col + 1] and self.board[row - 1][col + 1].player != player:
                    to_row, to_col = row - 1, col + 1
                    if to_row == promotion_row_red:
                        for piece_type in [PieceType.ROOK]: # Promotion piece types
                            moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col), piece_type))
                    else:
                        moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col)))
            case Player.BLUE:
                if row > 0 and col < 7 and self.board[row - 1][col + 1] and self.board[row - 1][col + 1].player != player:
                    to_row, to_col = row - 1, col + 1
                    if to_col == promotion_col_blue:
                        for piece_type in [PieceType.ROOK]: # Promotion piece types
                            moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col), piece_type))
                    else:
                        moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col)))
                if row < 7 and col < 7 and self.board[row + 1][col + 1] and self.board[row + 1][col + 1].player != player:
                    to_row, to_col = row + 1, col + 1
                    if to_col == promotion_col_blue:
                        for piece_type in [PieceType.ROOK]: # Promotion piece types
                            moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col), piece_type))
                    else:
                        moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col)))
            case Player.YELLOW:
                if row < 7 and col > 0 and self.board[row + 1][col - 1] and self.board[row + 1][col - 1].player != player:
                    to_row, to_col = row + 1, col - 1
                    if to_row == promotion_row_yellow:
                        for piece_type in [PieceType.ROOK]: # Promotion piece types
                            moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col), piece_type))
                    else:
                        moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col)))
                if row < 7 and col < 7 and self.board[row + 1][col + 1] and self.board[row + 1][col + 1].player != player:
                    to_row, to_col = row + 1, col + 1
                    if to_row == promotion_row_yellow:
                        for piece_type in [PieceType.ROOK]: # Promotion piece types
                            moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col), piece_type))
                    else:
                        moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col)))
            case Player.GREEN:
                if row > 0 and col > 0 and self.board[row - 1][col - 1] and self.board[row - 1][col - 1].player != player:
                    to_row, to_col = row - 1, col - 1
                    if to_col == promotion_col_green:
                        for piece_type in [PieceType.ROOK]: # Promotion piece types
                            moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col), piece_type))
                    else:
                        moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col)))
                if row < 7 and col > 0 and self.board[row + 1][col - 1] and self.board[row + 1][col - 1].player != player:
                    to_row, to_col = row + 1, col - 1
                    if to_col == promotion_col_green:
                        for piece_type in [PieceType.ROOK]: # Promotion piece types
                            moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col), piece_type))
                    else:
                        moves.append(Move(BoardLocation(row, col), BoardLocation(to_row, to_col)))

        return moves

    def get_knight_moves(self, row, col):
        moves = []
        player = self.board[row][col].player
        for r, c in [(row-2, col-1), (row-2, col+1), (row-1, col-2), (row-1, col+2), (row+1, col-2), (row+1, col+2), (row+2, col-1), (row+2, col+1)]:
            # if 0 <= r < 14 and 0 <= c < 14 and (self.board[r][c] is None or self.board[r][c].player != player):
            if self.is_valid_square(r, c) and (self.board[r][c] is None or self.board[r][c].player != player):
                moves.append(Move(BoardLocation(row, col), BoardLocation(r, c)))
        return moves

    def get_bishop_moves(self, row, col):
        moves = []
        player = self.board[row][col].player
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            r, c = row + dr, col + dc
            while self.is_valid_square(r, c):
                if self.board[r][c] is None:
                    moves.append(Move(BoardLocation(row, col), BoardLocation(r, c)))
                elif self.board[r][c].player != player:
                    moves.append(Move(BoardLocation(row, col), BoardLocation(r, c)))
                    break
                else: 
                    break
                r, c = r + dr, c + dc
        return moves

    def get_rook_moves(self, row, col):
        moves = []
        player = self.board[row][col].player
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            while self.is_valid_square(r, c):
                if self.board[r][c] is None:
                    moves.append(Move(BoardLocation(row, col), BoardLocation(r, c)))
                elif self.board[r][c].player != player:
                    moves.append(Move(BoardLocation(row, col), BoardLocation(r, c)))
                    break
                else:
                    break
                r, c = r + dr, c + dc
        return moves

    def get_king_moves(self, row, col):
        moves = []
        player = self.board[row][col].player
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if self.is_valid_square(r, c) and (self.board[r][c] is None or self.board[r][c].player != player):
                    moves.append(Move(BoardLocation(row, col), BoardLocation(r, c)))
        return moves

    def make_move(self, move):
        # 1. STORE THE CURRENT BOARD STATE
        self._previous_board_state = self.copy()

        piece = self.board[move.from_loc.row][move.from_loc.col]
        self.board[move.from_loc.row][move.from_loc.col] = None 
        captured_piece = self.board[move.to_loc.row][move.to_loc.col]
        captured_player = captured_piece.player if captured_piece else None

        # if captured_piece and (captured_piece.is_dead == False or captured_piece.piece_type == PieceType.DEAD_KING): ##### THIS IS NOT GIVING POINTS FOR CAPTURING DEAD KINGS BECAUSE is_dead is True # FIXED 
        if captured_piece and captured_piece.is_dead == False:
            captured_player = captured_piece.player
            self.player_points[piece.player] += self.get_piece_capture_value(captured_piece)

            if captured_piece.piece_type == PieceType.KING:
              self.eliminate_player(captured_player)
              
        self.board[move.to_loc.row][move.to_loc.col] = piece
        if move.promotion_piece_type:
            self.board[move.to_loc.row][move.to_loc.col].piece_type = move.promotion_piece_type

        # Update fifty_move_counter
        is_pawn_move = (piece.piece_type == PieceType.PAWN)
        is_capture = (captured_piece is not None)
        if is_pawn_move or is_capture:
            self.fifty_move_counter = 0
        else:
            self.fifty_move_counter += 1

        # Update position history
        position_key = self.get_position_key()
        self.position_history.append(position_key)
        self.game_history.append(move)

        self.current_player = Player((self.current_player.value + 1) % 4)
        while (self.current_player not in self.active_players):
            self.current_player = Player((self.current_player.value + 1) % 4)

        return captured_piece, []  # No need to return eliminated players

    def undo_move(self, move, captured_piece=None, eliminated_players=None): # Remove unused params
        # 1. RESTORE THE PREVIOUS BOARD STATE
        if self._previous_board_state is not None:
            # Copy all relevant attributes from the saved state
            self.board = self._previous_board_state.board
            self.active_players = self._previous_board_state.active_players
            self.player_points = self._previous_board_state.player_points
            self.current_player = self._previous_board_state.current_player
            self.position_history = self._previous_board_state.position_history
            self.fifty_move_counter = self._previous_board_state.fifty_move_counter
            self.termination_reason = self._previous_board_state.termination_reason
            self.game_history = self._previous_board_state.game_history

            self._previous_board_state = None # Clear after restoring.

    def get_position_key(self):
        return tuple(tuple(
            (piece.player.value if piece else None, piece.piece_type.value if piece else None)
            for piece in row
        ) for row in self.board)

    def get_piece_value(self, piece):
        match piece.piece_type:
            case PieceType.PAWN:
                return 1
            case PieceType.KNIGHT:
                return 3
            case PieceType.BISHOP:
                return 5
            case PieceType.ROOK:
                return 5
            case PieceType.QUEEN:
                return 9
            case PieceType.ONE_POINT_QUEEN:
                return 11
            case PieceType.KING:
                return 3
            case PieceType.DEAD_KING:
                return 0

    def get_piece_capture_value(self, piece):
        match piece.piece_type:
            case PieceType.PAWN:
                return 1
            case PieceType.KNIGHT:
                return 3
            case PieceType.BISHOP:
                return 5
            case PieceType.ROOK:
                return 5
            case PieceType.QUEEN:
                return 9
            case PieceType.ONE_POINT_QUEEN:
                return 1
            case PieceType.KING:
                return 3
            case PieceType.DEAD_KING:
                return 3
    
    # eval ideas
    # for king -> +10 cp for every friendly piece adjancent to king and -10 for every enemy piece
    # for pawns -> bonus for moving forward towards promotion | blocked pawn penalty
    # for pieces -> small penalty for being on back rank
    def evaluate(self):
        scores = {player: 0 for player in Player}

        king_coords = {player: None for player in Player}
        coord_sums = {player: [0, 0] for player in Player}
        piece_counts = {player: 0 for player in Player}
        king_present = {player: False for player in Player}

        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece:
                    coord_sums[piece.player][0] += row
                    coord_sums[piece.player][1] += col
                    piece_counts[piece.player] += 1
                    if not piece.is_dead and piece.player in self.active_players:
                      # scores[piece.player] += self.get_piece_value(piece) * 1.25 
                      scores[piece.player] += self.get_piece_value(piece)

                      if piece.piece_type == PieceType.KNIGHT or piece.piece_type == PieceType.BISHOP:
                          if ((piece.player == Player.RED and row == 7) # on back rank
                          or (piece.player == Player.YELLOW and row == 0)
                          or (piece.player == Player.GREEN and col == 7)
                          or (piece.player == Player.BLUE and col == 0)):
                              scores[piece.player] -= 0.4 # penalty for undeveloped pieces

                      
                      if piece.piece_type == PieceType.KING:
                          for dr in [-1, 0, 1]:
                              for dc in [-1, 0, 1]:
                                  if dr == 0 and dc == 0:
                                      continue
                                  r, c = row + dr, col + dc
                                  if self.is_valid_square(r, c) and (self.board[r][c] is not None):
                                      if self.board[r][c].player == piece.player:
                                          if self.board[r][c].piece_type == PieceType.PAWN:
                                              scores[piece.player] += 0.2 # king near friendly pawn
                                          else:
                                              scores[piece.player] += 0.05 # king near friendly piece
                                      else:
                                          if self.board[r][c].player not in self.active_players:
                                              scores[piece.player] += 0.15 # shelter from dead pieces
                                          else:
                                              scores[piece.player] -= 0.15 # king near enemy piece
                          # Store the coordinates of the king for each color
                          king_coords[piece.player] = [row, col]
                          
                          king_present[piece.player] = True
                          
                      if piece.piece_type == PieceType.PAWN:
                          if piece.player == Player.RED:
                              scores[piece.player] += 0.2*(6-row)
                              if self.is_valid_square((row-1), col) and self.board[row-1][col] != None and self.board[row-1][col].player != piece.player:
                                  scores[piece.player] -= 0.2 # blocked pawn
                              for dr in [-1]:
                                  for dc in [-1, 1]:
                                      r, c = row + dr, col + dc
                                      if self.is_valid_square(r, c):
                                          if self.board[r][c] != None:
                                              target = self.board[r][c]
                                              if target.player == piece.player:
                                                  if target.piece_type == PieceType.BISHOP or target.piece_type == PieceType.KNIGHT:
                                                      scores[piece.player] += 0.2 # piece on outpost
                                              else:
                                                  scores[piece.player] += 0.2 # attacking enemy piece
                                                  if target.piece_type == PieceType.KING:
                                                      scores[piece.player] += 0.1 # attacking enemy king
                                                      scores[target.player] -= 0.5 # king in danger - avoid getting attacked by enemy pawns
                          elif piece.player == Player.BLUE:
                              scores[piece.player] += 0.2*(col-1)
                              if self.is_valid_square(row, (col+1)) and self.board[row][col+1] != None and self.board[row][col+1].player != piece.player:
                                  scores[piece.player] -= 0.2
                              for dr in [-1, 1]:
                                  for dc in [1]:
                                      r, c = row + dr, col + dc
                                      if self.is_valid_square(r, c):
                                          if self.board[r][c] != None:
                                              target = self.board[r][c]
                                              if target.player == piece.player:
                                                  if target.piece_type == PieceType.BISHOP or target.piece_type == PieceType.KNIGHT:
                                                      scores[piece.player] += 0.2 # piece on outpost
                                              else:
                                                  scores[piece.player] += 0.2 # attacking enemy piece
                                                  if target.piece_type == PieceType.KING:
                                                      scores[piece.player] += 0.1 # attacking enemy king
                                                      scores[target.player] -= 0.5 # king in danger - avoid getting attacked by enemy pawns
                          elif piece.player == Player.YELLOW:
                              scores[piece.player] += 0.2*(row-1)
                              if self.is_valid_square((row+1), col) and self.board[row+1][col] != None and self.board[row+1][col].player != piece.player:
                                  scores[piece.player] -= 0.2
                              for dr in [1]:
                                  for dc in [-1, 1]:
                                      r, c = row + dr, col + dc
                                      if self.is_valid_square(r, c):
                                          if self.board[r][c] != None:
                                              target = self.board[r][c]
                                              if target.player == piece.player:
                                                  if target.piece_type == PieceType.BISHOP or target.piece_type == PieceType.KNIGHT:
                                                      scores[piece.player] += 0.2 # piece on outpost
                                              else:
                                                  scores[piece.player] += 0.2 # attacking enemy piece
                                                  if target.piece_type == PieceType.KING:
                                                      scores[piece.player] += 0.1 # attacking enemy king
                                                      scores[target.player] -= 0.5 # king in danger - avoid getting attacked by enemy pawns
                          elif piece.player == Player.GREEN:
                              scores[piece.player] += 0.2*(6-col)
                              if self.is_valid_square(row, (col-1)) and self.board[row][col-1] != None and self.board[row][col-1].player != piece.player:
                                  scores[piece.player] -= 0.2
                              for dr in [-1, 1]:
                                  for dc in [-1]:
                                      r, c = row + dr, col + dc
                                      if self.is_valid_square(r, c):
                                          if self.board[r][c] != None:
                                              target = self.board[r][c]
                                              if target.player == piece.player:
                                                  if target.piece_type == PieceType.BISHOP or target.piece_type == PieceType.KNIGHT:
                                                      scores[piece.player] += 0.2 # piece on outpost
                                              else:
                                                  scores[piece.player] += 0.2 # attacking enemy piece
                                                  if target.piece_type == PieceType.KING:
                                                      scores[piece.player] += 0.1 # attacking enemy king
                                                      scores[target.player] -= 0.5 # king in danger - avoid getting attacked by enemy pawns
        
        for player in Player:
            if not (king_present[player] == True and player in self.active_players):
                scores[player] = -999    
        for player in Player:
            scores[player] += self.player_points[player]
            # scores[player] -= 20*1.25
            scores[player] -= 20

        # Final rounding of scores to 2 decimal places
        for player in Player:
            scores[player] = round(scores[player], 2)
        return scores

    def is_game_over(self):
        if len(self.active_players) <= 1:
            self.termination_reason = "elimination"
            return True

        if self.fifty_move_counter >= 50:
            self.termination_reason = "fifty_move_rule"
            return True

        current_position_key = self.get_position_key()
        if self.position_history.count(current_position_key) >= 3:
            self.termination_reason = "threefold_repetition"
            return True

        return False
    
    def get_game_result(self):
        if not self.is_game_over():
            return None

        results = {}
        for player in Player:
            results[player] = self.player_points[player]  # Start with base points

        if self.termination_reason in ["fifty_move_rule", "threefold_repetition"]:
            # Draw conditions: remaining players get +3
            for player in self.active_players:
                results[player] += 3
        # else (game ended by elimination): no additional points needed, base points are final

        return results
    
    def get_winner(self):
        if not self.is_game_over():
            return None

        game_results = self.get_game_result()
        sorted_results = sorted(game_results.items(), key=lambda item: item[1], reverse=True)
        return sorted_results[0][0]

    def eliminate_player(self, player):
        # print(f"{player} eliminated!")
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece:
                    if not piece.is_dead and piece.player == player:
                      if piece.piece_type != PieceType.KING:
                          piece.is_dead = True
                      else:
                          piece.piece_type = PieceType.DEAD_KING 
        self.active_players.remove(player)

    def print_board(self):
        # Red pieces
        red_king = "\033[31m♔\033[0m"
        red_queen = "\033[31m♕\033[0m"
        red_rook = "\033[31m♖\033[0m"
        red_bishop = "\033[31m♗\033[0m"
        red_knight = "\033[31m♘\033[0m"
        red_pawn = "\033[31m♙\033[0m"

        # Yellow pieces
        yellow_king = "\033[33m♔\033[0m"
        yellow_queen = "\033[33m♕\033[0m"
        yellow_rook = "\033[33m♖\033[0m"
        yellow_bishop = "\033[33m♗\033[0m"
        yellow_knight = "\033[33m♘\033[0m"
        yellow_pawn = "\033[33m♙\033[0m"

        # Blue pieces
        blue_king = "\033[34m♔\033[0m"
        blue_queen = "\033[34m♕\033[0m"
        blue_rook = "\033[34m♖\033[0m"
        blue_bishop = "\033[34m♗\033[0m"
        blue_knight = "\033[34m♘\033[0m"
        blue_pawn = "\033[34m♙\033[0m"

        # Green pieces
        green_king = "\033[32m♔\033[0m"
        green_queen = "\033[32m♕\033[0m"
        green_rook = "\033[32m♖\033[0m"
        green_bishop = "\033[32m♗\033[0m"
        green_knight = "\033[32m♘\033[0m"
        green_pawn = "\033[32m♙\033[0m"
        
        # Dead pieces
        dead_king = "♔"
        dead_queen = "♕"
        dead_rook = "♖"
        dead_bishop = "♗"
        dead_knight = "♘"
        dead_pawn = "♙"
        print("   0  1  2  3  4  5  6  7")
        for row in range(8):
            print(row, end=" ")
            for col in range(8):
                piece = self.board[row][col]
                if piece:
                    if piece.is_dead: # or piece.piece_type == PieceType.DEAD_KING:
                        match piece.piece_type:
                            case PieceType.PAWN:
                                symbol = dead_pawn
                            case PieceType.KNIGHT:
                                symbol = dead_knight
                            case PieceType.BISHOP:
                                symbol = dead_bishop
                            case PieceType.ROOK:
                                symbol = dead_rook
                            case PieceType.QUEEN:
                                symbol = dead_queen
                            case PieceType.KING:
                                symbol = dead_king
                            case PieceType.ONE_POINT_QUEEN:
                                symbol = dead_queen
                    else:
                        match piece.player:
                            case Player.RED:
                                match piece.piece_type:
                                    case PieceType.PAWN:
                                        symbol = red_pawn
                                    case PieceType.KNIGHT:
                                        symbol = red_knight
                                    case PieceType.BISHOP:
                                        symbol = red_bishop
                                    case PieceType.ROOK:
                                        symbol = red_rook
                                    case PieceType.QUEEN:
                                        symbol = red_queen
                                    case PieceType.KING:
                                        symbol = red_king
                                    case PieceType.ONE_POINT_QUEEN:
                                        symbol = red_queen
                                    case PieceType.DEAD_KING:
                                        symbol = red_king
                            case Player.BLUE:
                                match piece.piece_type:
                                    case PieceType.PAWN:
                                        symbol = blue_pawn
                                    case PieceType.KNIGHT:
                                        symbol = blue_knight
                                    case PieceType.BISHOP:
                                        symbol = blue_bishop
                                    case PieceType.ROOK:
                                        symbol = blue_rook
                                    case PieceType.QUEEN:
                                        symbol = blue_queen
                                    case PieceType.KING:
                                        symbol = blue_king
                                    case PieceType.ONE_POINT_QUEEN:
                                        symbol = blue_queen
                                    case PieceType.DEAD_KING:
                                        symbol = blue_king
                            case Player.YELLOW:
                                match piece.piece_type:
                                    case PieceType.PAWN:
                                        symbol = yellow_pawn
                                    case PieceType.KNIGHT:
                                        symbol = yellow_knight
                                    case PieceType.BISHOP:
                                        symbol = yellow_bishop
                                    case PieceType.ROOK:
                                        symbol = yellow_rook
                                    case PieceType.QUEEN:
                                        symbol = yellow_queen
                                    case PieceType.KING:
                                        symbol = yellow_king
                                    case PieceType.ONE_POINT_QUEEN:
                                        symbol = yellow_queen
                                    case PieceType.DEAD_KING:
                                        symbol = yellow_king
                            case Player.GREEN:
                                match piece.piece_type:
                                    case PieceType.PAWN:
                                        symbol = green_pawn
                                    case PieceType.KNIGHT:
                                        symbol = green_knight
                                    case PieceType.BISHOP:
                                        symbol = green_bishop
                                    case PieceType.ROOK:
                                        symbol = green_rook
                                    case PieceType.QUEEN:
                                        symbol = green_queen
                                    case PieceType.KING:
                                        symbol = green_king
                                    case PieceType.ONE_POINT_QUEEN:
                                        symbol = green_queen
                                    case PieceType.DEAD_KING:
                                        symbol = green_king
                    print(f"[{symbol}]", end="")
                else:
                    print("[ ]", end="")
            print()

    def resign(self): 
        player = self.current_player
        self.eliminate_player(player) 
        if len(self.active_players) != 1:
            self.current_player = Player((self.current_player.value + 1) % 4)
            while (self.current_player not in self.active_players):
                self.current_player = Player((self.current_player.value + 1) % 4)
