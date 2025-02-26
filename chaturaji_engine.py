# chaturaji_engine.py

import random
import cProfile
from enum import Enum
import time

import torch
from model import ChaturajiNN # Import the NN model
from search import get_best_move_mcts # Import the MCTS search function
from utils import *
from board import Board
from pieces import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    board = Board()
    total_execution_time = 0
    num_searches = 0

    # Instantiate the Neural Network Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
    network = ChaturajiNN().to(device)
    # Load model weights if you have a pre-trained model
    # if os.path.exists("model.pth"):
    #     network.load_state_dict(torch.load("model.pth", map_location=device))
    #     network.eval() # Set to evaluation mode if loaded for inference

    for i in range(100):
        print(f"Searching for the best move for {board.current_player}.")
        start_time = time.time()
        best_move = get_best_move_mcts(board, network) # Pass the NN model to MCTS, scores are not returned now
        end_time = time.time()
        # nodes = get_nodes() # Removed as get_nodes() is undefined and not directly relevant to MCTS in this context.
        execution_time = end_time - start_time
        total_execution_time += execution_time
        num_searches += 1

        print("Search completed")
        # print(f"Number of nodes visited: {nodes + 1}") # Removed, nodes is not tracked.
        print(f"Search simulations completed.") # More appropriate for MCTS
        print(f"Execution time for this search: {execution_time} seconds")

        # Calculate and print average execution time
        avg_execution_time = total_execution_time / num_searches
        print(f"Average execution time so far: {avg_execution_time} seconds")

        # nps = (nodes + 1) / (execution_time+0.001) # Removed, nodes is not tracked.
        # print(f"Nodes per second (NPS): {nps}") # Removed, nodes is not tracked.
        if best_move:
            print(f"Best move: ({best_move.from_loc.row}, {best_move.from_loc.col}) to ({best_move.to_loc.row}, {best_move.to_loc.col})") # Removed Scores: {scores}
            print(f"san string of best_move: {get_san_string(best_move, board)}")
            print(f"uci string of best_move: {get_uci_string(best_move, board)}")
            # What happens if we play the best move
            board.make_move(best_move)
            print("Board state after playing best move: ")
            board.print_board()
            print("Turn: ", board.current_player)
            print("Active players: ", board.active_players)
            # print("board.evaluate() output: ", board.evaluate(), "\n") # Still using handcrafted eval? Consider NN value output in MCTS - Commented out for now.
            print(f"Points: {board.player_points}")
            print(f"Total moves played in this game so far: {i}")
        else:
            print(f"No valid moves found for {board.current_player}.")
            if not board.is_game_over(): # Only resign if game is not already over
                print(f"{board.current_player} resigns.")
                board.resign() # Current player resigns
            else:
                print("Game is already over.")

        if board.is_game_over():
            print(f"Game over! Termination reason: {board.termination_reason}") # Print termination reason
            print(f"Final scores: {board.player_points}")
            break # End game loop

    if (len(board.active_players) == 1): # Redundant check, game_over already checked in loop
        print(f"Winner: {board.get_winner()}") # Print winner based on points