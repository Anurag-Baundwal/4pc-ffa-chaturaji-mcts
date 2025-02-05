# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from self_play import SelfPlay, _generate_game_static #, _process_game_result_static # No longer need to import static methods from SelfPlay
from model import ChaturajiNN
from utils import board_to_tensor, move_to_index
import os, multiprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ChessDataset and train() function remain the same as in your previous train.py

def generate_games_parallel(model_class, simulations_per_move, temp_threshold, num_processes, games_per_process=4): # Pass simulations_per_move, temp_threshold, num_processes as arguments
    processes = num_processes
    total_games = processes * games_per_process
    print(f"Generating {total_games} games across {processes} processes...")
    with multiprocessing.Pool(processes=processes) as pool:
        game_args = [(model_class, simulations_per_move, temp_threshold) for _ in range(total_games)] # Use passed arguments directly
        games_data_list = pool.starmap(_generate_game_static, game_args)
    return [item for sublist in games_data_list for item in sublist]

def train():
    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model directory
    model_dir = '/content/drive/MyDrive/models'
    os.makedirs(model_dir, exist_ok=True)

    network = ChaturajiNN().to(device)
    optimizer = optim.Adam(network.parameters(), lr=0.001, weight_decay=1e-4)

    simulations_per_move_val = 25 # Store simulations in a simple variable
    temp_threshold_val = 5 # Store temp_threshold in a simple variable
    self_play = SelfPlay(network, simulations_per_move=simulations_per_move_val, temp_threshold=temp_threshold_val) # Still instantiate SelfPlay for serial game generation if needed, but not used in parallel part

    # Load existing model if available
    if os.path.exists("model.pth"):
        network.load_state_dict(torch.load("model.pth", map_location=device))

    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} processes for self-play.")


    # Training loop
    for iteration in range(100):
        print(f"---------- ITERATION {iteration+1} ----------")

        # Generate games
        print(f"Generating games in parallel...")
        games = generate_games_parallel(ChaturajiNN, simulations_per_move_val, temp_threshold_val, num_processes, games_per_process=4) # Pass simulations_per_move_val and temp_threshold_val directly


        # # Original serial game generation (comment out)
        # print(f"Generating games...")
        # games = [self_play.generate_game() for _ in range(25)]  # Parallel self-play # changed 20 to 50


        # Rest of the train() function (dataset creation, loader, training epoch, saving model) remains the same


if __name__ == "__main__":
    train()