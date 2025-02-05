# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from self_play import SelfPlay, _generate_game_static, _process_game_result_static
from model import ChaturajiNN # Import model class
from utils import board_to_tensor, move_to_index
import os, multiprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ChessDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        board, policy, value = self.buffer[idx]
        state_tensor = board_to_tensor(board, device) # ADDED device here
        policy_tensor = self._policy_to_tensor(policy, board)
        return state_tensor, policy_tensor, torch.tensor(value, dtype=torch.float32)

    def _policy_to_tensor(self, policy, board):
        legal_moves = board.get_psuedo_legal_moves(board.current_player)
        policy_tensor = torch.zeros(4096)  # 8x8x64 = 4096 possible moves
        for move, prob in policy.items():
            idx = move_to_index(move)
            policy_tensor[idx] = prob
        return policy_tensor

def train():
    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model directory
    model_dir = '/content/drive/MyDrive/models'
    os.makedirs(model_dir, exist_ok=True)

    network = ChaturajiNN().to(device)
    optimizer = optim.Adam(network.parameters(), lr=0.001, weight_decay=1e-4)
    self_play = SelfPlay(network, simulations_per_move=25)

    # Load existing model if available
    if os.path.exists("model.pth"):
        network.load_state_dict(torch.load("model.pth", map_location=device))

    num_processes = multiprocessing.cpu_count() # Use all CPUs available
    print(f"Using {num_processes} processes for self-play.")

    def generate_games_parallel(iteration_num, games_per_process=4): # Generate games in parallel, reduced games_per_process to 4, processes to 8 -> 32 games total
        processes = num_processes
        total_games = processes * games_per_process # e.g. 8 processes * 4 games/process = 32 games
        print(f"Generating {total_games} games across {processes} processes...")
        with multiprocessing.Pool(processes=processes) as pool:
            # Pass model class, simulations, temp_threshold
            game_args = [(ChaturajiNN, self_play.simulations, self_play.temp_threshold) for _ in range(total_games)] # Arguments for each game generation
            games_data_list = pool.starmap(_generate_game_static, game_args) # Generate games in parallel
        return [item for sublist in games_data_list for item in sublist] # Flatten list of lists

    # Training loop
    for iteration in range(100):
        print(f"---------- ITERATION {iteration+1} ----------")

        # Generate games
        print(f"Generating games in parallel...")
        games = generate_games_parallel(iteration) # Use parallel game generation

        # # Original serial game generation (comment out)
        # print(f"Generating games...")
        # games = [self_play.generate_game() for _ in range(25)]  # Parallel self-play # changed 20 to 50

        # Create dataset
        dataset = ChessDataset([item for game in games for item in game])
        loader = DataLoader(dataset, batch_size=256, shuffle=True) # batch size increased from 32 to 256 for colab gpus

        print(f"Generated {len(dataset)} data points from games.")

        scaler = torch.GradScaler(enabled=torch.cuda.is_available()) # Explicitly enable only when CUDA is available

        # Training epoch
        network.train()
        for epoch in range(5):  # 5 epochs per iteration
            total_loss = 0.0
            num_batches = len(loader)
            for batch_idx, (states, policies, values) in enumerate(loader):
                states = states.to(device)
                policies = policies.to(device)
                values = values.to(device)

                # Forward pass
                optimizer.zero_grad()

                with torch.autocast():
                    policy_pred, value_pred = network(states)

                    # Calculate losses
                    policy_loss = torch.mean(-torch.sum(policies * torch.log_softmax(policy_pred, dim=1), dim=1))
                    value_loss = torch.nn.functional.mse_loss(value_pred.squeeze(), values)
                    loss = policy_loss + value_loss

                scaler.scale(loss).backward()  # Scale the loss
                scaler.step(optimizer)  # Unscale and update
                scaler.update()

                total_loss += loss.item()

                # Print progress within epoch
                if (batch_idx + 1) % 10 == 0:  # Print every 10 batches
                    print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}")

            print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

        # Save model
        # torch.save(network.state_dict(), f"model.pth")
        # print(f"Model saved after iteration {iteration+1}")

        # Save every 10 iterations
        if (iteration+1) % 10 == 0:
            save_path = f'{model_dir}/chaturaji_iter_{iteration+1}.pth'
            torch.save(network.module.state_dict() if isinstance(network, torch.nn.DataParallel) else network.state_dict(), save_path) # handles dataparallel models
        print(f"Model saved after iteration {iteration+1}")

if __name__ == "__main__":
    train()