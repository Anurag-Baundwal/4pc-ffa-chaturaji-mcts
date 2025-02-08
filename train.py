# train.py (Corrected)

import multiprocessing  # Import multiprocessing at the VERY top
multiprocessing.set_start_method('spawn', force=True) # Set start method to 'spawn' as the ABSOLUTE FIRST line, and use force=True

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from self_play import SelfPlay, _generate_game_static #, _process_game_result_static # No longer need to import static methods from SelfPlay
from model import ChaturajiNN
from utils import board_to_tensor, move_to_index
import os

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

def generate_games_parallel(network_class, state_dict, simulations_per_move, temp_threshold, num_processes, games_per_process=4):
    processes = num_processes
    total_games = processes * games_per_process
    print(f"Generating {total_games} games across {processes} processes...")
    with multiprocessing.Pool(processes=processes) as pool:
        # Pass the CLASS, STATE_DICT, and other parameters
        game_args = [(network_class, state_dict, simulations_per_move, temp_threshold, 1.0) for _ in range(total_games)]
        games_data_list = pool.starmap(_generate_game_static, game_args)
    return [item for sublist in games_data_list for item in sublist]

def train():
    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model directory
    model_dir = '/content/drive/MyDrive/models' #  Change this to where you want to save the model weights. 
    os.makedirs(model_dir, exist_ok=True)

    network = ChaturajiNN().to(device)
    optimizer = optim.Adam(network.parameters(), lr=0.001, weight_decay=1e-4)

    simulations_per_move_val = 25
    temp_threshold_val = 5
    self_play = SelfPlay(network, simulations_per_move=simulations_per_move_val, temp_threshold=temp_threshold_val)

    # Load existing model if available
    if os.path.exists("model.pth"):
        network.load_state_dict(torch.load("model.pth", map_location=device))

    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} processes for self-play.")


    for iteration in range(100):
        print(f"---------- ITERATION {iteration+1} ----------")

        print(f"Generating games in parallel...")
        # Pass network CLASS and STATE_DICT
        games = generate_games_parallel(ChaturajiNN, network.state_dict(), simulations_per_move_val, temp_threshold_val, num_processes, games_per_process=4)

        dataset = ChessDataset([item for game in games for item in game])
        loader = DataLoader(dataset, batch_size=256, shuffle=True)

        print(f"Generated {len(dataset)} data points from games.")

        scaler = torch.GradScaler(enabled=torch.cuda.is_available())

        network.train()
        for epoch in range(5):
            total_loss = 0.0
            num_batches = len(loader)
            for batch_idx, (states, policies, values) in enumerate(loader):
                states = states.to(device)
                policies = policies.to(device)
                values = values.to(device)

                optimizer.zero_grad()

                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                    policy_pred, value_pred = network(states)
                    policy_loss = torch.mean(-torch.sum(policies * torch.log_softmax(policy_pred, dim=1), dim=1))
                    value_loss = torch.nn.functional.mse_loss(value_pred.squeeze(), values)
                    loss = policy_loss + value_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

                if (batch_idx + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}")

            print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

        if (iteration+1) % 10 == 0:
            save_path = f'{model_dir}/chaturaji_iter_{iteration+1}.pth'
            torch.save(network.module.state_dict() if isinstance(network, torch.nn.DataParallel) else network.state_dict(), save_path)
        print(f"Model saved after iteration {iteration+1}")

if __name__ == "__main__":
    train()