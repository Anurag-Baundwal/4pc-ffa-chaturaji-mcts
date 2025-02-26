#train.py (CORRECTED)

import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from self_play import SelfPlay
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
        state_tensor = board_to_tensor(board, device)
        policy_tensor = self._policy_to_tensor(policy, board)
        return state_tensor, policy_tensor, torch.tensor(value, dtype=torch.float32)  # value is now ALWAYS a float

    def _policy_to_tensor(self, policy, board):
        legal_moves = board.get_psuedo_legal_moves(board.current_player)  # Corrected: Use board.current_player
        policy_tensor = torch.zeros(4096)  # 8x8x64 = 4096 possible moves
        for move, prob in policy.items():
            idx = move_to_index(move)
            policy_tensor[idx] = prob
        return policy_tensor

def train():
    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create timestamped model directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Get timestamp
    model_dir = f'/content/drive/MyDrive/models/run_{timestamp}' # Include timestamp in directory name
    os.makedirs(model_dir, exist_ok=True)
    print(f"Model directory created: {model_dir}") # Optional: Print the directory name

    network = ChaturajiNN().to(device)
    optimizer = optim.Adam(network.parameters(), lr=0.001, weight_decay=1e-4)
    self_play = SelfPlay(network, simulations_per_move=50)

    # Load existing model if available (path needs update if you want to load from a specific run)
    # if os.path.exists("model.pth"): # Original path - consider if you want to load from a previous run
    #     network.load_state_dict(torch.load("model.pth", map_location=device))

    # Load existing model from the *new* timestamped directory (if you want to load from the *previous* run)
    # last_run_dir = sorted([d for d in os.listdir('/content/drive/MyDrive/models') if d.startswith('run_')])[-1] # Get last run dir
    # if last_run_dir:
    #     model_path = os.path.join('/content/drive/MyDrive/models', last_run_dir, 'chaturaji_iter_50.pth') # Assuming you want to load the last iteration's model
    #     if os.path.exists(model_path):
    #         network.load_state_dict(torch.load(model_path, map_location=device))
    #         print(f"Loaded model from: {model_path}")
    #     else:
    #         print(f"No model found in last run directory at: {model_path}. Starting from scratch.")
    # else:
    #     print("No previous runs found. Starting from scratch.")


    # Training loop
    for iteration in range(50):
        print(f"---------- ITERATION {iteration+1} ----------")

        # Generate games and populate the buffer
        print(f"Generating games...")
        self_play.buffer.clear()  # Clear the buffer at the start of each iteration
        for _ in range(50):  # Parallel self-play. # Increased to 50
            self_play.generate_game()  # Directly modifies self_play.buffer

        # Create dataset from the buffer
        dataset = ChessDataset(self_play.buffer)  # Use the buffer directly!
        loader = DataLoader(dataset, batch_size=4096, shuffle=True)

        print(f"Generated {len(dataset)} data points from games.")

        scaler = torch.GradScaler(enabled=torch.cuda.is_available()) # Explicitly enable only when CUDA is available

        # Training epoch
        network.train()
        for epoch in range(25):  # 50 epochs per iteration (was 5)
            total_loss = 0.0
            num_batches = len(loader)
            for batch_idx, (states, policies, values) in enumerate(loader):
                states = states.to(device)
                states = states.squeeze(1)
                policies = policies.to(device)
                values = values.to(device)

                # Forward pass
                optimizer.zero_grad()

                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
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

        # Save every 1 iterations
        if (iteration+1) % 1 == 0:
            save_path = os.path.join(model_dir, f'chaturaji_iter_{iteration+1}.pth') # Save in the timestamped directory
            torch.save(network.state_dict(), save_path)
            print(f"Model saved after iteration {iteration+1} to: {save_path}") # Print the full save path


if __name__ == "__main__":
    train()