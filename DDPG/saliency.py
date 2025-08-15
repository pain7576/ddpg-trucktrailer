import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import copy  # Import the copy module to save the best model state
from seed_utils import set_seed

def load_transitions(filename):
    """Load transition history from file"""
    replay_file = f'{filename}'
    if os.path.exists(replay_file):
        with open(replay_file, 'rb') as f:
            transitions = pickle.load(f)
        print(f"Loaded {len(transitions)} episodes of transitions from {replay_file}")
        return transitions
    else:
        print(f"No transitions file found at {replay_file}")
        return None

class SimpleRewardModel(nn.Module):
    """
    Simple feedforward neural network for reward prediction.
    Takes state and action as input, outputs a scalar reward.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(SimpleRewardModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Concatenate state and action as input
        input_dim = state_dim + action_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1)
        )

    def forward(self, state, action):
        """
        Forward pass of the reward model.
        state: torch.tensor of shape (batch_size, state_dim)
        action: torch.tensor of shape (batch_size, action_dim)
        Returns: torch.tensor of shape (batch_size, 1)
        """
        # Concatenate state and action
        state_action = torch.cat([state, action], dim=1)
        reward = self.network(state_action)
        return reward

class TransitionDataset(Dataset):
    """
    Dataset class for training the reward model.
    """
    def __init__(self, transitions):
        """
        transitions: list of tuples (obs, action, reward, obs_next, done)
        """
        self.states = []
        self.actions = []
        self.rewards = []

        for episode_transitions in transitions:
            for transition in episode_transitions:
                obs, action, reward, obs_next, done = transition
                self.states.append(obs)
                self.actions.append(action)
                self.rewards.append(reward)

        self.states = torch.tensor(np.array(self.states), dtype=torch.float32)
        self.actions = torch.tensor(np.array(self.actions), dtype=torch.float32)
        self.rewards = torch.tensor(self.rewards, dtype=torch.float32).unsqueeze(1)  # (N, 1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.rewards[idx]

def train_reward_model(reward_model, train_loader, loss_threshold=5.0, max_epochs=3000, learning_rate=1e-3):
    """
    Train the reward model until average loss is below a threshold.
    Saves a new model to the 'model/' directory every time a new best loss is found.
    """
    optimizer = optim.Adam(reward_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    best_model_state = None

    # --- CHANGED: Create the model directory if it doesn't exist ---
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    print(f"Models will be saved in the '{model_dir}/' directory.")
    # -----------------------------------------------------------------

    reward_model.train()

    print(f"Starting training. Will stop when avg loss < {loss_threshold} or after {max_epochs} epochs.")

    for epoch in range(max_epochs):
        total_loss = 0.0
        num_batches = 0

        for states, actions, true_rewards in train_loader:
            optimizer.zero_grad()
            predicted_rewards = reward_model(states, actions)
            loss = criterion(predicted_rewards, true_rewards)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{max_epochs}, Average Loss: {avg_loss:.6f}")

        # Check if this is the best model so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = copy.deepcopy(reward_model.state_dict())

            # --- CHANGED: Save the new best model to a uniquely named file ---
            save_path = os.path.join(model_dir, f'best_model_epoch_{epoch + 1}_loss_{best_loss:.6f}.pth')
            torch.save(reward_model.state_dict(), save_path)
            print(f"  -> New best model found! Loss: {best_loss:.6f}. Saved to {save_path}")
            # -----------------------------------------------------------------

        # Check for stopping condition
        if avg_loss < loss_threshold:
            print(f"\nTraining stopped: Average loss {avg_loss:.6f} is below the threshold {loss_threshold}.")
            break
    else: # This 'else' belongs to the 'for' loop, it runs if the loop completes without a 'break'
        print(f"\nTraining finished after reaching max_epochs ({max_epochs}).")

    # Load the best model's state before returning to ensure the returned object is the best one
    if best_model_state:
        print(f"\nLoading best model state (loss: {best_loss:.6f}) for post-training use.")
        reward_model.load_state_dict(best_model_state)
    else:
        print("\nWarning: Training finished, but no best model was saved.")

    print("Training process completed!")
    return reward_model

def compute_raw_gradient_saliency(reward_model, state_vec, action_vec):
    """
    Compute Raw Gradient Saliency for a single (state, action) pair.
    state_vec: torch.tensor of shape (state_dim,)
    action_vec: torch.tensor of shape (action_dim,)
    reward_model: callable taking (state, action) -> scalar tensor
    """
    state_vec = state_vec.clone().detach().requires_grad_(True)
    action_vec = action_vec.clone().detach()
    reward = reward_model(state_vec.unsqueeze(0), action_vec.unsqueeze(0))
    reward.backward()
    saliency = state_vec.grad.detach().abs()
    return saliency

def train_and_use_reward_model(stored_transitions, state_dim, action_dim):
    """
    Train reward model and compute saliency using your existing data loading pattern.
    """
    if not stored_transitions:
        print("No transitions loaded!")
        return None, None

    reward_model = SimpleRewardModel(state_dim, action_dim, hidden_dim=64)

    print("Creating dataset and training reward model...")
    dataset = TransitionDataset(stored_transitions)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

    print(f"Dataset size: {len(dataset)} transitions")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")

    # Train the model with the new stopping and saving condition
    trained_model = train_reward_model(
        reward_model,
        train_loader,
        loss_threshold=5.0, # Target loss
        max_epochs=2500     # Safety break
    )

    # --- REMOVED: The final save is no longer needed here ---
    # The best models are saved progressively inside train_reward_model
    print("\nModel training complete. Best models were saved in the 'model/' directory.")
    # --------------------------------------------------------

    print("\nComputing saliency with the best trained model...")
    all_saliencies = []
    # Ensure model is in evaluation mode for inference
    trained_model.eval()

    for episode_transitions in stored_transitions:
        for transition in episode_transitions:
            obs, action, reward, obs_next, done = transition
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action_tensor = torch.tensor(action, dtype=torch.float32)
            saliency_vec = compute_raw_gradient_saliency(trained_model, obs_tensor, action_tensor)
            all_saliencies.append(saliency_vec.cpu().numpy())

    all_saliencies = np.array(all_saliencies)
    print("\nSaliency Analysis Results:")
    print("All saliency shape:", all_saliencies.shape)
    print("Average saliency per state dimension:", np.mean(all_saliencies, axis=0))
    print("Std of saliency per state dimension:", np.std(all_saliencies, axis=0))

    return trained_model, all_saliencies

def main():
    # Adjust based on your environment's observation and action space
    SEED = 32
    set_seed(SEED)
    state_dim = 23
    action_dim = 1
    load_filename = r'C:\Users\harsh\OneDrive\Desktop\truck_trailer_DDPG\DDPG\back.pkl'

    stored_transitions = load_transitions(load_filename)
    if stored_transitions:
        train_and_use_reward_model(
            stored_transitions, state_dim, action_dim
        )

if __name__ == "__main__":
    main()