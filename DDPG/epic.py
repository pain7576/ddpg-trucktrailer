import torch
import numpy as np
import pickle
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Copy the existing classes from your code
class SimpleRewardModel(nn.Module):
    """
    Simple feedforward neural network for reward prediction.
    This class definition MUST match the one used for training.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(SimpleRewardModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
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
        state_action = torch.cat([state, action], dim=1)
        reward = self.network(state_action)
        return reward

class RewardPredictor:
    """
    A class to load a trained reward model and use it for inference.
    """
    def __init__(self, model_path, state_dim, action_dim, device='cpu'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        self.device = torch.device(device)
        print(f"Loading model from {model_path} on device: {self.device}")

        # Instantiate the model with the correct architecture
        self.model = SimpleRewardModel(state_dim, action_dim)

        # Load the saved weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Move the model to the specified device
        self.model.to(self.device)

        # Set the model to evaluation mode
        self.model.eval()

        print(f"Reward model loaded successfully from {model_path}")

    def predict_batch_rewards(self, states, actions):
        """
        Predicts rewards for a batch of states and actions.

        Args:
            states (torch.Tensor): Batch of states [batch_size, state_dim]
            actions (torch.Tensor): Batch of actions [batch_size, action_dim]

        Returns:
            torch.Tensor: Predicted rewards [batch_size, 1]
        """
        states = states.to(self.device)
        actions = actions.to(self.device)

        with torch.no_grad():
            rewards = self.model(states, actions)

        return rewards

class TransitionDataset(Dataset):
    """
    Dataset class holding transitions for evaluating reward models.
    """
    def __init__(self, transitions):
        self.states = []
        self.actions = []
        self.next_states = []

        for episode_transitions in transitions:
            for transition in episode_transitions:
                obs, action, _, obs_next, _ = transition
                self.states.append(obs)
                self.actions.append(action)
                self.next_states.append(obs_next)

        # Convert to tensors
        self.states = torch.tensor(np.array(self.states), dtype=torch.float32)
        self.actions = torch.tensor(np.array(self.actions), dtype=torch.float32)
        self.next_states = torch.tensor(np.array(self.next_states), dtype=torch.float32)

        # Ensure actions are 2D
        if self.actions.dim() == 1:
            self.actions = self.actions.unsqueeze(1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.next_states[idx]

    def get_marginal_samples(self, n_samples):
        """
        Returns random samples for states and actions from the dataset.
        """
        state_indices = torch.randint(0, len(self), (n_samples,))
        action_indices = torch.randint(0, len(self), (n_samples,))

        marginal_states = self.states[state_indices]
        marginal_actions = self.actions[action_indices]

        return marginal_states, marginal_actions

def canonically_shape_rewards_sa(
        reward_model,
        transitions,
        marginal_states,
        marginal_actions,
        gamma=0.99
):
    """
    Computes the canonically shaped rewards for a batch of transitions
    for a reward model of the form R(state, action).
    """
    states, actions, next_states = transitions
    N = states.shape[0]  # Batch size of transitions
    M = marginal_states.shape[0]  # Number of marginal samples

    # 1. R(s, a) - The base reward
    base_rewards = reward_model.predict_batch_rewards(states, actions)

    # 2. E[γ * R(s', A)] - Expectation over marginal actions A
    s_prime_repeated = next_states.repeat_interleave(M, dim=0)
    A_tiled = marginal_actions.tile(N, 1)
    r_s_prime_A = reward_model.predict_batch_rewards(s_prime_repeated, A_tiled)
    exp_r_s_prime = gamma * r_s_prime_A.view(N, M).mean(dim=1, keepdim=True)

    # 3. E[R(s, A)] - Expectation over marginal actions A
    s_repeated = states.repeat_interleave(M, dim=0)
    r_s_A = reward_model.predict_batch_rewards(s_repeated, A_tiled)
    exp_r_s = r_s_A.view(N, M).mean(dim=1, keepdim=True)

    # 4. E[γ * R(S, A)] - A single constant value over the whole batch
    S_repeated = marginal_states.repeat_interleave(M, dim=0)
    A_tiled_const = marginal_actions.tile(M, 1)
    r_S_A = reward_model.predict_batch_rewards(S_repeated, A_tiled_const)
    exp_r_const = gamma * r_S_A.mean()

    # Combine the terms
    canonical_rewards = base_rewards + exp_r_s_prime - exp_r_s - exp_r_const
    return canonical_rewards

def calculate_epic_distance(c_rewards_A, c_rewards_B):
    """
    Calculates the EPIC distance between two sets of canonically shaped rewards.
    """
    # Ensure inputs are flat
    c_rewards_A = c_rewards_A.flatten()
    c_rewards_B = c_rewards_B.flatten()

    # Center the variables by subtracting the mean
    a_centered = c_rewards_A - c_rewards_A.mean()
    b_centered = c_rewards_B - c_rewards_B.mean()

    # Calculate Pearson correlation coefficient
    covariance = (a_centered * b_centered).mean()
    std_a = torch.sqrt((a_centered**2).mean())
    std_b = torch.sqrt((b_centered**2).mean())

    # Handle the case of zero standard deviation
    if std_a < 1e-8 or std_b < 1e-8:
        # If both are constant and equal, correlation is 1, distance is 0.
        # Otherwise, they are not perfectly correlated, distance is > 0.
        # Returning 1.0 is a safe choice if they aren't identical.
        return 0.0 if torch.allclose(c_rewards_A, c_rewards_B) else 1.0
    else:
        corr = covariance / (std_a * std_b)

    # EPIC distance is 1 - correlation
    return (1 - corr).item()

def load_transitions(transitions_file):
    """
    Load transitions from file. Assumes pickle format.
    """
    try:
        with open(transitions_file, 'rb') as f:
            transitions = pickle.load(f)
        print(f"Loaded transitions from {transitions_file}")
        return transitions
    except Exception as e:
        print(f"Error loading transitions from {transitions_file}: {e}")
        return None

def main():
    # === FIX STARTS HERE ===
    # Configuration: Use the correct, full paths inside the main function.
    model_paths = {
        'ground_truth': r'C:\Users\harsh\OneDrive\Desktop\truck_trailer_DDPG\DDPG\best_smooth.pth',
        'model_back': r'C:\Users\harsh\OneDrive\Desktop\truck_trailer_DDPG\DDPG\best_back.pth',
        'model_no_back': r'C:\Users\harsh\OneDrive\Desktop\truck_trailer_DDPG\DDPG\best_no_back.pth'
    }

    transitions_file = r'C:\Users\harsh\OneDrive\Desktop\truck_trailer_DDPG\DDPG\transitions_episode_10529_replay_buffer.pkl'
    # === FIX ENDS HERE ===

    # Hyperparameters
    gamma = 0.99
    n_marginal_samples = 1000  # Number of samples for marginal distributions
    batch_size = 1000  # Process transitions in batches to avoid memory issues

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load transitions
    print("Loading transitions...")
    transitions = load_transitions(transitions_file)
    if transitions is None:
        print("Failed to load transitions. Please check the file path and format.")
        return

    # Create dataset
    print("Creating dataset...")
    dataset = TransitionDataset(transitions)
    print(f"Dataset size: {len(dataset)} transitions")

    # Infer dimensions from the dataset
    sample_state, sample_action, _ = dataset[0]
    state_dim = sample_state.shape[0]
    action_dim = sample_action.shape[0] if sample_action.dim() > 0 else 1

    print(f"Inferred dimensions - State: {state_dim}, Action: {action_dim}")

    # Load all models
    print("Loading models...")
    models = {}
    for name, path in model_paths.items():
        try:
            models[name] = RewardPredictor(path, state_dim, action_dim, device)
        except FileNotFoundError:
            print(f"Model file {path} not found. Please check the path.")
            return

    # Get marginal samples
    print("Generating marginal samples...")
    marginal_states, marginal_actions = dataset.get_marginal_samples(n_marginal_samples)
    marginal_states = marginal_states.to(device)
    marginal_actions = marginal_actions.to(device)

    # Calculate canonical rewards for each model
    print("Calculating canonical rewards...")
    canonical_rewards = {}

    # Process in batches to avoid memory issues
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for model_name, model in models.items():
        print(f"Processing {model_name}...")
        batch_canonical_rewards = []

        for batch_idx, (states, actions, next_states) in enumerate(dataloader):
            # Move batch to the correct device
            transitions_batch = (states.to(device), actions.to(device), next_states.to(device))

            c_rewards = canonically_shape_rewards_sa(
                model,
                transitions_batch,
                marginal_states,
                marginal_actions,
                gamma
            )
            # Move results to CPU to avoid filling up GPU memory
            batch_canonical_rewards.append(c_rewards.cpu())

            if batch_idx % 10 == 0:
                print(f"  Processed batch {batch_idx+1}/{len(dataloader)}")

        # Concatenate all batches
        canonical_rewards[model_name] = torch.cat(batch_canonical_rewards, dim=0)
        print(f"  {model_name} canonical rewards shape: {canonical_rewards[model_name].shape}")

    # Calculate EPIC distances
    print("\nCalculating EPIC distances...")

    ground_truth_rewards = canonical_rewards['ground_truth']

    epic_distances = {}
    for model_name in ['model_back', 'model_no_back']:
        distance = calculate_epic_distance(
            ground_truth_rewards,
            canonical_rewards[model_name]
        )
        epic_distances[model_name] = distance
        print(f"EPIC distance between ground truth and {model_name}: {distance:.6f}")

    # Compare the two models
    distance_between_models = calculate_epic_distance(
        canonical_rewards['model_back'],
        canonical_rewards['model_no_back']
    )
    epic_distances['model_back_vs_model_no_back'] = distance_between_models
    print(f"EPIC distance between model_back and model_no_back: {distance_between_models:.6f}")

    # Print summary
    print("\n" + "="*50)
    print("EPIC DISTANCE SUMMARY")
    print("="*50)
    for comparison, distance in epic_distances.items():
        print(f"{comparison}: {distance:.6f}")

    return epic_distances

if __name__ == "__main__":
    epic_distances = main()