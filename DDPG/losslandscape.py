import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle
import glob
from tqdm import tqdm
from sklearn.decomposition import PCA

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

class LossLandscapeVisualizer:
    """
    Class to visualize the loss landscape of a neural network model.
    """

    def __init__(self, model, criterion, data_loader):
        """
        Initialize the visualizer.

        Args:
            model: The trained neural network model
            criterion: Loss function (e.g., nn.MSELoss())
            data_loader: DataLoader containing the dataset
        """
        self.model = model
        self.criterion = criterion
        self.data_loader = data_loader
        self.device = next(model.parameters()).device

        # Compute and store the original loss
        self.original_loss = self.compute_loss()
        print(f"Original model loss: {self.original_loss:.6f}")

    def get_model_parameters_as_vector(self):
        """Convert model parameters to a single vector."""
        params = []
        for param in self.model.parameters():
            params.append(param.data.view(-1))
        return torch.cat(params)

    def set_model_parameters_from_vector(self, param_vector):
        """Set model parameters from a vector."""
        pointer = 0
        for param in self.model.parameters():
            num_param = param.numel()
            param.data.copy_(param_vector[pointer:pointer + num_param].view_as(param))
            pointer += num_param

    def compute_loss(self):
        """Compute the loss over the entire dataset."""
        self.model.eval()
        total_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for states, actions, true_rewards in self.data_loader:
                states, actions, true_rewards = states.to(self.device), actions.to(self.device), true_rewards.to(self.device)
                predicted_rewards = self.model(states, actions)
                loss = self.criterion(predicted_rewards, true_rewards)
                total_loss += loss.item() * states.size(0)
                num_samples += states.size(0)

        return total_loss / num_samples

    def load_saved_models_from_directory(self, model_dir='model'):
        """
        Load all saved models from the training directory and extract their parameters.

        Args:
            model_dir: Directory containing saved model files

        Returns:
            List of parameter vectors, list of corresponding losses, list of epochs
        """
        print(f"Loading saved models from '{model_dir}' directory...")

        # Find all model files in the directory
        model_pattern = os.path.join(model_dir, 'best_model_epoch_*_loss_*.pth')
        model_files = glob.glob(model_pattern)

        if not model_files:
            print(f"No model files found in '{model_dir}' directory!")
            print(f"Looking for pattern: {model_pattern}")
            return None, None, None

        print(f"Found {len(model_files)} saved models")

        # Sort files by epoch number (extract from filename)
        def extract_epoch_and_loss(filename):
            # Extract epoch and loss from filename like 'best_model_epoch_123_loss_4.567890.pth'
            basename = os.path.basename(filename)
            try:
                parts = basename.split('_')
                epoch_idx = parts.index('epoch') + 1
                loss_idx = parts.index('loss') + 1
                epoch = int(parts[epoch_idx])
                loss_str = parts[loss_idx].replace('.pth', '')
                loss = float(loss_str)
                return epoch, loss
            except (ValueError, IndexError):
                print(f"Warning: Could not parse epoch/loss from filename: {filename}")
                return 0, float('inf')

        # Sort by epoch number
        model_files.sort(key=lambda x: extract_epoch_and_loss(x)[0])

        # Load all models and extract their parameters
        all_param_vectors = []
        all_losses = []
        all_epochs = []

        # Store original model state to restore later
        original_state = self.model.state_dict().copy()

        print("Loading and analyzing saved models...")
        for model_file in tqdm(model_files):
            try:
                # Load model state
                state_dict = torch.load(model_file, map_location='cpu')
                self.model.load_state_dict(state_dict)

                # Extract parameters as vector
                param_vector = self.get_model_parameters_as_vector()
                all_param_vectors.append(param_vector.cpu().numpy())

                # Compute loss for this model
                loss = self.compute_loss()
                all_losses.append(loss)

                # Extract epoch and loss from filename
                epoch, file_loss = extract_epoch_and_loss(model_file)
                all_epochs.append(epoch)

                if abs(loss - file_loss) > 0.01:  # Small tolerance for floating point differences
                    print(f"Warning: Computed loss ({loss:.6f}) differs from filename loss ({file_loss:.6f}) for {os.path.basename(model_file)}")

            except Exception as e:
                print(f"Error loading model {model_file}: {e}")
                continue

        # Restore original model state
        self.model.load_state_dict(original_state)

        if not all_param_vectors:
            print("No models could be loaded successfully!")
            return None, None, None

        print(f"Successfully loaded {len(all_param_vectors)} models")
        print(f"Loss range: {min(all_losses):.6f} to {max(all_losses):.6f}")
        print(f"Epoch range: {min(all_epochs)} to {max(all_epochs)}")

        return np.array(all_param_vectors), np.array(all_losses), np.array(all_epochs)

    def generate_pca_directions_from_saved_models(self, model_dir='model', reference_model='best'):
        """
        Generate PCA-based directions using saved models from training.

        Args:
            model_dir: Directory containing saved model files
            reference_model: Which model to use as reference ('best', 'final', or 'mean')
                           - 'best': Model with lowest loss
                           - 'final': Last saved model (highest epoch)
                           - 'mean': Mean of all model parameters

        Returns:
            List of the top 2 principal component directions, PCA object, explained variance ratios
        """
        print(f"{'='*60}")
        print("GENERATING PCA DIRECTIONS FROM SAVED TRAINING MODELS")
        print(f"{'='*60}")

        # Load all saved models
        all_param_vectors, all_losses, all_epochs = self.load_saved_models_from_directory(model_dir)

        if all_param_vectors is None:
            print("Falling back to random perturbation method...")
            return self.generate_pca_directions_random()

        # Choose reference model
        if reference_model == 'best':
            ref_idx = np.argmin(all_losses)
            ref_params = all_param_vectors[ref_idx]
            print(f"Using best model (epoch {all_epochs[ref_idx]}, loss {all_losses[ref_idx]:.6f}) as reference")
        elif reference_model == 'final':
            ref_idx = np.argmax(all_epochs)
            ref_params = all_param_vectors[ref_idx]
            print(f"Using final model (epoch {all_epochs[ref_idx]}, loss {all_losses[ref_idx]:.6f}) as reference")
        elif reference_model == 'mean':
            ref_params = np.mean(all_param_vectors, axis=0)
            print(f"Using mean of all models as reference")
        else:
            raise ValueError("reference_model must be 'best', 'final', or 'mean'")

        # Compute parameter differences relative to reference
        param_differences = all_param_vectors - ref_params[np.newaxis, :]

        print(f"Computing PCA on {len(param_differences)} parameter difference vectors...")
        print(f"Parameter vector dimension: {param_differences.shape[1]}")

        # Remove any zero vectors (models identical to reference)
        norms = np.linalg.norm(param_differences, axis=1)
        non_zero_mask = norms > 1e-10
        if np.sum(non_zero_mask) < len(param_differences):
            print(f"Removing {len(param_differences) - np.sum(non_zero_mask)} nearly identical models")
            param_differences = param_differences[non_zero_mask]
            all_losses = all_losses[non_zero_mask]
            all_epochs = all_epochs[non_zero_mask]

        if len(param_differences) < 2:
            print("Not enough diverse models for PCA analysis. Falling back to random perturbation method...")
            return self.generate_pca_directions_random()

        # Perform PCA on parameter differences
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(param_differences)

        # Get the top 2 principal components
        pc1 = torch.tensor(pca.components_[0], dtype=torch.float32)
        pc2 = torch.tensor(pca.components_[1], dtype=torch.float32)

        # Normalize to unit vectors (should already be normalized from PCA)
        pc1 = pc1 / torch.norm(pc1)
        pc2 = pc2 / torch.norm(pc2)

        # Print PCA statistics
        explained_variance_ratio = pca.explained_variance_ratio_
        print(f"\nPCA Results from Training Models:")
        print(f"PC1 explained variance: {explained_variance_ratio[0]:.4f} ({explained_variance_ratio[0]*100:.2f}%)")
        print(f"PC2 explained variance: {explained_variance_ratio[1]:.4f} ({explained_variance_ratio[1]*100:.2f}%)")
        print(f"Cumulative variance (PC1+PC2): {np.sum(explained_variance_ratio[:2]):.4f} ({np.sum(explained_variance_ratio[:2])*100:.2f}%)")

        # Analyze correlation with loss changes
        loss_changes = all_losses - all_losses[np.argmin(all_losses)]  # Changes relative to best loss
        pc1_projections = np.dot(param_differences, pca.components_[0])
        pc2_projections = np.dot(param_differences, pca.components_[1])

        pc1_loss_corr = np.corrcoef(pc1_projections, loss_changes)[0, 1] if len(loss_changes) > 1 else 0
        pc2_loss_corr = np.corrcoef(pc2_projections, loss_changes)[0, 1] if len(loss_changes) > 1 else 0

        print(f"PC1 correlation with loss change: {pc1_loss_corr:.4f}")
        print(f"PC2 correlation with loss change: {pc2_loss_corr:.4f}")

        # Additional insights about the training trajectory
        print(f"\nTraining Trajectory Insights:")
        print(f"Total models analyzed: {len(all_param_vectors)}")
        print(f"Loss improvement: {max(all_losses) - min(all_losses):.6f}")
        print(f"Parameter space exploration: {np.mean(norms):.6f} ± {np.std(norms):.6f} (avg ± std distance from reference)")

        return [pc1, pc2], pca, explained_variance_ratio

    def generate_pca_directions_random(self, num_samples=1000, perturbation_scale=0.1):
        """
        Fallback method: Generate PCA-based directions using random perturbations.
        This is the original method, kept as a fallback.
        """
        print(f"Generating {num_samples} random parameter perturbations for PCA analysis...")

        original_params = self.get_model_parameters_as_vector()
        perturbations = []
        losses = []

        # Generate random perturbations and compute their losses
        for i in tqdm(range(num_samples), desc="Computing random perturbations"):
            # Generate random perturbation
            perturbation = torch.randn_like(original_params) * perturbation_scale
            perturbations.append(perturbation.cpu().numpy())

            # Apply perturbation and compute loss
            perturbed_params = original_params + perturbation
            self.set_model_parameters_from_vector(perturbed_params)
            loss = self.compute_loss()
            losses.append(loss)

        # Restore original parameters
        self.set_model_parameters_from_vector(original_params)

        # Convert to numpy arrays
        perturbations = np.array(perturbations)
        losses = np.array(losses)

        print(f"Loss range in perturbations: {np.min(losses):.6f} to {np.max(losses):.6f}")
        print(f"Original loss: {self.original_loss:.6f}")

        # Perform PCA on perturbations
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(perturbations)

        # Get the top 2 principal components
        pc1 = torch.tensor(pca.components_[0], dtype=torch.float32)
        pc2 = torch.tensor(pca.components_[1], dtype=torch.float32)

        # Normalize to unit vectors
        pc1 = pc1 / torch.norm(pc1)
        pc2 = pc2 / torch.norm(pc2)

        explained_variance_ratio = pca.explained_variance_ratio_
        print(f"\nPCA Results (Random Perturbations):")
        print(f"PC1 explained variance: {explained_variance_ratio[0]:.4f} ({explained_variance_ratio[0]*100:.2f}%)")
        print(f"PC2 explained variance: {explained_variance_ratio[1]:.4f} ({explained_variance_ratio[1]*100:.2f}%)")

        return [pc1, pc2], pca, explained_variance_ratio

    def create_loss_landscape_1d(self, direction, range_val=1.0, num_points=50):
        """
        Create a 1D loss landscape along a specific direction.

        Args:
            direction: Direction vector in parameter space
            range_val: Range of perturbation (e.g., -range_val to +range_val)
            num_points: Number of points to sample

        Returns:
            alphas: Array of perturbation values
            losses: Array of corresponding losses
        """
        original_params = self.get_model_parameters_as_vector()
        alphas = np.linspace(-range_val, range_val, num_points)
        losses = []

        print(f"Computing 1D loss landscape with {num_points} points...")

        for alpha in tqdm(alphas):
            # Perturb parameters
            perturbed_params = original_params + alpha * direction
            self.set_model_parameters_from_vector(perturbed_params)

            # Compute loss
            loss = self.compute_loss()
            losses.append(loss)

        # Restore original parameters
        self.set_model_parameters_from_vector(original_params)

        return alphas, np.array(losses)

    def create_loss_landscape_2d(self, direction1, direction2, range_val=1.0, num_points=25):
        """
        Create a 2D loss landscape along two directions.

        Args:
            direction1, direction2: Two direction vectors in parameter space
            range_val: Range of perturbation
            num_points: Number of points per dimension

        Returns:
            alphas: Array of perturbation values for direction 1
            betas: Array of perturbation values for direction 2
            loss_grid: 2D array of losses
        """
        original_params = self.get_model_parameters_as_vector()
        alphas = np.linspace(-range_val, range_val, num_points)
        betas = np.linspace(-range_val, range_val, num_points)
        loss_grid = np.zeros((num_points, num_points))

        print(f"Computing 2D loss landscape with {num_points}x{num_points} grid...")

        total_iterations = num_points * num_points
        with tqdm(total=total_iterations) as pbar:
            for i, alpha in enumerate(alphas):
                for j, beta in enumerate(betas):
                    # Perturb parameters
                    perturbed_params = original_params + alpha * direction1 + beta * direction2
                    self.set_model_parameters_from_vector(perturbed_params)

                    # Compute loss
                    loss = self.compute_loss()
                    loss_grid[i, j] = loss

                    pbar.update(1)

        # Restore original parameters
        self.set_model_parameters_from_vector(original_params)

        return alphas, betas, loss_grid

    def plot_1d_landscape(self, alphas, losses, title="1D Loss Landscape", pc_info=None):
        """Plot a 1D loss landscape."""
        plt.figure(figsize=(12, 7))
        plt.plot(alphas, losses, 'b-', linewidth=2, label='Loss Landscape')
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.8, linewidth=2, label='Your Trained Model')
        plt.axhline(y=self.original_loss, color='r', linestyle=':', alpha=0.6, label=f'Original Loss: {self.original_loss:.4f}')

        # Find and mark the minimum
        min_idx = np.argmin(losses)
        min_alpha = alphas[min_idx]
        min_loss = losses[min_idx]
        plt.scatter([min_alpha], [min_loss], color='green', s=100, zorder=5, label=f'Minimum: {min_loss:.4f}')

        plt.xlabel('Parameter Perturbation (α)', fontsize=12)
        plt.ylabel('Loss', fontsize=12)

        # Add PCA information to title if available
        if pc_info:
            pc_num, variance_ratio, loss_corr = pc_info
            title += f"\nPC{pc_num}: {variance_ratio:.1%} variance, {loss_corr:.3f} loss correlation"

        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()

        print(f"  Minimum loss found: {min_loss:.6f} at α={min_alpha:.3f}")
        if min_loss < self.original_loss:
            improvement = ((self.original_loss - min_loss) / self.original_loss) * 100
            print(f"  Potential improvement: {improvement:.2f}%")

    def plot_2d_landscape(self, alphas, betas, loss_grid, title="2D Loss Landscape", pca_info=None):
        """Plot a 2D loss landscape with both surface and contour plots."""
        fig = plt.figure(figsize=(18, 6))

        # Find minimum and original positions
        min_idx = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
        min_alpha = alphas[min_idx[0]]
        min_beta = betas[min_idx[1]]
        min_loss = loss_grid[min_idx]
        original_idx = (len(alphas)//2, len(betas)//2)

        # Add PCA information to title if available
        if pca_info:
            explained_var = pca_info['explained_variance']
            title += f"\nPC1: {explained_var[0]:.1%} var, PC2: {explained_var[1]:.1%} var (Total: {sum(explained_var[:2]):.1%})"

        # 3D Surface plot
        ax1 = fig.add_subplot(111, projection='3d')
        A, B = np.meshgrid(alphas, betas)
        surf = ax1.plot_surface(A, B, loss_grid.T, cmap='viridis', alpha=0.8, edgecolor='none')
        #ax1.scatter([0], [0], [self.original_loss], color='red', s=150, label='Your Model', zorder=10)
        ax1.scatter([min_alpha], [min_beta], [min_loss], color='green', s=150, label='Minimum', zorder=10)
        ax1.set_xlabel('PC1 Direction (α)')
        ax1.set_ylabel('PC2 Direction (β)')
        ax1.set_zlabel('Loss')
        ax1.set_title('3D Loss Surface')
        ax1.legend()
        fig.colorbar(surf, ax=ax1, shrink=0.6)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

        print(f"  2D Landscape Statistics:")
        print(f"  Your model loss: {self.original_loss:.6f}")
        print(f"  Minimum loss found: {min_loss:.6f} at PC1={min_alpha:.3f}, PC2={min_beta:.3f}")
        if min_loss < self.original_loss:
            improvement = ((self.original_loss - min_loss) / self.original_loss) * 100
            print(f"  Potential improvement: {improvement:.2f}%")

def visualize_pretrained_model_landscape_with_saved_models(
        model_path, data_filename, state_dim, action_dim, model_dir='model',
        range_val=0.5, num_points_1d=50, num_points_2d=25, reference_model='best'):
    """
    Visualize the loss landscape of a pretrained reward model using PCA directions from saved training models.

    Args:
        model_path: Path to your saved model (.pth file)
        data_filename: Name of the transition file
        state_dim: Dimension of the state space
        action_dim: Dimension of the action space
        model_dir: Directory containing saved models from training
        range_val: Range for parameter perturbation
        num_points_1d: Number of points for 1D landscape
        num_points_2d: Number of points per dimension for 2D landscape
        reference_model: Which model to use as reference ('best', 'final', or 'mean')
    """

    print("="*60)
    print("LOSS LANDSCAPE VISUALIZATION USING SAVED TRAINING MODELS")
    print("="*60)

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return

    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"Error: Model directory '{model_dir}' not found!")
        print("Make sure you have run the training script that saves models to this directory.")
        return

    # Load transitions
    print("Loading transition data...")
    transitions = load_transitions(data_filename)
    if transitions is None:
        print("Error: Could not load transition data!")
        return

    # Create dataset and dataloader
    print("Preparing dataset...")
    dataset = TransitionDataset(transitions)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    print(f"Dataset contains {len(dataset)} transitions")

    # Load pretrained model
    print(f"Loading pretrained model from '{model_path}'...")
    model = SimpleRewardModel(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print("Model loaded successfully!")

    # Create visualizer
    criterion = nn.MSELoss()
    visualizer = LossLandscapeVisualizer(model, criterion, data_loader)

    # Generate PCA-based directions from saved models
    directions, pca, explained_variance_ratio = visualizer.generate_pca_directions_from_saved_models(
        model_dir=model_dir, reference_model=reference_model
    )

    # Create and plot 1D landscapes
    print(f"\n{'='*50}")
    print("CREATING 1D LOSS LANDSCAPES ALONG TRAINING-BASED PRINCIPAL COMPONENTS")
    print('='*50)

    for i, direction in enumerate(directions):
        print(f"\nPrincipal Component {i+1} (from training trajectory):")
        alphas, losses = visualizer.create_loss_landscape_1d(
            direction, range_val=range_val, num_points=num_points_1d
        )

        pc_loss_corr = np.corrcoef(alphas, losses - visualizer.original_loss)[0, 1]
        pc_info = (i+1, explained_variance_ratio[i], pc_loss_corr)
        visualizer.plot_1d_landscape(
            alphas, losses,
            f"1D Loss Landscape - Training PC{i+1}",
            pc_info=pc_info
        )

    # Create and plot 2D landscape
    print(f"\n{'='*50}")
    print("CREATING 2D LOSS LANDSCAPE (Training PC1 vs Training PC2)")
    print('='*50)

    alphas, betas, loss_grid = visualizer.create_loss_landscape_2d(
        directions[0], directions[1], range_val=range_val, num_points=num_points_2d
    )

    pca_info = {
        'explained_variance': explained_variance_ratio,
        'cumulative_variance': np.sum(explained_variance_ratio[:2])
    }

    visualizer.plot_2d_landscape(
        alphas, betas, loss_grid,
        "2D Loss Landscape (Training-based Principal Components)",
        pca_info=pca_info
    )

    # Final summary
    print(f"\n{'='*60}")
    print("TRAINING-BASED LANDSCAPE ANALYSIS COMPLETE")
    print('='*60)

    min_loss_2d = np.min(loss_grid)
    landscape_range = np.max(loss_grid) - np.min(loss_grid)

    print(f"Original model loss: {visualizer.original_loss:.6f}")
    print(f"Best loss found in 2D landscape: {min_loss_2d:.6f}")
    print(f"Loss landscape range: {landscape_range:.6f}")
    print(f"Landscape roughness: {np.std(loss_grid):.6f}")

    print(f"\nTraining-based PCA Insights:")
    print(f"Top 2 PCs from training explain {np.sum(explained_variance_ratio[:2])*100:.1f}% of parameter variance")
    print(f"Training PC1 explains {explained_variance_ratio[0]*100:.1f}% of variance")
    print(f"Training PC2 explains {explained_variance_ratio[1]*100:.1f}% of variance")
    print(f"These directions represent the primary ways parameters changed during training!")

    if min_loss_2d < visualizer.original_loss:
        improvement = ((visualizer.original_loss - min_loss_2d) / visualizer.original_loss) * 100
        print(f"\nMaximum improvement found: {improvement:.2f}%")
        print("→ There might be better parameter combinations along your training trajectory!")
    else:
        print("\n→ Your final model appears to be optimal along the training trajectory directions!")

    return {
        'pca': pca,
        'explained_variance_ratio': explained_variance_ratio,
        'principal_directions': directions,
        'loss_grid': loss_grid,
        'min_loss': min_loss_2d,
        'original_loss': visualizer.original_loss
    }

# Simple usage function
def quick_visualize_with_saved_models(model_path, data_filename, state_dim, action_dim, model_dir='model'):
    """Quick visualization with default parameters using saved training models for PCA"""
    return visualize_pretrained_model_landscape_with_saved_models(
        model_path=model_path,
        data_filename=data_filename,
        state_dim=state_dim,
        action_dim=action_dim,
        model_dir=model_dir,
        range_val=0.3,              # Smaller range for detailed view
        num_points_1d=40,           # Good resolution
        num_points_2d=40,           # Reasonable computation time
        reference_model='best'       # Use best model as reference point
    )

# Example usage
if __name__ == "__main__":
    # REPLACE THESE WITH YOUR ACTUAL VALUES:
    MODEL_PATH = r"C:\Users\harsh\OneDrive\Desktop\truck_trailer_DDPG\DDPG\best_no_back.pth"     # Path to your .pth file
    DATA_FILENAME = r"C:\Users\harsh\OneDrive\Desktop\truck_trailer_DDPG\DDPG\noback.pkl"       # Your replay buffer filename
    STATE_DIM = 23                               # Your state dimension
    ACTION_DIM = 1                              # Your action dimension
    MODEL_DIR = 'model'                         # Directory with saved training models

    # Quick visualization using saved training models (RECOMMENDED)
    print("Using saved training models for PCA analysis...")
    results = quick_visualize_with_saved_models(MODEL_PATH, DATA_FILENAME, STATE_DIM, ACTION_DIM, MODEL_DIR)

    # Or use the full function with custom parameters:
    # results = visualize_pretrained_model_landscape_with_saved_models(
    #     model_path=MODEL_PATH,
    #     data_filename=DATA_FILENAME,
    #     state_dim=STATE_DIM,
    #     action_dim=ACTION_DIM,
    #     model_dir=MODEL_DIR,
    #     range_val=0.5,                    # Increase for wider view, decrease for detailed view
    #     num_points_1d=50,                 # More points = higher resolution, slower computation
    #     num_points_2d=25,                 # Reduce if computation is too slow
    #     reference_model='best'            # 'best', 'final', or 'mean'
    # )

    # The function returns useful information about the training-based analysis:
    if results:
        print(f"\nReturned results contain:")
        print(f"- PCA object based on training trajectory")
        print(f"- Principal component directions from actual training")
        print(f"- Explained variance ratios")
        print(f"- Loss landscape data along training directions")