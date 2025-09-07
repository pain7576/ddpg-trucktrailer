import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class HistoryAnimator:
    def __init__(self, folder_path, chunk_size=50):
        self.folder_path = folder_path
        self.chunk_size = chunk_size
        # list and sort pickles
        self.files = sorted(
            [f for f in os.listdir(folder_path) if f.endswith('.pkl')],
            key=lambda f: os.path.getctime(os.path.join(folder_path, f))
        )
        # build chunks of chunk_size files
        self.chunks = [self.files[i:i+chunk_size] for i in range(0, len(self.files), chunk_size)]
        self.cmap = plt.get_cmap('viridis')

    def load_chunk(self, chunk_files):
        """Load a chunk of files and return concatenated x2, y2 and per-point normalized row (0..1)"""
        xs = []
        ys = []
        rows_norm = []
        oris = []

        for fname in chunk_files:
            path = os.path.join(self.folder_path, fname)
            try:
                with open(path, 'rb') as f:
                    episode_data = pickle.load(f)
            except Exception as e:
                print(f"Warning: failed to load {fname}: {e}")
                continue

            states = np.array(episode_data.get('states', []))
            if states.size == 0:
                continue

            # x2 and y2 columns (4 and 5)
            x2 = states[:, 4]
            y2 = states[:, 5]
            ori = states[:, 1]
            n = len(x2)

            if n == 0:
                continue

            # normalize row index to [0,1] per episode (preserves "progress in episode")
            if n == 1:
                norm = np.array([0.0], dtype=float)
            else:
                norm = np.arange(n, dtype=float) / float(n - 1)

            xs.append(x2)
            ys.append(y2)
            oris.append(ori)
            rows_norm.append(norm)

        if not xs:
            return np.array([]), np.array([]), np.array([])

        return np.concatenate(xs), np.concatenate(ys), np.concatenate(rows_norm)

    def animate(self, save_path=None, fps=1):
        if len(self.chunks) == 0:
            raise RuntimeError("No .pkl files found in folder.")

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim([-40, 40])
        ax.set_ylim([-40, 40])
        ax.set_xlabel("x2")
        ax.set_ylabel("y2")
        ax.grid(True)

        # Create an initial scatter (empty). Use vmin/vmax 0..1 because we normalize per-episode to [0,1].
        scatter = ax.scatter([], [], c=[], cmap='viridis', vmin=0.0, vmax=1.0, s=10)
        # Create a ScalarMappable for a stable colorbar (0..1 = relative progress)
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0.0, vmax=1.0))
        sm.set_array([])  # necessary for colorbar
        cbar = fig.colorbar(sm, ax=ax, label='relative row (0 = start, 1 = end)')

        def update(frame_idx):
            chunk = self.chunks[frame_idx]
            x, y, rows_norm = self.load_chunk(chunk)

            if x.size == 0:
                # empty chunk: clear
                scatter.set_offsets(np.empty((0, 2)))
                scatter.set_array(np.array([]))
            else:
                offsets = np.column_stack((x, y))
                scatter.set_offsets(offsets)
                # rows_norm must be a 1D float array with same length as offsets
                scatter.set_array(rows_norm)

            start_idx = frame_idx * self.chunk_size + 1
            end_idx = start_idx + len(chunk) - 1
            ax.set_title(f"Files {start_idx} to {end_idx} (chunk {frame_idx+1}/{len(self.chunks)})")
            return scatter,

        anim = FuncAnimation(fig, update, frames=len(self.chunks), blit=False, repeat=False)

        if save_path:
            # requires ffmpeg installed if saving to mp4
            anim.save(save_path, writer='ffmpeg', fps=fps)
            print(f"Saved animation to {save_path}")
        else:
            plt.show()


# Example usage
if __name__ == "__main__":
    folder = r"C:\Users\harsh\OneDrive\Desktop\truck_trailer_DDPG\Results\final_stage2_smooth\episode_replays"
    animator = HistoryAnimator(folder_path=folder, chunk_size=50)
    animator.animate()  # or animator.animate("episodes_chunks.mp4", fps=1)
