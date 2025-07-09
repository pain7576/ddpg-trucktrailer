import numpy as np
import matplotlib.pyplot as plt

def Plot_learning_curve(x, scores, success, figure_file):
    running_avg_score = np.zeros(len(scores))
    running_avg_success = np.zeros(len(success))

    for i in range(len(running_avg_score)):
        running_avg_score[i] = np.mean(scores[max(0, i-100):(i+1)])
        running_avg_success[i] = np.mean(success[max(0, i-100):(i+1)])

    fig, ax1 = plt.subplots()

    # Plot scores on left y-axis
    ax1.plot(x, running_avg_score, color='blue', label='Score (Running Avg)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a second y-axis for success rate
    ax2 = ax1.twinx()
    ax2.plot(x, running_avg_success, color='green', label='Success Rate (Running Avg)')
    ax2.set_ylabel('Success Rate', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Title and grid
    plt.title('Running Average of Scores and Success Rate')
    fig.tight_layout()
    plt.grid(True)
    plt.savefig(figure_file)
    plt.close()