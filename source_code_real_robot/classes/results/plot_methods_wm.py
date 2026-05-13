import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

NUM_FILES = 20
LOAD_MAX_EPOCH = 250
PLOT_MAX_EPOCH = 200 

METHODS_ORDER = ['exploration', 'exploration_path', 'improvement']
BASE_PATH_CLAS = "/Users/jakubmuller/Desktop/WORK/iwinac2026/source_code/results_data/cur_" 
BASE_PATH_CUR = "/Users/jakubmuller/Desktop/WORK/iwinac2026/source_code/results_data/cur_wm_"
BASE_PATH_NOV = "/Users/jakubmuller/Desktop/WORK/iwinac2026/source_code/results_data/cur_comb_"

COLORS = {'exploration': '#1f77b4', 'exploration_path': '#ff7f0e', 'improvement': '#2ca02c'}
LABELS = {'exploration': 'Exploration', 'exploration_path': 'Exploration-path', 'improvement': 'Utility-model'}

def get_detailed_data(base_path, num_files, max_epoch):
    results = {m: np.zeros((num_files, max_epoch + 1)) for m in METHODS_ORDER}
    
    for i in range(num_files):
        filename = f"{base_path}{i}.txt"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        epoch = int(parts[0])
                        method = parts[1]
                        if epoch <= max_epoch and method in METHODS_ORDER:
                            results[method][i, epoch] = 1
    return results

def cut_data_matrix(data_dict):
    new_data = {}
    for method, matrix in data_dict.items():
        part1 = matrix[:, :100] 
        part2 = matrix[:, 150:] 

        new_data[method] = np.concatenate((part1, part2), axis=1)
    return new_data

def plot_with_std(ax, detailed_data, event_text):
    current_max_epoch = detailed_data[METHODS_ORDER[0]].shape[1] - 1
    epochs = np.arange(current_max_epoch + 1)
    
    for method in METHODS_ORDER:
        matrix = detailed_data[method]
        
        mean_vals = np.mean(matrix, axis=0)
        std_vals = np.std(matrix, axis=0)
        
        mean_smoothed = pd.Series(mean_vals).rolling(window=20, center=True, min_periods=1, win_type='gaussian').mean(std=5)
        std_smoothed = pd.Series(std_vals).rolling(window=20, center=True, min_periods=1, win_type='gaussian').mean(std=5)
        
        ax.plot(epochs, mean_smoothed, label=LABELS[method], color=COLORS[method], linewidth=2)
        
        lower_bound = np.clip(mean_smoothed - std_smoothed, 0, 1)
        upper_bound = np.clip(mean_smoothed + std_smoothed, 0, 1)
        
        ax.fill_between(epochs, lower_bound, upper_bound, color=COLORS[method], alpha=0.15)

    ax.tick_params(axis='both', which='major', labelsize=16)

    ax.set_xticks(np.arange(0, PLOT_MAX_EPOCH + 1, 25))
    ax.set_yticks(np.arange(0, 1.25, 0.25))
    ax.set_ylim(-0.1, 1.025)
    ax.grid(True, linestyle='-', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.axvline(x=100, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.text(65, 1.05, event_text, fontsize=16, color='black')

data_class_detailed = get_detailed_data(BASE_PATH_CLAS, NUM_FILES, LOAD_MAX_EPOCH)
data_cur_detailed = get_detailed_data(BASE_PATH_CUR, NUM_FILES, LOAD_MAX_EPOCH)
data_nov_detailed = get_detailed_data(BASE_PATH_NOV, NUM_FILES, LOAD_MAX_EPOCH)

data_class_detailed = cut_data_matrix(data_class_detailed)
data_cur_detailed = cut_data_matrix(data_cur_detailed)
data_nov_detailed = cut_data_matrix(data_nov_detailed)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

plot_with_std(ax1, data_class_detailed, "Goal pos. change")
plot_with_std(ax2, data_cur_detailed, "World Model change")
plot_with_std(ax3, data_nov_detailed, "WM & Goal pos. change")

# Nastavení osy x pro všechny grafy
for ax in (ax1, ax2, ax3):
    ax.set_xlabel("Epochs", fontsize=16)
    ax.set_xlim(0, PLOT_MAX_EPOCH)

fig.supylabel("Mean Activity Level", fontsize=16, x=0.01)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False, fontsize=16)

plt.tight_layout()
plt.savefig('plot_ijcr_activity_plot_wm.png', dpi=300, bbox_inches='tight')
plt.show()